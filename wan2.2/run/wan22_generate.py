import argparse
import os
import subprocess
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio
import numpy as np
import torch
from PIL import Image
from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.textimage2video import WanTI2V


def save_video(video, save_file, fps=16):
    """Save [C, T, H, W] float32 tensor in [-1,1] to mp4 via imageio."""
    frames = ((video.permute(1, 2, 3, 0).clamp(-1, 1) + 1) / 2 * 255).byte().numpy()
    with imageio.get_writer(save_file, fps=fps, format='ffmpeg', codec='libx264') as writer:
        for frame in frames:
            writer.append_data(frame)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="A red panda playing in bamboo forest, cinematic, 4k")
    p.add_argument("--image", default=None,
                   help="Input image for i2v mode. Use 'example' for the bundled example image.")
    p.add_argument("--size", default="832x480", help="WxH e.g. 832x480")
    p.add_argument("--frames", type=int, default=21)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--shift", type=float, default=None,
                   help="Noise schedule shift. Defaults to 5.0.")
    p.add_argument("--guide-scale", type=float, default=5.0)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="/home/ubuntu/wan22_output.mp4")
    p.add_argument("--checkpoint-dir", default="/home/ubuntu/Wan2.2-TI2V-5B")
    p.add_argument("--tp-degree", type=int, default=1,
                   help="Tensor parallel degree. Use 4 for all NeuronCores on trn2.3xlarge. "
                        "Must be launched with: torchrun --nproc-per-node TP_DEGREE")
    p.add_argument("--compile", action="store_true",
                   help="Apply torch.compile(backend='neuron') to the DiT model.")
    # Internal flags for two-phase execution (TP mode)
    p.add_argument("--phase", choices=["denoise", "vae"], default=None,
                   help="Internal: run only denoise or vae phase (used by two-phase TP mode)")
    p.add_argument("--latent-path", default=None,
                   help="Internal: path to saved latent tensor for vae phase")
    return p.parse_args()


def run_denoise_phase(args):
    """TP denoise phase: runs under torchrun, saves latent to disk on rank 0."""
    w, h = map(int, args.size.split("x"))

    img = None
    if args.image == 'example':
        args.image = f"{args.checkpoint_dir}/examples/i2v_input.JPG"
    if args.image:
        img = Image.open(args.image).convert('RGB')
        print(f"Using input image: {args.image} ({img.width}x{img.height})", flush=True)

    print("Loading WanTI2V (TP denoise)...", flush=True)
    pipeline = WanTI2V(
        config=ti2v_5B,
        checkpoint_dir=args.checkpoint_dir,
        device_id=0,
        init_on_cpu=True,
        convert_model_dtype=True,
        tp_degree=args.tp_degree,
        compile_model=args.compile,
    )
    print("Loaded.", flush=True)

    shift = args.shift if args.shift is not None else 5.0

    t0 = time.time()
    latent = pipeline.generate(
        input_prompt=args.prompt,
        img=img,
        size=(w, h),
        max_area=w * h,
        frame_num=args.frames,
        shift=shift,
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
        offload_model=False,
        latent_only=True,
    )
    denoise_time = time.time() - t0

    # Only rank 0 has the latent and should print/save
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        f_lat = (args.frames - 1) // 4 + 1
        # BF16 peak: 158 TFLOP/s per NeuronCore; TP uses tp_degree cores
        peak_flops = 158e12 * args.tp_degree
        tokens = f_lat * (h // 32) * (w // 32)
        mfu = (6 * 5e9 * tokens * args.steps) / (denoise_time * peak_flops) * 100
        print(f"Denoise time: {denoise_time:.1f}s | tokens: {tokens} | MFU: {mfu:.2f}% (TP={args.tp_degree})", flush=True)

        if args.compile:
            try:
                import torch_neuronx
                fallback = torch_neuronx.get_fallback_ops()
                if fallback:
                    print(f"[neuron] CPU fallback ops: {fallback}", flush=True)
                else:
                    print("[neuron] No CPU fallback ops.", flush=True)
            except Exception as e:
                print(f"[neuron] Could not get fallback ops: {e}", flush=True)

        torch.save(latent.cpu(), args.latent_path)
        print(f"Latent saved to {args.latent_path}", flush=True)


def run_vae_phase(args):
    """VAE decode phase: runs single-process on fresh HBM (no DiT NEFFs resident)."""
    w, h = map(int, args.size.split("x"))

    print("Loading VAE...", flush=True)
    from wan.modules.vae2_2 import Wan2_2_VAE
    vae = Wan2_2_VAE(
        vae_pth=os.path.join(args.checkpoint_dir, ti2v_5B.vae_checkpoint),
        device=torch.device('neuron:0'),
    )
    print("VAE loaded.", flush=True)

    latent = torch.load(args.latent_path)
    x0 = [latent.to(vae.device)]

    print("Running VAE decode...", flush=True)
    videos = vae.decode(x0)
    save_video(videos[0], save_file=args.output, fps=args.fps)
    print(f"Saved to {args.output}", flush=True)


def main():
    args = parse_args()

    if args.phase == "denoise":
        run_denoise_phase(args)
        return

    if args.phase == "vae":
        run_vae_phase(args)
        return

    # --- Two-phase execution for TP mode ---
    latent_path = args.latent_path or "/tmp/wan_tp_latent.pt"
    script = os.path.abspath(__file__)
    base_cmd = [
        "--checkpoint-dir", args.checkpoint_dir,
        "--size", args.size,
        "--frames", str(args.frames),
        "--steps", str(args.steps),
        "--guide-scale", str(args.guide_scale),
        "--fps", str(args.fps),
        "--seed", str(args.seed),
        "--output", args.output,
        "--latent-path", latent_path,
        "--tp-degree", str(args.tp_degree),
    ]
    if args.prompt:
        base_cmd += ["--prompt", args.prompt]
    if args.image:
        base_cmd += ["--image", args.image]
    if args.shift:
        base_cmd += ["--shift", str(args.shift)]
    if args.compile:
        base_cmd += ["--compile"]

    t_total = time.time()

    if args.tp_degree > 1:
        print(f"=== Phase 1: TP Denoise (torchrun --nproc-per-node {args.tp_degree}) ===", flush=True)
        denoise_cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc-per-node={args.tp_degree}",
            script,
        ] + base_cmd + ["--phase", "denoise"]
    else:
        print("=== Phase 1: Denoise ===", flush=True)
        denoise_cmd = [sys.executable, script] + base_cmd + ["--phase", "denoise"]

    subprocess.run(denoise_cmd, check=True)

    print("=== Phase 2: VAE decode (fresh process, no DiT NEFFs in HBM) ===", flush=True)
    vae_cmd = [sys.executable, script] + base_cmd + ["--phase", "vae"]
    subprocess.run(vae_cmd, check=True)

    print(f"Total time: {time.time() - t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
