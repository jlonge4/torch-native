import argparse
import os
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
    p.add_argument("--checkpoint-dir", default="/home/ubuntu/Wan2.2/Wan2.2-TI2V-5B")
    p.add_argument("--tp-degree", type=int, default=1,
                   help="Tensor parallel degree. Use 4 for all NeuronCores on trn2.3xlarge. "
                        "Must be launched with: torchrun --nproc-per-node TP_DEGREE")
    p.add_argument("--compile", action="store_true",
                   help="Apply torch.compile(backend='neuron') to the DiT model.")
    return p.parse_args()


def main():
    args = parse_args()
    w, h = map(int, args.size.split("x"))

    img = None
    if args.image == 'example':
        args.image = f"{args.checkpoint_dir}/examples/i2v_input.JPG"
    if args.image:
        img = Image.open(args.image).convert('RGB')
        print(f"Using input image: {args.image} ({img.width}x{img.height})", flush=True)

    print("Loading WanTI2V...", flush=True)
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
    video = pipeline.generate(
        input_prompt=args.prompt,
        img=img,
        size=(w, h),
        max_area=w * h,
        frame_num=args.frames,
        shift=shift,
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
        offload_model=(args.tp_degree == 1),
    )
    denoise_time = time.time() - t0
    f_lat = (args.frames - 1) // 4 + 1
    # BF16 peak: 158 TFLOP/s per NeuronCore; TP uses tp_degree cores
    peak_flops = 158e12 * args.tp_degree
    tokens = f_lat * (h // 32) * (w // 32)
    mfu = (6 * 5e9 * tokens * args.steps) / (denoise_time * peak_flops) * 100
    print(f"Denoise time: {denoise_time:.1f}s | tokens: {tokens} | MFU: {mfu:.2f}% (TP={args.tp_degree})", flush=True)

    if video is not None:
        save_video(video, save_file=args.output, fps=args.fps)
        print(f"Saved to {args.output}", flush=True)

    if args.compile:
        try:
            import torch_neuronx
            fallback = torch_neuronx.get_fallback_ops()
            if fallback:
                print(f"[rank 0] Neuron CPU fallback ops: {fallback}", flush=True)
            else:
                print("[rank 0] No CPU fallback ops detected.", flush=True)
        except Exception as e:
            print(f"[rank 0] Could not get fallback ops: {e}", flush=True)


if __name__ == "__main__":
    main()
