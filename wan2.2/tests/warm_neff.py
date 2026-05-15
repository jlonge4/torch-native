"""
Iterative NEFF warming for WanTI2V on Neuron.

The Neuron compiler compiles a unique NEFF per (model, input shape). On first
call for a new shape the process blocks until compilation completes. With
NEURON_LAUNCH_BLOCKING=1 the compilation is synchronous, so each step here
must complete before the next starts.

The 12 NEFFs currently cached cover:
    size=(480,256), frame_num=21, steps=20 (the confirmed working config)

Run this script to warm NEFFs for a new target shape BEFORE running the full
pipeline — otherwise the full pipeline hangs waiting for compilation.

Usage:
    NEURON_LAUNCH_BLOCKING=1 python tests/warm_neff.py --frame-num 41

Once complete, run the full pipeline:
    python run/wan22_generate.py --frames 41 --steps 20
"""
import argparse
import os
import sys
import math

# MUST be set before any torch/neuron imports
os.environ.setdefault('NEURON_LAUNCH_BLOCKING', '1')

sys.path.insert(0, '/home/ubuntu/Wan2.2')

import torch
import torch_neuronx


def p(msg):
    print(msg, flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame-num', type=int, default=41,
                    help='Target frame_num (must be 4n+1)')
    ap.add_argument('--size', default='480x256', help='WxH')
    ap.add_argument('--checkpoint-dir',
                    default='/home/ubuntu/Wan2.2/Wan2.2-TI2V-5B')
    ap.add_argument('--steps', type=int, default=1,
                    help='Denoising steps to warm (1 = single-step compile)')
    return ap.parse_args()


def main():
    args = parse_args()
    w, h = map(int, args.size.split('x'))
    F = args.frame_num
    assert (F - 1) % 4 == 0, f'frame_num must be 4n+1, got {F}'

    # Compute expected latent / seq shapes (vae_stride=(4,16,16), patch=(1,2,2))
    t_lat = (F - 1) // 4 + 1
    h_lat = h // 16
    w_lat = w // 16
    seq_len = t_lat * h_lat * w_lat // 4   # patch_size 2×2
    z_dim = 48

    p(f'Target: frame_num={F}, size=({w}x{h})')
    p(f'Latent: [{z_dim}, {t_lat}, {h_lat}, {w_lat}]  seq_len={seq_len}')
    p(f'NEURON_LAUNCH_BLOCKING={os.environ.get("NEURON_LAUNCH_BLOCKING")}')

    # ── Step 1: Load WanModel directly onto Neuron ──────────────────────────
    p('\n[1/3] Loading WanModel onto Neuron...')
    from wan.modules.model import WanModel
    model = WanModel.from_pretrained(args.checkpoint_dir)
    model.eval().requires_grad_(False).to(torch.bfloat16).to('neuron:0')
    p('      Model on neuron:0')

    device = torch.device('neuron:0')

    # ── Step 2: Single DiT forward pass — forces NEFF compilation ───────────
    p(f'\n[2/3] DiT forward (seq_len={seq_len}) — will compile NEFF...')
    x = [torch.randn(z_dim, t_lat, h_lat, w_lat, dtype=torch.bfloat16, device=device)]
    t = torch.tensor([500.0], device=device, dtype=torch.float32)
    context = [torch.randn(512, 4096, dtype=torch.bfloat16, device=device)]

    with torch.no_grad():
        out = model(x, t=t, context=context, seq_len=seq_len)
    p(f'      DiT OK — output shape: {out[0].shape}')
    p(f'      Fallback ops: {torch_neuronx.get_fallback_ops()}')

    # ── Step 3: Full pipeline single step (warms T5 + VAE NEFFs too) ────────
    p(f'\n[3/3] Full pipeline single step (frame_num={F})...')
    del model  # free HBM before loading full pipeline

    from wan.configs.wan_ti2v_5B import ti2v_5B
    from wan.textimage2video import WanTI2V
    import torchvision

    pipeline = WanTI2V(
        config=ti2v_5B,
        checkpoint_dir=args.checkpoint_dir,
        device_id=0,
        init_on_cpu=True,
        convert_model_dtype=True,
    )

    video = pipeline.generate(
        input_prompt='A red panda playing in bamboo forest, cinematic, 4k',
        size=(w, h),
        frame_num=F,
        sampling_steps=1,
        guide_scale=5.0,
        seed=42,
    )

    if video is not None:
        frames = ((video.permute(1, 2, 3, 0).clamp(-1, 1) + 1) / 2 * 255).byte()
        out_path = f'/home/ubuntu/warm_neff_{w}x{h}_{F}f.mp4'
        torchvision.io.write_video(out_path, frames, fps=16)
        p(f'\nWARM COMPLETE — {out_path}  shape={video.shape}')
        p('NEFFs for this shape are now cached. Run full pipeline next.')
    else:
        p('\nWARNING: generate() returned None')


if __name__ == '__main__':
    main()
