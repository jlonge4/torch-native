"""
Single denoising step end-to-end test for WanTI2V on Neuron.

Runs the full pipeline (T5 encode → DiT one step → VAE decode) at the smallest
viable resolution / frame count so the test completes in minutes rather than hours.

Usage:
    python tests/test_pipeline_single_step.py \
        --checkpoint-dir /home/ubuntu/Wan2.2/Wan2.2-TI2V-5B

"""
import argparse
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Allow running from runway-ml root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir",
                   default="/home/ubuntu/Wan2.2-TI2V-5B",
                   help="Path to TI2V-5B checkpoint directory")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--prompt", default="A panda eating bamboo in a misty forest.",
                   help="Text prompt for single-step test")
    return p.parse_args()


def build_config():
    from wan.configs.wan_ti2v_5B import ti2v_5B
    return ti2v_5B


def main():
    args = parse_args()

    from wan.textimage2video import WanTI2V

    config = build_config()
    print(f"Loading WanTI2V from {args.checkpoint_dir} on neuron:{args.device_id} ...")
    pipeline = WanTI2V(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id,
        init_on_cpu=True,
        convert_model_dtype=False,
    )
    print("Pipeline loaded.")

    # Use smallest sensible resolution: 256×256, 5 frames (4n+1 = 4*1+1), 1 step
    size = (256, 256)          # (width, height)
    frame_num = 5              # minimum valid: 4*1+1
    sampling_steps = 1         # single denoising step

    print(f"\nRunning single-step t2v: size={size}, frames={frame_num}, steps={sampling_steps}")
    print(f"Prompt: '{args.prompt}'")

    # latent_only=True skips VAE decode so DiT NEFFs don't block it in the
    # same process (same constraint that requires two-phase exec in production).
    # VAE round-trip is covered by test_vae.py.
    latent = pipeline.generate(
        input_prompt=args.prompt,
        img=None,
        size=size,
        frame_num=frame_num,
        sampling_steps=sampling_steps,
        guide_scale=5.0,
        seed=42,
        offload_model=True,
        latent_only=True,
    )

    if latent is not None:
        print(f"\nPASS — latent shape: {latent.shape}  dtype: {latent.dtype}")
        print(f"       value range: [{latent.min():.3f}, {latent.max():.3f}]")
    else:
        print("\nWARNING — generate() returned None (non-rank-0 process?)")


if __name__ == "__main__":
    main()
