"""
Single denoising step end-to-end test for WanTI2V on Neuron.

Runs the full pipeline (T5 encode → DiT one step → VAE decode) at the smallest
viable resolution / frame count so the test completes in minutes rather than hours.

Usage:
    python tests/test_pipeline_single_step.py \
        --checkpoint-dir /home/ubuntu/Wan2.2/Wan2.2-TI2V-5B

The script imports WanTI2V from the installed wan package (after patches have
been applied with tests/apply_patches.sh). To run against the local patch files
directly, set WAN_PKG environment variable to the wan package parent directory:
    WAN_PKG=/home/ubuntu/Wan2.2 python tests/test_pipeline_single_step.py ...
"""
import argparse
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Allow running from runway-ml root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

wan_pkg = os.environ.get("WAN_PKG", "/home/ubuntu/Wan2.2")
if wan_pkg not in sys.path:
    sys.path.insert(0, wan_pkg)

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir",
                   default="/home/ubuntu/Wan2.2/Wan2.2-TI2V-5B",
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

    video = pipeline.generate(
        input_prompt=args.prompt,
        img=None,
        size=size,
        frame_num=frame_num,
        sampling_steps=sampling_steps,
        guide_scale=5.0,
        seed=42,
        offload_model=True,
    )

    if video is not None:
        print(f"\nPASS — output video shape: {video.shape}  dtype: {video.dtype}")
        print(f"       value range: [{video.min():.3f}, {video.max():.3f}]")
    else:
        print("\nWARNING — generate() returned None (non-rank-0 process?)")


if __name__ == "__main__":
    main()
