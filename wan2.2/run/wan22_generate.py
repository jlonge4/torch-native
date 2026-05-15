import argparse
import sys
sys.path.insert(0, '/home/ubuntu/Wan2.2')

import torch
import torchvision
from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.textimage2video import WanTI2V


def save_video(video, save_file, fps=16):
    """Save [C, T, H, W] float32 tensor in [-1,1] to mp4 via torchvision."""
    frames = ((video.permute(1, 2, 3, 0).clamp(-1, 1) + 1) / 2 * 255).byte()
    torchvision.io.write_video(save_file, frames, fps=fps)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="A red panda playing in bamboo forest, cinematic, 4k")
    p.add_argument("--size", default="480x256", help="WxH e.g. 480x256")
    p.add_argument("--frames", type=int, default=21)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="/home/ubuntu/wan22_output.mp4")
    p.add_argument("--checkpoint-dir", default="/home/ubuntu/Wan2.2/Wan2.2-TI2V-5B")
    return p.parse_args()


def main():
    args = parse_args()
    w, h = map(int, args.size.split("x"))

    print("Loading WanTI2V...", flush=True)
    pipeline = WanTI2V(
        config=ti2v_5B,
        checkpoint_dir=args.checkpoint_dir,
        device_id=0,
        init_on_cpu=True,
        convert_model_dtype=True,
    )
    print("Loaded.", flush=True)

    video = pipeline.generate(
        input_prompt=args.prompt,
        img=None,
        size=(w, h),
        frame_num=args.frames,
        sampling_steps=args.steps,
        guide_scale=5.0,
        seed=args.seed,
        offload_model=True,
    )

    if video is not None:
        save_video(video, save_file=args.output, fps=args.fps)
        print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
