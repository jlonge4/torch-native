import sys
sys.path.insert(0, '/home/ubuntu/Wan2.2')

import torch
import av
import numpy as np
import torchvision
from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.textimage2video import WanTI2V


def save_video_pyav(tensor, save_file, fps=24):
    """Save video tensor [C, T, H, W] to mp4 using PyAV (avoids imageio quality kwarg issue)."""
    # tensor: [C, T, H, W] float32 in [-1, 1]
    tensor = tensor.clamp(-1, 1)
    # Add batch dim for make_grid: [1, C, T, H, W]
    tensor = tensor.unsqueeze(0)
    # Build [T, H, W, 3] uint8
    frames_t = torch.stack([
        torchvision.utils.make_grid(u, nrow=8, normalize=True, value_range=(-1, 1))
        for u in tensor.unbind(2)
    ], dim=1).permute(1, 2, 3, 0)
    frames = (frames_t * 255).byte().cpu().numpy()  # [T, H, W, 3]

    container = av.open(save_file, mode='w')
    stream = container.add_stream('h264', rate=fps)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = 'yuv420p'
    for frame_np in frames:
        frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


config = ti2v_5B

print("Loading WanTI2V...", flush=True)
pipeline = WanTI2V(
    config=config,
    checkpoint_dir='/home/ubuntu/Wan2.2/Wan2.2-TI2V-5B',
    device_id=0,
    init_on_cpu=True,
    convert_model_dtype=True,
)
print("Loaded.", flush=True)

video = pipeline.generate(
    input_prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    img=None,
    size=(256, 256),
    frame_num=5,
    sampling_steps=1,
    guide_scale=5.0,
    seed=42,
)

if video is not None:
    save_video_pyav(video, save_file='/home/ubuntu/wan22_output.mp4', fps=24)
    print("Saved to /home/ubuntu/wan22_output.mp4", flush=True)
