"""
T5 text encoder test.

Tokenises a short prompt and runs the encoder forward pass on Neuron.

Usage:
    python tests/test_t5.py \
        --checkpoint-dir /home/ubuntu/Wan2.2/Wan2.2-TI2V-5B

Expects:
    <checkpoint_dir>/models_t5_umt5-xxl-enc-bf16.pth
    <checkpoint_dir>/google/umt5-xxl/          (tokenizer)
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir",
                   default="/home/ubuntu/Wan2.2-TI2V-5B",
                   help="Path to TI2V-5B checkpoint directory")
    p.add_argument("--device", default="neuron",
                   help="'cpu' or 'neuron' (default: neuron)")
    p.add_argument("--text-len", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()

    from wan.modules.t5 import T5EncoderModel

    ckpt_dir = args.checkpoint_dir
    t5_pth = os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    tok_path = os.path.join(ckpt_dir, "google/umt5-xxl")

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Loading T5 from {t5_pth}")

    encoder = T5EncoderModel(
        text_len=args.text_len,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=t5_pth,
        tokenizer_path=tok_path,
    )
    print("T5EncoderModel loaded.")

    prompts = [
        "A cat sitting on a windowsill watching rain fall.",
        "Fireworks over a city at night.",
    ]
    print(f"Encoding {len(prompts)} prompts...")
    context = encoder(prompts, device=device)

    for i, (prompt, ctx) in enumerate(zip(prompts, context)):
        print(f"  [{i}] '{prompt[:40]}...'  →  context shape: {ctx.shape}  dtype: {ctx.dtype}")

    print("\nPASS — T5 encoder forward pass completed.")


if __name__ == "__main__":
    main()
