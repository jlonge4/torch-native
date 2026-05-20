"""
VAE encode → decode round-trip test.

Usage:
    python tests/test_vae.py --checkpoint-dir /path/to/Wan2.2-TI2V-5B
    python tests/test_vae.py --checkpoint-dir /path/to/... --device neuron
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from wan.configs.wan_ti2v_5B import ti2v_5B


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default="/home/ubuntu/Wan2.2-TI2V-5B",
                   help="Path to TI2V-5B checkpoint directory")
    p.add_argument("--device", default="cpu",
                   help="'cpu' or 'neuron' (default: cpu)")
    return p.parse_args()


def main():
    args = parse_args()

    from wan.modules.vae2_2 import Wan2_2_VAE

    vae_pth = os.path.join(args.checkpoint_dir, ti2v_5B.vae_checkpoint)
    print(f"Device: {args.device}")
    print(f"Loading VAE from {vae_pth}")

    vae = Wan2_2_VAE(
        vae_pth=vae_pth,
        device=torch.device(args.device),
    )
    print("VAE initialised.")

    # Small dummy video: [C=3, T=1, H=64, W=64]
    dummy_video = torch.randn(3, 1, 64, 64, dtype=torch.float32)

    print("Running encode...")
    latents = vae.encode([dummy_video])
    assert latents is not None and len(latents) > 0, "encode() returned None"
    print(f"  latent shape: {latents[0].shape}  dtype: {latents[0].dtype}")

    print("Running decode...")
    recon = vae.decode(latents)
    assert recon is not None and len(recon) > 0, "decode() returned None"
    print(f"  recon  shape: {recon[0].shape}  dtype: {recon[0].dtype}")

    print("\nPASS — VAE encode/decode completed.")


if __name__ == "__main__":
    main()
