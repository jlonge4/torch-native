"""
VAE encode → decode round-trip test.

Validates the torch.amp.autocast fix in vae2_2.py using a small CPU tensor
(no real checkpoint needed — exercises model forward pass with random weights).

Usage:
    python tests/test_vae.py [--vae-pth /path/to/Wan2.2-TI2V-5B/Wan_VAE_C48.pth]

Without --vae-pth, runs with randomly-initialized weights (useful for checking
that the autocast API works before loading 2+ GB of weights).
"""
import argparse
import sys
import os

# Allow running from runway-ml root or the tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.amp as amp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-pth", default=None,
                   help="Path to Wan_VAE_C48.pth (optional; skip to use random weights)")
    p.add_argument("--device", default="cpu",
                   help="'cpu' or 'neuron' (default: cpu)")
    return p.parse_args()


def main():
    args = parse_args()

    # Import after path setup so we can also run from the tests/ dir
    try:
        from wan22_neuron_patches.vae2_2 import Wan2_2_VAE
    except ModuleNotFoundError:
        # Files deployed flat (no package wrapper)
        sys.path.insert(0, os.path.dirname(__file__))
        from vae2_2 import Wan2_2_VAE

    dtype = torch.bfloat16 if args.device == "neuron" else torch.float32
    print(f"Device: {args.device}  dtype: {dtype}")

    vae = Wan2_2_VAE(
        vae_pth=args.vae_pth,
        dtype=dtype,
        device=args.device,
    )
    print("VAE initialised.")

    # Small dummy video: [C=3, T=1, H=64, W=64] — one frame, tiny spatial size
    dummy_video = torch.randn(3, 1, 64, 64, dtype=torch.float32)

    print("Running encode...")
    latents = vae.encode([dummy_video])
    assert latents is not None, "encode() returned None — check autocast fix"
    print(f"  latent shape : {latents[0].shape}  dtype: {latents[0].dtype}")

    print("Running decode...")
    recon = vae.decode(latents)
    assert recon is not None, "decode() returned None — check autocast fix"
    print(f"  recon  shape : {recon[0].shape}  dtype: {recon[0].dtype}")

    print("\nPASS — VAE encode/decode completed without error.")


if __name__ == "__main__":
    main()
