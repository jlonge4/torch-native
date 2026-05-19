# Wan 2.2 on Trainium 2

Running Wan2.2-TI2V-5B (image-to-video) on AWS Trainium 2 using TorchNeuron Eager Mode.
No diffusers — patched base code from the upstream Wan2.2 repo.

## Instance

| | |
|---|---|
| Host | `trn2-2` (see `~/.ssh/config`) |
| Type | `trn2.3xlarge` — 1 Neuron device, 4 NeuronCores, 96 GB HBM total |
| Venv | `/home/ubuntu/moduscope-deps-20260518-105742/ms_venv` |
| Wan base | `/home/ubuntu/Wan2.2` (patched in-place) |
| Generate script | `/home/ubuntu/wan22_generate.py` |

## Repo structure

```
wan2.2/
├── wan22_neuron_patches/       # drop-in replacements for Wan2.2/wan/
│   ├── apply_patches.sh        # copies patches to /home/ubuntu/Wan2.2/wan/
│   ├── attention.py            # flash_attention CUDA guard → SDPA fallback
│   ├── model.py                # dtype fixes, CPU arithmetic, real-valued RoPE
│   ├── textimage2video.py      # device, autocast, blending, scheduler on CPU
│   ├── vae2_2.py               # autocast fix (neuron→cpu)
│   ├── t5.py                   # (unchanged from upstream, kept for completeness)
│   └── __init__.py
├── run/
│   └── wan22_generate.py       # CLI: --image, --size, --frames, --steps, --fps
├── tests/
│   └── ...
└── outputs/
    ├── wan22_480x256_21f_20steps_v2.mp4
    └── wan22_480x256_81f_30steps.mp4
```

## Confirmed working configurations

| Resolution | Frames | Steps | Time | Notes |
|---|---|---|---|---|
| 256×256 | 5 | 1 | ~2 s | smoke test |
| 480×256 | 21 | 20 | ~60 s | |
| 480×256 | 81 | 30 | ~115 s | |
| **832×480** | **21** | **30** | **~100 s** | **best quality — confirmed good output** |
| 832×480 | 121 | — | OOM | seq too large for per-core HBM |

## Base repo

```bash
git clone https://github.com/Wan-Video/Wan2.2.git /home/ubuntu/Wan2.2
cd /home/ubuntu/Wan2.2 && git checkout 42bf4cf
```

## Quickstart

### 1. Copy patches

```bash
rsync -avz wan22_neuron_patches/ trn2-2:/home/ubuntu/wan22_neuron_patches/
scp run/wan22_generate.py trn2-2:/home/ubuntu/wan22_generate.py
ssh trn2-2 "bash /home/ubuntu/wan22_neuron_patches/apply_patches.sh"
```

### 2. Generate video

```bash
ssh trn2-2 "source /home/ubuntu/moduscope-deps-20260518-105742/ms_venv/bin/activate && \
  python /home/ubuntu/wan22_generate.py \
    --image example \
    --size 480x256 \
    --frames 81 \
    --steps 30 \
    --fps 24 \
    --prompt 'A red panda playing in bamboo forest, cinematic, 4k' \
    --output /home/ubuntu/output.mp4 2>&1 | tee /home/ubuntu/generate.log"
scp trn2-2:/home/ubuntu/output.mp4 outputs/
```

- `--image example` uses the bundled `Wan2.2-TI2V-5B/examples/i2v_input.JPG`
- `--image /path/to/img.jpg` for a custom image
- Omit `--image` for text-to-video (t2v) mode

---

## Pipeline: what runs where

```
Input image (PIL)
       │
       ▼
┌─────────────────────────────────────────────┐
│  VAE encode  [CPU]                          │
│  Wan2_2_VAE(device='cpu')                   │
│  CausalConv3d encoder → 48-channel latent   │
│  Output: z [48, F_lat, H_lat, W_lat]        │
└─────────────────────────────────────────────┘
       │
       ▼  blend: z[:,0] = encoded ref frame
          z[:,1:] = random noise
       │
┌─────────────────────────────────────────────┐
│  T5 text encoder  [CPU]                     │
│  t5_cpu=True; context tensors moved to      │
│  Neuron only for the model forward pass     │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Denoising loop (N steps)                   │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │  Scheduler / mask / timestep [CPU]   │   │
│  │  FlowUniPCMultistepScheduler         │   │
│  │  masks_like, temp_ts construction    │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │  WanModel forward  [NEURON:0]        │   │
│  │  5B-param DiT, 30 transformer blocks │   │
│  │  Input: 48-ch latent on neuron:0     │   │
│  │  Output: 48-ch noise pred → .cpu()   │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │  Scheduler step + re-blend  [CPU]    │   │
│  │  Re-pin frame 0 to reference latent  │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  VAE decode  [CPU]                          │
│  48-ch latent → [3, T, H, W] video          │
└─────────────────────────────────────────────┘
       │
       ▼
     mp4 (imageio + ffmpeg)
```

**Why CPU for everything except the DiT forward:** TorchNeuron Eager
compiles a new NEFF for every unique op+tensor-shape combination on first
use. Scalar ops, per-element mask arithmetic, and scheduler math each
produce tiny compilations that take minutes each. Keeping those on CPU
avoids hundreds of spurious compilations per step.

---

## Patches: what changed and why

### `attention.py`

**Problem:** `flash_attention()` asserts `q.device.type == 'cuda'`.
`flash_attn` is a CUDA-only library; it is not installed on trn2-2.

**Fix:** Added a non-CUDA branch before the assert that falls through to
`F.scaled_dot_product_attention`. On Neuron this hits the hardware's fused
SDPA kernel; on CPU it uses PyTorch's math path.

`model.py` also imports `attention as _attention` and calls `_attention()`
instead of `flash_attention()` at both self-attention and cross-attention sites.

---

### `model.py`

| Problem | Root cause | Fix |
|---|---|---|
| `autocast('neuron')` causes device confusion / hang | TorchNeuron 2.11 bug | Changed all autocast contexts to `autocast('cpu', enabled=False)` |
| dtype mismatch at `patch_embedding` | `convert_model_dtype=True` casts weights to bf16; inputs are fp32 | `x = [u.to(dtype=emb_dtype) for u in x]` before patch_embedding |
| dtype mismatch at `time_embedding` | `sin_emb_dev` hardcoded to `torch.bfloat16` | Cast to `emb_dtype` instead |
| dtype mismatch at `text_embedding` | Zero-pad tensors were fp32; linear layer was bf16 | Cast context to `emb_dtype` during padding |
| `WanLayerNorm` crash with bf16 weights | `F.layer_norm` with bf16 weight/bias requires bf16 input, but code does `x.float()` first | Cast `weight` and `bias` to float32 explicitly in `WanLayerNorm.forward` |
| RoPE `view_as_complex` deadlocks on Neuron | Neuron doesn't support float64 or complex dtypes | Replaced with real-valued float32 rotation; `rope_params` returns `(cos,sin)` stacked as `[L, c, 2]`; `rope_apply` uses explicit 2D rotation |
| Per-element timestep/mask ops trigger NEFF recompilation | Each new tensor shape on Neuron compiles a NEFF on first use | All mask/timestep/scheduler ops kept on CPU; only `.to(self.device)` before DiT forward |

---

### `textimage2video.py`

**autocast:** Both t2v and i2v paths used `autocast('cuda', ...)` / `autocast('neuron', ...)`.
Changed to `autocast('cpu', enabled=False)`.

**i2v architecture — the key one:**
An incorrect assumption was that the TI2V model takes separate `y` channels
(mask + z_ref). The correct architecture is:
- `vae.model.z_dim = 48` — VAE outputs 48-channel latents
- `WanModel in_dim = 48` — takes the 48-ch blended latent directly; no `y` concatenation
- Reference conditioning is done by **blending** (not a separate channel)

```python
# Before loop: pin frame 0 to VAE-encoded reference
latent_cpu = (1. - mask2[0]) * z[0].cpu() + mask2[0] * noise_cpu_i2v
latent = latent_cpu.to(self.device)

# After each step: re-pin frame 0
latent_cpu = (1. - mask2[0]) * z[0].cpu() + mask2[0] * latent_cpu
```

Previous wrong approach built `y_model = cat([mask(48), zref(48)])` = 96 channels,
then the model did `cat([x(48), y(96)])` = 144 channels into a `in_dim=48`
conv, causing the error:
```
RuntimeError: expected input[1, 144, 2, 68, 50] to have 48 channels
```

**VAE encode device:** `img.to(self.device)` moved the image to Neuron, but
the VAE runs on CPU. Fixed: `z = self.vae.encode([img.cpu()])`.

**seed_g generator:** Must be on CPU; Neuron doesn't support device-placed generators.

**`--size` ignored in i2v:** The `generate()` method passes `max_area` not `size`
to `i2v()`. Fixed by passing `max_area=w*h` from `wan22_generate.py`.

---

### `vae2_2.py`

`encode()` and `decode()` used `amp.autocast('neuron', ...)`.
VAE runs on CPU; `'neuron'` autocast causes device confusion.
Changed to `amp.autocast('cpu', dtype=self.dtype, enabled=False)`.

---

### `run/wan22_generate.py`

- Added `--image [path|'example']` for i2v mode
- Added `max_area=w*h` in the `generate()` call so `--size` is respected in i2v
- Replaced `torchvision.io.write_video` (removed in torchvision 0.26) with
  `imageio.get_writer(..., format='ffmpeg', codec='libx264')`

---

## Resolution / frame count limits

Token count: `T = F_lat × (H/32) × (W/32)` where `F_lat = (frames-1)//4 + 1`

| Config | T | Status |
|---|---|---|
| 256×256, 5f | 160 | ✓ |
| 480×256, 21f | 720 | ✓ |
| 480×256, 81f | 2520 | ✓ |
| 832×480, 121f | 12090 | ✗ OOM on Neuron (`Failed to allocate resource`) |

Keep T below ~4000 to stay within per-core HBM limits.

---

## Environment

| Package | Version |
|---|---|
| torch (TorchNeuron Eager) | 2.10.0+cpu |
| torch_neuronx | 0.1.0+29019d8b |
| torchvision | 0.26.0+cpu |
| imageio | 2.37.3 |
| Python | 3.12 |
