# Wan 2.2 on Trainium 2 — TP + torch.compile

Running Wan2.2-TI2V-5B (image-to-video) on AWS Trainium 2 using TorchNeuron Eager Mode
with 4-way tensor parallelism across all NeuronCores and `torch.compile(backend="neuron")`.

> **Branch:** `tp-compile` — for the single-core eager baseline see `main`.

## Instance

| | |
|---|---|
| Type | `trn2.3xlarge` — 1 Neuron device, 4 NeuronCores, 96 GB HBM total |
| Venv | `/home/ubuntu/moduscope-deps-20260518-105742/ms_venv` |
| Repo | `/home/ubuntu/torch-native` |
| Generate script | `wan2.2/run/wan22_generate.py` |

## Repo structure

```
wan2.2/
├── wan/
│   ├── modules/
│   │   ├── attention.py        # flash_attention CUDA guard → SDPA fallback
│   │   ├── model.py            # dtype fixes, real-valued RoPE, TP sharding
│   │   └── ...
│   ├── textimage2video.py      # device, autocast, blending, TP offload, compile
│   └── ...
├── run/
│   └── wan22_generate.py       # CLI: --image, --size, --frames, --steps, --tp-degree, --compile
├── tests/
└── outputs/
```

## Base repo

Started from [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) at commit `42bf4cf`.

## Quickstart

### 1. Install on instance

```bash
git clone -b tp-compile https://github.com/jlonge4/torch-native.git
```

### 2. Generate video

```bash
source /home/ubuntu/moduscope-deps-20260518-105742/ms_venv/bin/activate

# Single core (baseline)
python torch-native/wan2.2/run/wan22_generate.py \
  --image example --size 832x480 --frames 21 --steps 30 \
  --output output_single.mp4

# TP=4 across all NeuronCores
torchrun --nproc-per-node 4 torch-native/wan2.2/run/wan22_generate.py \
  --tp-degree 4 \
  --image example --size 832x480 --frames 21 --steps 30 \
  --output output_tp4.mp4

# TP=4 + torch.compile (first run compiles NEFFs, subsequent runs use cache)
torchrun --nproc-per-node 4 torch-native/wan2.2/run/wan22_generate.py \
  --tp-degree 4 --compile \
  --image example --size 832x480 --frames 21 --steps 30 \
  --output output_tp4_compiled.mp4
```

## Confirmed working configurations

| Resolution | Frames | Steps | Mode | Denoise time | Notes |
|---|---|---|---|---|---|
| 256×256 | 5 | 1 | single | ~2 s | smoke test |
| **832×480** | **21** | **30** | **single** | **~60 s** | **confirmed good quality** |
| 832×480 | 61 | 30 | single | ~5 min | good quality |
| 832×480 | 21 | 30 | TP=4 | ~60 s | confirmed good quality |

---

## Pipeline: what runs where (TP=4)

```
Input image (PIL)
       │
       ▼
┌─────────────────────────────────────────────┐
│  VAE encode  [NEURON:0]                     │
│  Wan2_2_VAE → 48-channel latent             │
└─────────────────────────────────────────────┘
       │
       ▼  blend: z[:,0] = encoded ref frame
          z[:,1:] = random noise
       │
┌─────────────────────────────────────────────┐
│  T5 text encoder  [CPU → offloaded]         │
│  Offloaded to CPU after encoding in TP mode │
│  to free HBM for the DiT shards             │
└─────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Denoising loop (N steps)                                │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Scheduler / mask / timestep  [CPU]                │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  WanModel forward  [NEURON:0-3, TP=4]             │  │
│  │  Q,K: full weights on each rank (correct RMSNorm) │  │
│  │  V,O,FFN: col/row sharded + all-reduce            │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Scheduler step + re-blend  [CPU]                  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  VAE decode  [NEURON:0]  (DiT offloaded)    │
└─────────────────────────────────────────────┘
       │
       ▼
     mp4 (imageio + ffmpeg)
```

---

## Changes from upstream

### `attention.py`

**Problem:** `flash_attention()` asserts `q.device.type == 'cuda'`.

**Fix:** Non-CUDA branch falls through to `F.scaled_dot_product_attention`.

---

### `model.py`

| Problem | Root cause | Fix |
|---|---|---|
| `autocast('neuron')` causes hang | TorchNeuron 2.11 bug | Changed to `autocast('cpu', enabled=False)` |
| dtype mismatch at `patch_embedding` | `convert_model_dtype=True` casts weights to bf16; inputs are fp32 | `x = [u.to(dtype=emb_dtype) for u in x]` before patch_embedding |
| dtype mismatch at `time_embedding` / `text_embedding` | Hardcoded bf16 or fp32 | Cast to `emb_dtype` throughout |
| `WanLayerNorm` crash with bf16 weights | `F.layer_norm` with bf16 weight requires bf16 input; code upcasts to fp32 first | Cast `weight`/`bias` to float32 explicitly |
| RoPE `view_as_complex` deadlocks | Neuron doesn't support float64 or complex dtypes | Real-valued float32 rotation; `rope_params` returns `(cos,sin)` stacked as `[L, c, 2]` |
| Per-element ops trigger NEFF recompilation | Each new shape compiles a NEFF | Scheduler/mask/timestep ops kept on CPU |
| TP sharding broke video quality (garbled output) | Col-sharding Q/K changed RMSNorm denominator from 3072→768 elements | Q and K kept unsharded (full weights on each rank); only V, O, FFN sharded |
| `sinusoidal_embedding_1d` crashes Neuron compiler | `aten.div.Tensor` with int operand not supported in Torch-MLIR | `@torch._dynamo.disable` keeps it in eager |

**TP sharding plan:**
```
Q, K  — unsharded (full [5120, 3072] on every rank); norm computed over full dim
V     — ColwiseParallel  ([5120, 768] per rank for TP=4)
O     — RowwiseParallel  ([768, 5120] per rank) + all-reduce hook
FFN[0]— ColwiseParallel
FFN[2]— RowwiseParallel + all-reduce hook
```

---

### `textimage2video.py`

**T5 offload in TP mode:** T5 (~11 GB) is offloaded to CPU after encoding so
the 4 DiT shards fit in HBM.

**DiT offload before VAE decode:** DiT shards are offloaded and `empty_cache()`
called before VAE decode to avoid HBM fragmentation.

**`torch.compile` with TP:**
- `torch.compile(self.model, backend="neuron", fullgraph=False)`
- `dist.barrier()` before the denoising loop so only one rank wins the Neuron
  NEFF compile-cache lock; others wait cleanly instead of racing it
- First run compiles and caches NEFFs; subsequent runs load from cache

---

## Resolution / frame count limits

Token count: `T = F_lat × (H/32) × (W/32)` where `F_lat = (frames-1)//4 + 1`

| Config | T | Status |
|---|---|---|
| 256×256, 5f | 160 | ✓ |
| 832×480, 21f | 2340 | ✓ |
| 832×480, 61f | 6240 | ✓ |
| 832×480, 121f | 12090 | ✗ OOM (single core) |

With TP=4 the per-core HBM footprint is reduced — higher frame counts may become feasible.

---

## Environment

| Package | Version |
|---|---|
| torch | 2.10.0+cpu |
| torchvision | 0.26.0+cpu |
| imageio | 2.37.3 |
| Python | 3.12 |
