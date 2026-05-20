# Wan 2.2 on AWS Trainium 2

Running [Wan2.2-TI2V-5B](https://github.com/Wan-Video/Wan2.2) (text/image-to-video) on AWS Trainium 2 using TorchNeuron Eager Mode with 4-way tensor parallelism and `torch.compile(backend="neuron")`.

## Outputs

| Config | Output |
|---|---|
| 832×480, 61f, TP=4 + compile | [wan22_tp4_compile_61f_v2.mp4](outputs/wan22_tp4_compile_61f_v2.mp4) |
| 832×480, 61f, TP=4 eager | [wan22_tp4_eager_61f.mp4](outputs/wan22_tp4_eager_61f.mp4) |

## Performance

All numbers on `trn2.3xlarge` (4 NeuronCores, 96 GB HBM), 832×480, 30 steps.

| Mode | Frames | Denoise | Total | MFU |
|---|---|---|---|---|
| Single-core eager | 21f | ~60s | ~90s | — |
| Single-core eager | 61f | ~5m | 6m37s | — |
| TP=4 eager | 61f | 147s | 331s | 6.06% |
| **TP=4 + torch.compile** | **61f** | **94s** | **279s** | **9.42%** |

MFU = model FLOP utilization against 4 × 158 TFLOP/s (BF16) = 632 TFLOP/s aggregate.

**torch.compile diagnostics (TP=4, 61f 832×480):**
- Graph breaks: **2 unique sites** × 32 blocks = ~65 graphs total
  - `model.py:570` — `sinusoidal_embedding_1d` (`@torch._dynamo.disable` — uses float64/CPU, deadlocks on Neuron)
  - `model.py:178` — `rope_apply` (`@torch._dynamo.disable` — per-sample for-loops over dynamic shapes)
- CPU fallback ops: **none**

## Instance

| | |
|---|---|
| Instance type | `trn2.3xlarge` |
| NeuronCores | 4 (24 GB HBM each, 96 GB total) |

## Quickstart

### Requirements

- Python 3.12
- [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22-pip-install.html) (`torch-neuronx`, `neuronx-cc`)
- `torch`, `torchvision`, `diffusers`, `transformers`, `imageio[ffmpeg]`, `Pillow`, `tqdm`
- Wan2.2-TI2V-5B weights from [Hugging Face](https://huggingface.co/Wan-Video/Wan2.2-TI2V-5B)

### Install

```bash
git clone https://github.com/jlonge4/torch-native.git
cd torch-native/wan2.2
pip install -r requirements.txt
```

### Generate

```bash
# TP=4 + torch.compile — two-phase execution (denoise then VAE decode)
# First run compiles and caches NEFFs (~6 min); subsequent runs use cache (~5 min)
python run/wan22_generate.py \
  --tp-degree 4 --compile \
  --checkpoint-dir /path/to/Wan2.2-TI2V-5B \
  --frames 61 --size 832x480 --steps 30 \
  --output output.mp4

# Single-core eager baseline
python run/wan22_generate.py \
  --checkpoint-dir /path/to/Wan2.2-TI2V-5B \
  --frames 21 --size 832x480 --steps 30 \
  --output output.mp4
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--tp-degree` | `1` | Tensor parallel degree. `4` = all NeuronCores on trn2.3xlarge |
| `--compile` | off | `torch.compile(backend="neuron")` on the DiT |
| `--frames` | `21` | Number of frames (must be 4n+1) |
| `--size` | `832x480` | WxH |
| `--steps` | `30` | Denoising steps |
| `--seed` | `42` | RNG seed |

## Architecture

### Two-phase execution (TP mode)

Neuron NEFFs (compiled kernels) **stay resident in HBM for the entire process lifetime** — `model.cpu()`, `empty_cache()`, and Python GC do not evict them. After 30 denoising steps the DiT's compiled graphs occupy most of each NeuronCore's HBM. If VAE decode runs in the same process it OOMs.

Fix: split into two subprocesses:

```
Phase 1 — torchrun --nproc-per-node 4 (TP denoising)
  └─ all 4 ranks denoise → rank 0 saves latent to /tmp/wan_tp_latent.pt → all ranks exit
     (DiT NEFFs freed on process exit)

Phase 2 — python (single process, clean HBM)
  └─ load latent → VAE decode on neuron:0 → save mp4
```

The parent `run/wan22_generate.py` orchestrates both phases via `subprocess.run`.

### TP sharding plan

Q and K are **not sharded** — `WanRMSNorm` computes RMS over the full projection dimension (3072). Sharding Q/K changes that denominator and produces garbled output.

```
Q, K   — unsharded: full [5120, 3072] on every rank
V      — ColwiseParallel: [5120, 768] per rank (TP=4)
O      — RowwiseParallel: [768, 5120] per rank + all-reduce
FFN[0] — ColwiseParallel
FFN[2] — RowwiseParallel + all-reduce
```

### What runs where (TP=4)

```
Input prompt / image
        │
        ▼
T5 text encoder [CPU]
  offloaded after encoding to free HBM for DiT shards
        │
        ▼
VAE encode [neuron:0, rank 0 only]
  image → 48-channel latent; blended with noise for i2v
        │
        ▼
Denoising loop (N steps) ──── scheduler / mask / timestep  [CPU]
  WanModel forward [neuron:0-3, TP=4]
  Q,K unsharded · V,O,FFN sharded + all-reduce per block
        │
        ▼  (process exit — DiT NEFFs evicted)
VAE decode [neuron:0, fresh process]
  latent → pixel frames
        │
        ▼
mp4 (imageio + ffmpeg)
```

## Changes from upstream

### `attention.py`

`flash_attention()` asserts `q.device.type == 'cuda'`. Non-CUDA path falls through to `F.scaled_dot_product_attention` with `attn_mask=None` (Neuron SDPA does not accept boolean masks).

### `model.py`

| Problem | Fix |
|---|---|
| `autocast('neuron')` hangs | `autocast('cpu', enabled=False)` |
| `view_as_complex` / float64 deadlocks on Neuron | Real-valued float32 RoPE: `rope_params` returns `(cos, sin)` stacked as `[L, c, 2]` |
| `sinusoidal_embedding_1d` crashes Neuron compiler (`aten.div.Tensor` with int) | `@torch._dynamo.disable` |
| `rope_apply` for-loops over `grid_sizes.tolist()` break Dynamo trace | `@torch._dynamo.disable` |
| TP sharding garbled Q/K (wrong RMSNorm denominator) | Keep Q, K unsharded; shard V, O, FFN only |
| `dist.group.WORLD` in all-reduce hooks | Use `None` (correct default pg) |
| `WanLayerNorm` crash with bf16 weights | Cast weight/bias to float32 before `F.layer_norm` |

### `textimage2video.py`

| Problem | Fix |
|---|---|
| T5 (~11 GB) replicated on all 4 NeuronCores | `t5_cpu=True` forced when `tp_degree > 1` |
| VAE (~0.5 GB) allocated on all 4 NeuronCores | Only rank 0 gets `device=neuron:0`; others get `device=cpu` |
| `model.cpu()` called on compiled TP model | Skip offload when `tp_degree > 1` |
| DiT NEFFs block VAE decode (same process) | Two-phase subprocess split; `latent_only=True` returns raw latent |

## Confirmed working configurations

| Resolution | Frames | Steps | Mode | Status |
|---|---|---|---|---|
| 832×480 | 21f | 30 | single-core | ✓ |
| 832×480 | 61f | 30 | single-core | ✓ |
| 832×480 | 21f | 30 | TP=4 | ✓ |
| **832×480** | **61f** | **30** | **TP=4 + compile** | **✓** |
| 832×480 | 121f | 30 | single-core | ✗ OOM |

## Base repo

[Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) @ `42bf4cf`.

## Environment

| Package | Version |
|---|---|
| torch | 2.10.0+cpu |
| torch-neuronx (Neuron SDK) | 2.10.0 |
| torchvision | 0.26.0+cpu |
| Python | 3.12 |
