# Wan 2.2 on Trainium 2

Running Wan2.2-TI2V-5B on AWS Trainium 2 (Neuron SDK eager mode). No diffusers — patched base code only.

## Instance

| | |
|---|---|
| Host | `trn2-2` (see `~/.ssh/config`) |
| IP | `56.125.170.127` |
| Type | `trn2.3xlarge` — 1 Neuron device, 4 cores, 96 GB HBM |
| Venv | `/home/ubuntu/moduscope-deps-20260423-202725/ms_venv` |
| Base code | `/home/ubuntu/Wan2.2` (patched in-place) |
| Scripts | `/home/ubuntu/runway-ml/` |

## Repo structure

```
wan2.2/
├── wan22_neuron_patches/     # drop-in replacements for Wan2.2 base wan/
│   ├── apply_patches.sh      # copies patches → /home/ubuntu/Wan2.2/wan/
│   ├── model.py              # WanModel: sinusoidal embedding on CPU, e.float() head fix
│   ├── textimage2video.py    # WanTI2V: neuron:{device_id} device, T5 on CPU
│   ├── t5.py                 # T5EncoderModel: Neuron-compatible
│   ├── vae2_2.py             # Wan2_2_VAE: u.cpu() before decode
│   ├── vae2_1.py
│   ├── attention.py
│   └── __init__.py
├── run/
│   ├── wan22_run.py          # smoke test: 256×256, 5f, 1 step (~2 s, always works)
│   └── wan22_generate.py     # CLI: defaults = confirmed working config
├── tests/
│   ├── warm_neff.py          # iterative NEFF compiler warm-up for new shapes
│   ├── test_pipeline_single_step.py
│   ├── test_t5.py
│   └── test_vae.py
└── outputs/
    └── wan22_480x256_21f_20steps.mp4   # confirmed output
```

## Confirmed working

| Config | Time | Output |
|---|---|---|
| 256×256, 5f, 1 step | ~2 s | smoke test |
| **480×256, 21f, 20 steps** | ~90 s | `outputs/wan22_480x256_21f_20steps.mp4` |

The 12 cached NEFFs at `/var/tmp/neuron-compile-cache/neuronxcc-2.24.5133.0+58f8de22/` cover exactly the 480×256 / 21-frame config. Any new shape requires NEFF compilation first (see [Warming NEFFs for new shapes](#warming-neffs-for-new-shapes)).

## Base repo

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2 && git checkout 42bf4cf   # confirmed working commit
```

Then apply patches (see below).

## Quickstart

### 1. Apply patches

```bash
rsync -avz wan22_neuron_patches/ trn2-2:/home/ubuntu/runway-ml/
ssh trn2-2 "bash /home/ubuntu/runway-ml/apply_patches.sh"
```

### 2. Smoke test (~2 s — confirms pipeline works)

```bash
ssh trn2-2 "nohup /home/ubuntu/moduscope-deps-20260423-202725/ms_venv/bin/python -u \
  /home/ubuntu/wan22_run.py > /home/ubuntu/wan22_run.log 2>&1 &"
ssh trn2-2 "tail -f /home/ubuntu/wan22_run.log"
```

### 3. Full generation (480×256, 21f, 20 steps — uses cached NEFFs, ~90 s)

```bash
scp run/wan22_generate.py trn2-2:/home/ubuntu/runway-ml/wan22_generate.py
ssh trn2-2 "nohup /home/ubuntu/moduscope-deps-20260423-202725/ms_venv/bin/python -u \
  /home/ubuntu/runway-ml/wan22_generate.py \
  --output /home/ubuntu/wan22_output.mp4 \
  > /home/ubuntu/wan22_generate.log 2>&1 &"
ssh trn2-2 "tail -f /home/ubuntu/wan22_generate.log"
scp trn2-2:/home/ubuntu/wan22_output.mp4 outputs/
```

## Warming NEFFs for new shapes

Any frame count other than 21 (at 480×256) requires first-time NEFF compilation. Without it, the pipeline hangs indefinitely waiting for the compiler.

Run `warm_neff.py` first — it loads the DiT model directly and forces compilation in a controlled way using `NEURON_LAUNCH_BLOCKING=1`:

```bash
scp tests/warm_neff.py trn2-2:/home/ubuntu/runway-ml/warm_neff.py
ssh trn2-2 "NEURON_LAUNCH_BLOCKING=1 nohup \
  /home/ubuntu/moduscope-deps-20260423-202725/ms_venv/bin/python -u \
  /home/ubuntu/runway-ml/warm_neff.py \
  --frame-num 41 --size 480x256 \
  > /home/ubuntu/warm_neff.log 2>&1 &"
ssh trn2-2 "tail -f /home/ubuntu/warm_neff.log"
```

After it completes, run `wan22_generate.py --frames 41` normally.

## Key patches

| File | What changed | Why |
|---|---|---|
| `model.py` | `sinusoidal_embedding_1d` runs on CPU in float64, returns float32 | Neuron doesn't support float64 or int ops |
| `model.py` | `x = self.head(x, e.float())` | `convert_model_dtype=True` casts weights to bfloat16; head asserts float32 input |
| `vae2_2.py` | `self.model.decode(u.cpu().unsqueeze(0), ...)` | VAE decoder runs on CPU; latent must be moved off Neuron first |
| `textimage2video.py` | `self.device = torch.device(f"neuron:{device_id}")` | Neuron device string |

## What doesn't work

| | Reason |
|---|---|
| `diffusers.WanPipeline` | `NRT EXECUTION FAILED: Failed to allocate resource` in T5 `compute_bias` |
| `offload_model=True` with uncached NEFFs | Deadlocks — model is on CPU while Neuron compiler needs it on device |
| New frame counts without warming | Same deadlock — use `warm_neff.py` first |

## Video writer

Do **not** use `wan.utils.utils.save_video` — it calls `imageio.get_writer(..., quality=8)` which fails silently with ms_venv's PyAV-backed imageio (0-byte output).

Use `torchvision.io.write_video` directly:

```python
frames = ((video.permute(1,2,3,0).clamp(-1,1) + 1) / 2 * 255).byte()
torchvision.io.write_video(path, frames, fps=16)
```

## Environment

| Package | Version |
|---|---|
| torch | 2.10.0+cpu |
| torch_neuronx | 0.1.0+29019d8b |
| torchvision | 0.25.0+cpu |
| neuronxcc | 2.24.5133.0+58f8de22 |
