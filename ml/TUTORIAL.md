# ml/ tutorial — the whole workflow, end to end

A hands-on guide to experimenting on models in this folder. Everything below
was executed and verified on this machine (RTX 3050 Laptop 4 GB, torch
2.8.0+cu126, Python 3.10). Commands are PowerShell, run from `ml/` with the
venv active.

---

## 0. The mental model

One loop, three entry points. Everything flows through the same pipe:

```
data/  →  adapter (any model)  →  eval/runner  →  runs/eval/<ts>_<run>/result.json
              ↑
   training.train → training.export → models/<name>/ → generic adapter
```

You never edit the harness to try something new. You **add** something:

| to try… | you add… | picked up by |
|---|---|---|
| a new model config / inference knob | an entry in `eval/adapters.yaml` | `eval.runner --runs` |
| a new architecture | a class with `@register(...)` in `training/archs.py` | `train --arch`, export, eval |
| a new training loss | a function with `@register_loss(...)` or `@register_aux_loss(...)` in `training/losses.py` | `train --loss` |
| a trained checkpoint | `training.export` output in `models/<name>/` | `adapters.generic`, future `src/` engine |

**The one rule that makes numbers meaningful:** every result JSON carries a
`dataset_hash`. Two results are comparable *only* if the hash matches. See §2.

---

## 1. Setup

```powershell
cd C:\myspace\study\ai\audio-denoise-project\ml
.venv\Scripts\activate          # everything below runs from ml/ via python -m
```

Verify the environment once:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.8.0+cu126 True
```

First DFN3 use auto-downloads the pretrained model to
`%LOCALAPPDATA%\DeepFilterNet\...\DeepFilterNet3` (one time, ~few MB).

### Health check (rerun anytime)

```powershell
python -m scripts.smoke_test
```

Exercises the full loop with synthetic data (~2–5 min, no real data needed):
build eval set → manifest → metric sanity → 1-epoch train → export → eval
sweep (DFN3 + the tiny model) → summary JSON. If this passes, the harness is
healthy and any later failure is *your* change, not plumbing.

Note: the smoke test exits with `code 1` under PowerShell because the benign
`torchaudio.backend.common.AudioMetaData` deprecation warning is treated as
stderr — check the output, not the exit code.

---

## 2. Real data and the dataset hash

The current manifest (`eval/manifests/paired_v1.json`, hash
`6be6be9c8878a733`) is built from 4 **synthetic** tone pairs — its numbers are
plumbing-only and meaningless. Before trusting anything:

```powershell
# 1. Clean speech wavs in one dir, noise wavs in another (any SR; resampled to 48k)
python -m scripts.build_eval_set --clean data/src_clean --noise data/src_noise --num 20
# 2. Freeze it — this hash is now your reference for every future number
python -m eval.manifest build --dir data/eval/paired --out eval/manifests/paired_v1.json
```

Synthetic-only plumbing test (tones as "speech"): add `--synthetic` instead of
`--clean/--noise`. Do **not** report metrics from it.

Want a different eval set (more pairs, other SNRs)? Create `paired_v2.json` —
never rebuild in place. Old numbers stay comparable to old runs only; that's
the whole point of the hash. The same discipline applies to training data:
`data/train/paired/{noisy_XXXX, clean_XXXX}.wav`, same synthesis script,
different `--out`.

---

## 3. Baselines and inference knobs (no training — start here)

Everything goes in `eval/adapters.yaml`:

```yaml
runs:
  dfn3:
    module: adapters.dfnet
    args: { model: DeepFilterNet3 }
  dfn3_pf:
    module: adapters.dfnet
    args: { model: DeepFilterNet3, post_filter: true, atten_lim_db: 12 }
  dfn2:
    module: adapters.dfnet
    args: { model: DeepFilterNet2 }
```

```powershell
python -m eval.runner --manifest eval/manifests/paired_v1.json --runs dfn3 dfn3_pf dfn2
```

Output per run:

```
runs/eval/<timestamp>_<run>/result.json   # frozen schema: dataset_hash, metrics, rtf, hardware, date
runs/eval/<timestamp>_<run>/rows.csv      # per-file metrics + per-file RTF
runs/eval/<timestamp>_<run>/est/          # the enhanced audio — LISTEN to it
runs/eval/summary_<timestamp>.json        # all runs of the sweep in one file
```

Metrics (paired tier):

| metric | direction | notes |
|---|---|---|
| `si_sdr` | ↑ higher better | fullband, 48 kHz |
| `stoi` | ↑ higher better | computed at 16 kHz |
| `pesq_torch_loss` | **↓ lower better** | differentiable PESQ proxy, range [0, ∞); not ITU P.862 — proper PESQ needs the `pesq` C package on Linux/WSL |
| `rtf_mean` | ↓ (per second of audio) | file-mode; GPU here, CPU will be slower |

Sanity behavior (verified): clean-vs-clean SI-SDR ≈ +140 / pesq_loss 0.0 vs
clean-vs-noisy 0.0 dB / 5.89. STOI 1.0 vs 0.36.

Baseline on the *synthetic* set (plumbing numbers, not quality):

| run | SI-SDR ↑ | STOI ↑ | pesq_loss ↓ | RTF |
|---|---|---|---|---|
| dfn3 | 0.78 | 0.710 | 5.57 | 0.12 |
| tiny_smoke (1 epoch) | −5.32 | 0.602 | 5.89 | 0.003 |

Metrics mislead — over-suppression and musical noise show up in ears first.
Always audition `runs/eval/.../est/`. For real recordings without a clean
reference: listening + DNSMOS (stub; see §8).

---

## 4. Training: scratch, fine-tune, distill (one loop)

### Scratch

```powershell
python -m training.train --arch tiny_unet --data data/train/paired --epochs 20
```

Key args (4 GB VRAM guidance: `--segment-seconds 2-4 --batch 1-2`,
scale effective batch with `--grad-accum`):

| arg | meaning |
|---|---|
| `--arch` | any name in `training/archs.py REGISTRY` |
| `--arch-kwargs` | JSON arch constructor kwargs, e.g. `'{\"width\": 32}'` — saved into `arch.json` → exported metadata |
| `--data` | dir of `noisy_XXXX/clean_XXXX` pairs |
| `--segment-seconds` | training segment length (whole file if shorter) |
| `--lr`, `--epochs`, `--seed` | as usual |
| `--out` | run dir (default `runs/train_<arch>_<id>`); `config.json` snapshots the full invocation |

Each run dir ends with `ckpt.pt` + `arch.json` + `config.json`.

### Fine-tune

Same loop, plus an init and a lower LR:

```powershell
python -m training.train --arch my_crn --init-from models/my_crn_v1/weights.pt \
    --data data/train/paired --lr 1e-4 --epochs 5
```

Change **one** variable per run vs the baseline. Fine-tuning DeepFilterNet
*itself* doesn't go through this loop — use the Colab notebook
[`finetune_dfn3_colab.ipynb`](finetune_dfn3_colab.ipynb) (pip wheel's own model +
loss, no Rust build; T4, ~30–60 min) — a fine-tuned DFN is delivered via the
same `models/` format with `arch: "dfnet"` handled by the DFN engine.

### Distillation (response-based, two steps)

```powershell
# Step 1: teacher labels — run the teacher adapter over the TRAIN noisy files
python -c "
from adapters.dfnet import enhance_file
from pathlib import Path
for f in Path('data/train/paired').glob('noisy_*.wav'):
    Path('data/train/teacher').mkdir(exist_ok=True)
    enhance_file(str(f), str(Path('data/train/teacher') / f.name), model='DeepFilterNet3')
"
# Step 2: student learns to imitate the teacher output
python -m training.train --arch tiny_unet --data data/train/paired --teacher-dir data/train/teacher
```

With `--teacher-dir`, the teacher's output (same `noisy_XXXX.wav` filenames)
replaces the clean target. Nothing else changes. Afterwards evaluate student
vs teacher vs clean on the same manifest.

---

## 5. Custom losses

Two registries in `training/losses.py`, both wired to `--loss` / `--loss-kwargs`
(the chosen loss + kwargs land in the run's `config.json` automatically).

### Lane 1 — waveform losses (`@register_loss`)

Contract: `fn(est, target, **kwargs) -> scalar`, both `[B, T]` waveforms on
device. Built-ins: `default` (si_sdr + 0.3·mrstft, weights tunable), `si_sdr`,
`mrstft`.

```powershell
# PowerShell quoting: single quotes with \" — PS strips bare inner quotes
python -m training.train --arch tiny_unet --loss default `
    --loss-kwargs '{\"si_sdr_weight\": 1.0, \"stft_weight\": 1.0}'
```

```python
# training/losses.py
@register_loss("combo")
def combo(est, target, lambda1=0.5):
    return lambda1 * si_sdr_loss(est, target) + (1 - lambda1) * mrstft_loss(est, target)
# then: --loss combo --loss-kwargs '{\"lambda1\": 0.7}'
```

### Lane 2 — aux losses, arch internals (`@register_aux_loss`)

Contract: `fn(est, target, aux, **kwargs)`. `aux` is a dict containing `"x"`
(the input, added by train.py) plus whatever the arch returned from
`forward(x, return_aux=True)`.

An arch opts in by accepting `return_aux: bool = False` in its forward:
True must return `(est, aux_dict)`; False keeps the plain `[B, T]` contract,
so exported models work unchanged in the app. train.py detects the capability
by inspecting the forward signature and refuses aux losses on archs that
don't support them (clear error).

Built-ins:

- **`irm_mask`** — Ideal Ratio Mask supervision for STFT magnitude-masking
  archs. aux contract: `{"mask": [B, F, T]` post-sigmoid`, "n_fft": int, "hop": int}`.
  IRM = |STFT(clean)|² / |STFT(noisy)|² clamped to [0,1]; MSE vs the mask.
  Both sides use `torch.stft` defaults (hann, center=True) — the arch must
  match or shapes won't align.
- **`stages`** — multi-stage supervision (e.g. estimate-then-subtract archs).
  aux contract: `{"stages": {name: [B, T] intermediate estimate}}`. Weighted
  sum of si_sdr per stage; weights via
  `--loss-kwargs '{\"stage_weights\": {\"noise_est\": 0.5}}'`.

```powershell
python -m training.train --arch stft_mask_net --loss irm_mask --epochs 1
# arch config: --arch-kwargs '{\"n_fft\": 512, \"hop\": 128, \"width\": 24}'
```

`stft_mask_net` (11k params) is the worked example: predicts a sigmoid
magnitude mask over the noisy STFT, reuses noisy phase, iSTFTs back, and
exposes the mask. Real masking archs (GTCRN and friends) follow this shape.

**Not implemented on purpose:** feature matching (FitNets-style teacher
hidden-state distillation). It needs the teacher loaded in-loop plus
projection layers per feature-shape pair; response distillation (§4) usually
captures most of the gain. Build it when you have two own-registry archs
worth distilling between.

Verified behavior: gradients flow from `irm_mask` into the conv weights;
`stages` is finite and differentiable; bad arch/loss combinations fail fast
with readable messages; default-loss regression is byte-identical (5.4839)
across the refactor.

---

## 6. Adding an architecture

Two requirements: register it, obey the contract.

```python
# training/archs.py
@register("my_crn")            # or @register("my_crn", native_sr=16000)
class MyCRN(nn.Module):
    def __init__(self, width=32):   # constructor kwargs = config (via --arch-kwargs)
        ...
    def forward(self, x):           # [B, T] noisy waveform (float32)
        return est                  # [B, T] estimate, same length
```

Then:

```powershell
python -m training.train --arch my_crn --arch-kwargs '{\"width\": 32}' --epochs 1   # smoke: shape bugs die here, seconds not minutes
python -m training.train --arch my_crn --epochs 20
python -m training.export --ckpt runs/train_my_crn_XXXX/ckpt.pt --name my_crn_v1
# eval/adapters.yaml:
#   my_crn_v1: { module: adapters.generic, args: { model_dir: models/my_crn_v1 } }
python -m eval.runner --manifest eval/manifests/paired_v1.json --runs dfn3 my_crn_v1
```

Resampling is handled downstream (declare `native_sr` in the decorator for
narrowband models) — nothing else to touch.

---

## 7. Export and src/ compatibility

```powershell
python -m training.export --ckpt runs/train_xxx/ckpt.pt --name mymodel
```

Creates `models/<name>/{weights.pt, metadata.json}` — the delivery contract
(see `models/README.md`). The future app engine reads exactly these files:
metadata (`arch` + `config` + `native_sr`) → build via the same registry →
load `weights.pt` → run plain forward with resample in/out. That is the same
code path as `adapters/generic.py`, which is why **anything exported here and
passing eval is app-ready by construction**.

When promoting a model, reference its winning `result.json` path in the
metadata `notes`.

---

## 8. Stubs and what still needs doing

- **DNSMOS** (real-recordings tier): runner not ported.
  `python -m scripts.download_dnsmos.py` fetches `model_v8.onnx` into
  `eval/dnsmos/`; if the URL 404s, grab it from microsoft/DNS-Challenge
  manually, then port `dnsmos_local.py`.
- **Certified PESQ**: needs the `pesq` C package → Linux/WSL.
- **Real training data**: the train dataloader is naive (no augmentation —
  RIRs, dynamic SNR mixing). Fine for first experiments; upgrade before
  serious campaigns.
- **Streaming/latency metrics**: file-mode RTF only.
- **4 GB VRAM**: DFN fine-tuning is better done on a cloud GPU; local runs
  want small segments/batches + grad-accum.

---

## 9. Suggested first week

1. **Real eval set + manifest** (§2) — everything else is noise without it.
2. **Knob sweep** dfn3 vs dfn3_pf vs dfn2 (§3) — your baseline table.
3. **tiny_unet on real train pairs** (§4) — proves the loop on real data,
   gives the "worst acceptable" floor.
4. **Pick a direction**: arch experiments (§6), custom losses (§5), or
   distillation (§4) — with numbers in hand.

## 10. Gotchas (all hit for real)

| gotcha | reality |
|---|---|
| PowerShell JSON args | single quotes with `\"`: `--loss-kwargs '{\"w\": 1.0}'` — bare `"` gets stripped, `\"` inside `\"` mangles |
| DFN `enhance()` | requires a **CPU** input tensor; the library moves it to GPU itself |
| `pesq_torch_loss` | lower is better; factor fixed at 1.0 for comparability |
| torchaudio warning | `AudioMetaData` deprecation on df import — benign; breaks `$LASTEXITCODE` logic |
| synthetic eval set | hash `6be6be9c8878a733` = plumbing only; never quote its numbers |
| absolute metric values | meaningless across datasets — same `dataset_hash` + `metric_version` or it didn't happen |
