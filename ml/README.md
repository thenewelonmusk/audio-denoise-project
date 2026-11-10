# ml/ — model experiments

Separate from `src/`. This folder never imports `src/`, and the app refactor never touches this folder.

Full walkthrough: **[TUTORIAL.md](TUTORIAL.md)** — setup → data discipline → eval sweeps → training/distill/fine-tune → custom losses → export.

## Setup

```powershell
ml\.venv\Scripts\activate
python -m pip install -r ml\requirements.txt
```

Installed with CUDA torch (`requirements-torch-cuda.txt`) on the setup machine
(RTX 3050 Laptop, 4 GB VRAM). CPU-only machines: install torch from the CPU index
instead — see the comment at the top of `requirements.txt`.

## Verified environment (this machine)

- torch 2.8.0+cu126, CUDA available on RTX 3050 Laptop (4 GB), RTF/eval runs fine; training needs small batches.
- `deepfilternet` 0.5.6 + real Rust `DeepFilterLib` wheel (no compat shim needed on Py 3.10).
- Pretrained **DeepFilterNet3** auto-downloads to `%LOCALAPPDATA%\DeepFilterNet\...\DeepFilterNet3`.
- DFN call convention: `enhance()` expects a **CPU** input tensor (any sample rate audio must be
  loaded via `load_audio(path, sr=df_state.sr())`); the library moves inference to GPU itself.
- Benign warning on import: `torchaudio.backend.common.AudioMetaData` deprecation in df/io.py.

## Layout

```
adapters/         enhance_file(in, out, **args) per model family
                  dfnet.py   DFN2/DFN3 (--model DeepFilterNet2|3, --post-filter, --atten-lim-db)
                  generic.py loads exported models from models/<name>/ (see models/README.md)
eval/             metrics.py (SI-SDR, STOI, torch-pesq proxy), manifest.py, runner.py,
                  adapters.yaml (which runs a sweep evaluates), manifests/ (in git)
training/         archs.py (architecture registry — add your model here),
                  dataset.py (pair synthesis), losses.py, train.py, export.py
scripts/          build_eval_set.py, download_dnsmos.py, smoke_test.py
models/           exported app-compatible checkpoints (see models/README.md)
data/, runs/      audio + results (gitignored)
```

## First run

```powershell
python -m scripts.smoke_test        # full plumbing check, ~minutes, no real data needed

# Real eval set (do this before trusting any number):
python -m scripts.build_eval_set --clean <clean-speech-dir> --noise <noise-dir> --num 20
python -m eval.manifest build --dir data/eval/paired --out eval/manifests/paired_v1.json
python -m eval.runner --manifest eval/manifests/paired_v1.json --runs dfn3 dfn3_pf dfn2
```

## Experiment recipes

1. **Inference knobs** (no training): edit `eval/adapters.yaml` args (`post_filter`,
   `atten_lim_db`), add run names, sweep via `eval.runner`.
2. **Change architecture**: subclass in `training/archs.py` with
   `@register("name")`, contract = `[B, T] -> [B, T]` float32 waveform. Then
   `training.train --arch name --arch-kwargs '{\"width\": 32}'`, `training.export`, add to `adapters.yaml`.
3. **Fine-tune**: same arch + checkpoint as the base (`--init-from
   models/<name>/weights.pt`), lower `--lr`.
4. **Distillation (response-based)**: generate teacher outputs for the train
   noisy files with an adapter, then `training.train --teacher-dir <that dir>`;
   teacher outputs become the targets. For a DFN teacher:
   loop `adapters.dfnet.enhance_file(noisy_i, runs/teacher_out/noisy_i)`.
5. **Custom loss**: `training.losses` has a registry mirroring the arch one.
   Built-ins (`--loss default|si_sdr|mrstft`, tune weights via `--loss-kwargs`):
   ```powershell
   # PowerShell quoting: single quotes with \" (PS strips bare inner quotes)
   python -m training.train --arch tiny_unet --loss default --loss-kwargs '{\"si_sdr_weight\": 1.0, \"stft_weight\": 1.0}'
   ```
   Add your own in `training/losses.py`:
   ```python
   @register_loss("combo")
   def combo(est, target, lambda1=0.5):          # contract: [B, T] x [B, T] -> scalar
       return lambda1 * si_sdr_loss(est, target) + (1 - lambda1) * mrstft_loss(est, target)
   ```
   then `--loss combo --loss-kwargs '{\"lambda1\": 0.7}'`. The chosen loss and
   kwargs are snapshotted in the run's `config.json` automatically.

   **Aux losses** (need arch internals) live in a second registry,
   `AUX_LOSS_REGISTRY`, with contract `fn(est, target, aux, **kw)`. The arch
   opts in by accepting `return_aux: bool = False` in `forward` (True must
   return `(est, aux_dict)`; additions to the arch contract in archs.py).
   Built-ins:
   - `irm_mask` — mask supervision for STFT-magnitude masking archs
     (aux keys `mask`/`n_fft`/`hop`). Pair with the `stft_mask_net` example
     arch + `--arch-kwargs '{\"n_fft\": 512, \"hop\": 128}'`.
   - `stages` — multi-stage supervision (aux key `stages`: dict of
     intermediate `[B, T]` estimates), e.g. for estimate-then-subtract archs.
   ```powershell
   python -m training.train --arch stft_mask_net --loss irm_mask --epochs 1
   ```
   Feature matching (FitNets, teacher hidden-state distillation) is
   deliberately not implemented: it needs the teacher model loaded in-loop
   plus projection layers per shape pair — response distillation (recipe 4)
   usually captures most of the gain first.
6. **Compatibility with src/**: export writes `models/<name>/` in the format the
   future app engine consumes (models/README.md). Eval-validated = app-ready.

## Frozen experiment contract (v0.1)

- Adapter I/O: 48 kHz mono float32 WAV in, same length and sample rate out.
  Each adapter exposes `enhance_file(input, output, **args) -> dict` (used
  in-process by the runner) and a CLI (`python -m adapters.dfnet ...`) for
  manual use. 16 kHz-native models declare `native_sr` and resample internally.
- Eval tiers: `paired` (synthetic, has clean reference -> SI-SDR / STOI /
  torch-pesq loss [lower=better]; ITU PESQ needs the `pesq` C package on
  Linux/WSL) and `real` (no reference -> DNSMOS + listening notes).
- Every result records: dataset hash, metric code version, CPU/threads, date.
- Results JSON: `{engine, version, dataset_hash, metrics{}, rtf{}, hardware, date}`.
- Do not evaluate through the zip's `_libdf_compat` shim. Use `deepfilternet` + `DeepFilterLib`.

## Rules

- Never commit `data/`, `runs/`, `models/`, or `.venv/`.
- Do not modify `src/`, `app.py`, or the root `requirements.txt`.
- One experiment = one changed variable; keep the config snapshot in `runs/`.
