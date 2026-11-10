"""Single train loop covering the three experiment types.

  scratch:    python -m training.train --arch tiny_unet --data data/train/paired
  fine-tune:  python -m training.train --arch <arch-of-checkpoint> \
                  --init-from models/<name>/weights.pt --lr 1e-4 ...
  distill:    1) generate teacher targets with an adapter over the TRAIN noisy files
              2) python -m training.train --arch tiny_unet --teacher-dir runs/teacher_out ...

Loss is selectable via --loss <name> with --loss-kwargs JSON. Waveform losses
come from training.losses.LOSS_REGISTRY; arch-internal losses (mask / stage
supervision) from AUX_LOSS_REGISTRY - those need an arch accepting
return_aux=True in forward (training.archs.stft_mask_net is the worked
example). Add your own with @register_loss / @register_aux_loss.

Arch constructor kwargs: --arch-kwargs JSON (saved into arch.json and
re-emitted into exported metadata).

Data layout: data/train/paired/{noisy_XXXX, clean_XXXX}.wav (48 kHz mono).
For distillation, point --teacher-dir at a directory of noisy_XXXX.wav files
with the same numbering; those become the targets (teacher outputs).

4 GB VRAM guidance: segment_seconds=2-4, batch=1-2, grad-accum to taste.
"""

import argparse
import inspect
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from training import losses
from training.archs import REGISTRY, build, ARCH_NATIVE_SR

SR = 48000


def load_pair(noisy_path: Path, clean_path: Path, native_sr: int) -> tuple[np.ndarray, np.ndarray]:
    import math
    import scipy.signal as sig
    resamp = lambda x, a, b: (x if a == b else
        sig.resample_poly(x, b // math.gcd(a, b), a // math.gcd(a, b)).astype(np.float32))
    noisy, sr_n = sf.read(str(noisy_path), dtype="float32")
    clean, sr_c = sf.read(str(clean_path), dtype="float32")
    if noisy.ndim > 1: noisy = noisy.mean(axis=-1)
    if clean.ndim > 1: clean = clean.mean(axis=-1)
    noisy = resamp(np.nan_to_num(noisy), sr_n, native_sr)
    clean = resamp(np.nan_to_num(clean), sr_c, native_sr)
    n = min(noisy.size, clean.size)
    return noisy[:n], clean[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--arch-kwargs", default="{}", help="JSON kwargs for the arch constructor")
    ap.add_argument("--loss", default="default",
                    choices=sorted({**losses.LOSS_REGISTRY, **losses.AUX_LOSS_REGISTRY}),
                    help="key in training.losses (waveform or aux registry)")
    ap.add_argument("--loss-kwargs", default="{}", help="JSON kwargs for the loss fn")
    ap.add_argument("--data", default="data/train/paired")
    ap.add_argument("--teacher-dir", default=None,
                    help="dir of noisy_XXXX.wav teacher outputs (enables distillation)")
    ap.add_argument("--init-from", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--segment-seconds", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="output dir (default runs/train_<ts>)")
    a = ap.parse_args()
    loss_kwargs = json.loads(a.loss_kwargs)
    arch_kwargs = json.loads(a.arch_kwargs)
    if a.loss in losses.AUX_LOSS_REGISTRY:
        loss_fn, need_aux = losses.get_aux_loss(a.loss), True
    else:
        loss_fn, need_aux = losses.get_loss(a.loss), False

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    native_sr = ARCH_NATIVE_SR[a.arch]
    model = build(a.arch, **arch_kwargs).to(device)
    if need_aux and "return_aux" not in inspect.signature(model.forward).parameters:
        raise SystemExit(f"arch '{a.arch}' doesn't accept return_aux — "
                         f"required for --loss {a.loss}")
    if a.init_from:
        model.load_state_dict(torch.load(a.init_from, map_location="cpu", weights_only=True))
        print(f"initialized from {a.init_from} (fine-tune mode)")

    data_dir = Path(a.data)
    pairs = sorted(data_dir.glob("noisy_*.wav"))
    if not pairs:
        raise SystemExit(f"No noisy_*.wav in {data_dir}")
    seg_len = int(a.segment_seconds * native_sr)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    run_dir = Path(a.out or f"runs/train_{a.arch}_{np.random.randint(1e6):06d}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")

    step = 0
    for epoch in range(a.epochs):
        order = list(pairs); random.shuffle(order)
        epoch_loss, n_batches = 0.0, 0
        opt.zero_grad()
        for i, noisy_path in enumerate(order):
            clean_path = noisy_path.parent / noisy_path.name.replace("noisy_", "clean_", 1)
            target_path = (Path(a.teacher_dir) / noisy_path.name if a.teacher_dir else clean_path)
            if not target_path.exists():
                continue
            noisy, target = load_pair(noisy_path, target_path, native_sr)
            if noisy.size <= seg_len:
                segs = [(noisy, target)]
            else:
                starts = range(0, noisy.size - seg_len + 1, seg_len)
                segs = [(noisy[s:s+seg_len], target[s:s+seg_len]) for s in starts]
            for xs, ts in segs:
                x = torch.from_numpy(xs).unsqueeze(0).to(device)
                t = torch.from_numpy(ts).unsqueeze(0).to(device)
                with torch.autocast(device_type=device, enabled=(device == "cuda")):
                    if need_aux:
                        est, aux_m = model(x, return_aux=True)
                        loss = (loss_fn(est, t, {"x": x, **aux_m}, **loss_kwargs)
                                / a.grad_accum)
                    else:
                        est = model(x)
                        loss = loss_fn(est, t, **loss_kwargs) / a.grad_accum
                scaler.scale(loss).backward()
                epoch_loss += loss.item() * a.grad_accum
                n_batches += 1
                if n_batches % a.grad_accum == 0:
                    scaler.step(opt); scaler.update(); opt.zero_grad()
                step += 1
        print(f"epoch {epoch+1}/{a.epochs} loss={epoch_loss/max(n_batches,1):.4f}")

    torch.save(model.state_dict(), run_dir / "ckpt.pt")
    (run_dir / "arch.json").write_text(
        json.dumps({"arch": a.arch, "native_sr": native_sr, "config": arch_kwargs},
                   indent=2), encoding="utf-8")
    print(f"checkpoint -> {run_dir/'ckpt.pt'}  (export: python -m training.export --ckpt {run_dir/'ckpt.pt'} --name mymodel)")


if __name__ == "__main__":
    main()
