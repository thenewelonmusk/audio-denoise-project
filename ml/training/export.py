"""Export a trained checkpoint to the app-compatible model format.

  python -m training.export --ckpt runs/train_xxx/ckpt.pt --name mymodel

Creates models/<name>/ with:
  weights.pt      state dict
  metadata.json   {name, version, arch, native_sr, channels, config, license,
                   created, training_run}
This directory IS the delivery format: the future src/ engine wrapper reads
exactly these files (see models/README.md), and adapters.generic can already
consume it for evaluation.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch


def export(ckpt: str, name: str, license_: str = "internal") -> Path:
    ckpt_path = Path(ckpt)
    arch_info = json.loads((ckpt_path.parent / "arch.json").read_text(encoding="utf-8"))

    out = Path("models") / name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_path, out / "weights.pt")
    meta = {
        "name": name,
        "version": "0.1.0",
        "arch": arch_info["arch"],
        "native_sr": arch_info["native_sr"],
        "channels": 1,
        "config": arch_info.get("config", {}),  # arch constructor kwargs (from train --arch-kwargs)
        "license": license_,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_run": str(ckpt_path.parent),
        "notes": "state was trained with training/train.py; see training_run/config.json for the full invocation",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exported -> {out}  (evaluate: add a run to eval/adapters.yaml with "
          f"module=adapters.generic, args.model_dir={out})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--license", default="internal")
    a = ap.parse_args()
    export(a.ckpt, a.name, a.license)
