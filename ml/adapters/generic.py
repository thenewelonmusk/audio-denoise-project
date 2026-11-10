"""Adapter for exported local models (ml/models/<name>/).

Loads metadata.json, builds the architecture via training.archs, restores
weights. This is exactly the path the future src/ engine wrapper will use,
so anything evaluable here is app-compatible by construction.

CLI: python -m adapters.generic --model-dir models/tiny --in a.wav --out b.wav
"""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR = 48000
_lock = threading.Lock()
_loaded = {}


def _get_model(model_dir: str):
    if model_dir in _loaded:
        return _loaded[model_dir]
    d = Path(model_dir)
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/
    from training.archs import build
    model = build(meta["arch"], **meta.get("config", {}))
    state = torch.load(d / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _loaded[model_dir] = (model, meta)
    return _loaded[model_dir]


def enhance_file(input_path: str, output_path: str, *, model_dir: str) -> dict:
    model, meta = _get_model(model_dir)
    sr = meta.get("native_sr", SR)

    import math
    import scipy.signal as sig
    data, in_sr = sf.read(input_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=-1)
    if in_sr != sr:
        g = math.gcd(sr, in_sr)
        data = sig.resample_poly(data, sr // g, in_sr // g).astype(np.float32)
    duration = data.size / sr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x = torch.from_numpy(np.nan_to_num(data)).unsqueeze(0).to(device)
    with _lock, torch.inference_mode():
        t0 = time.perf_counter()
        y = model(x)
        elapsed = time.perf_counter() - t0
    est = y.squeeze(0).cpu().numpy().astype(np.float32)[: int(duration * sr)]

    if sr != SR:  # contract: always hand back 48 kHz
        g = math.gcd(SR, sr)
        est = sig.resample_poly(est, SR // g, sr // g).astype(np.float32)
    sf.write(output_path, np.clip(est, -1, 1), SR, subtype="PCM_16")
    return {
        "elapsed_seconds": round(elapsed, 4),
        "audio_duration_seconds": round(duration, 4),
        "rtf": round(elapsed / duration, 4) if duration > 0 else 0.0,
        "output_sr": SR,
        "model": meta["name"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    a = p.parse_args()
    print(json.dumps(enhance_file(a.inp, a.out, model_dir=a.model_dir)))
