"""DeepFilterNet adapter (DFN2 and DFN3 via --model).

Contract: enhance_file(input_path, output_path, **kwargs) -> dict
  Reads 48 kHz mono float32 WAV, writes same-length 48 kHz mono WAV,
  returns {"elapsed_seconds", "audio_duration_seconds", "rtf", "output_sr"}.

CLI: python -m adapters.dfnet --model DeepFilterNet3 --in a.wav --out b.wav
"""

import argparse
import time
import threading

import numpy as np
import soundfile as sf
import torch

SR = 48000
_lock = threading.Lock()
_models = {}  # (model_name, post_filter) -> (model, df_state)


def _get_model(model_name: str, post_filter: bool):
    key = (model_name, post_filter)
    if key not in _models:
        from df.enhance import init_df
        model, df_state, _ = init_df(model_base_dir=model_name, post_filter=post_filter)
        assert df_state.sr() == SR
        _models[key] = (model, df_state)
    return _models[key]


def _to_48k_mono(path: str) -> np.ndarray:
    import math
    import scipy.signal as sig
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=-1)
    if sr != SR:
        g = math.gcd(SR, sr)
        data = sig.resample_poly(data, SR // g, sr // g).astype(np.float32)
    return np.nan_to_num(data).astype(np.float32)


def enhance_file(input_path: str, output_path: str, *, model: str = "DeepFilterNet3",
                 post_filter: bool = False, atten_lim_db: float | None = None) -> dict:
    from df.enhance import enhance
    model_obj, df_state = _get_model(model, post_filter)
    audio = _to_48k_mono(input_path)
    duration = audio.size / SR

    # enhance() requires length multiple of the DF hop
    hop = df_state.hop_size()
    pad = (-audio.size) % hop
    if pad:
        audio = np.pad(audio, (0, pad))

    tensor = torch.from_numpy(audio).unsqueeze(0)  # CPU tensor; lib moves to device
    with _lock, torch.inference_mode():
        t0 = time.perf_counter()
        out = enhance(model_obj, df_state, tensor, atten_lim_db=atten_lim_db)
        elapsed = time.perf_counter() - t0

    est = out.squeeze(0).numpy().astype(np.float32)[: int(duration * SR)]
    sf.write(output_path, np.clip(est, -1, 1), SR, subtype="PCM_16")
    return {
        "elapsed_seconds": round(elapsed, 4),
        "audio_duration_seconds": round(duration, 4),
        "rtf": round(elapsed / duration, 4) if duration > 0 else 0.0,
        "output_sr": SR,
        "model": model,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="DeepFilterNet3")
    p.add_argument("--post-filter", action="store_true")
    p.add_argument("--atten-lim-db", type=float, default=None)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    a = p.parse_args()
    import json
    print(json.dumps(enhance_file(a.inp, a.out, model=a.model,
                                  post_filter=a.post_filter,
                                  atten_lim_db=a.atten_lim_db)))
