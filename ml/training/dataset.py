"""Pair synthesis: (clean speech, noise, SNR) -> (noisy, clean) at 48 kHz mono.

Deterministic given (seed, source files, output index): same inputs always
produce the same eval set, which is what makes dataset_hash meaningful.

CLI: python -m training.dataset synth --clean data/src_clean --noise data/src_noise \
        --out data/eval/paired --num 20 --seed 42
"""

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000
SEGMENT_SECONDS = 6.0


def _load_48k_mono(path: Path) -> np.ndarray:
    import scipy.signal as sig
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=-1)
    if sr != SR:
        g = math.gcd(SR, sr)
        data = sig.resample_poly(data, SR // g, sr // g).astype(np.float32)
    return np.nan_to_num(data).astype(np.float32)


def _random_segment(x: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if x.size <= length:
        out = np.zeros(length, dtype=np.float32)
        if x.size > 0:
            out[: x.size] = x
        return out
    start = rng.integers(0, x.size - length)
    return x[start : start + length].copy()

def _match_rms(x: np.ndarray, target_rms: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2))) + 1e-10
    return (x * (target_rms / rms)).astype(np.float32)


def _active_region(x: np.ndarray, sr: int, win_s: float = 0.05,
                   thresh: float = 0.1) -> np.ndarray:
    """Trim leading/trailing low-energy windows. UrbanSound8K clips are often a
    single event (bark, horn) inside seconds of silence - without this, mixed
    pairs get a much higher effective SNR than the requested one."""
    w = int(win_s * sr)
    if x.size <= w:
        return x
    n_win = x.size // w
    env = np.sqrt((x[: n_win * w].reshape(n_win, w) ** 2).mean(axis=1))
    active = env > thresh * (env.max() + 1e-10)
    if not active.any():
        return x
    first = int(np.argmax(active))
    last = int(len(active) - np.argmax(active[::-1]) - 1)
    return x[first * w : (last + 1) * w]


def synth_pair(clean_file: Path, noise_file: Path, out_dir: Path, idx: int,
               snr_db: float, seed: int) -> None:
    rng = np.random.default_rng(seed + idx)
    n = int(SEGMENT_SECONDS * SR)
    # pairs are as long as the utterance (max SEGMENT_SECONDS): zero-padding a
    # short clean clip would create a speech-free tail with wrong effective SNR
    clean_raw = _load_48k_mono(clean_file)
    n = min(n, clean_raw.size)
    clean = _random_segment(clean_raw, n, rng)
    noise = _active_region(_load_48k_mono(noise_file), SR)
    # short noise clips (e.g. UrbanSound8K ~5 s): tile before cropping
    if noise.size < n:
        noise = np.tile(noise, int(np.ceil(n / noise.size)) + 1)
    noise = _random_segment(noise, n, rng)

    clean_rms = float(np.sqrt(np.mean(clean ** 2))) + 1e-10
    noise = _match_rms(noise, clean_rms * 10 ** (-snr_db / 20.0))
    noisy = np.clip(clean + noise, -1.0, 1.0).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / f"clean_{idx:04d}.wav", clean, SR, subtype="PCM_16")
    sf.write(out_dir / f"noisy_{idx:04d}.wav", noisy, SR, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["synth"])
    ap.add_argument("--clean", required=True, help="dir of clean speech wavs")
    ap.add_argument("--noise", required=True, help="dir of noise wavs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snr-list", type=float, nargs="+",
                    default=[0, 5, 10, 15])
    a = ap.parse_args()

    cleans = sorted(Path(a.clean).rglob("*.wav"))
    noises = sorted(Path(a.noise).rglob("*.wav"))
    if not cleans or not noises:
        raise SystemExit("Need at least one clean and one noise wav.")
    out_dir = Path(a.out)
    pick = np.random.default_rng(a.seed)  # deterministic noise spread across pairs
    for i in range(a.num):
        snr = float(a.snr_list[i % len(a.snr_list)])
        c = cleans[i % len(cleans)]
        nz = noises[int(pick.integers(0, len(noises)))]
        synth_pair(c, nz, out_dir, i, snr, a.seed)
        print(f"[{i+1}/{a.num}] {c.name} + {nz.name} @ {snr:.0f} dB")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
