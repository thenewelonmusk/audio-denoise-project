"""Build the paired eval set. Thin wrapper over training.dataset.synth.

Real data (recommended): point --clean at clean speech (LibriSpeech/VCTK/VIVOS
subset) and --noise at noise (MUSAN/DEMAND or your own recordings).

Placeholder (plumbing tests only): --synthetic generates tones as "speech"
so the full loop can be tested without any data. Do NOT report metrics from
a synthetic eval set.
"""

import argparse
import numpy as np
from pathlib import Path

from training.dataset import synth_pair, SR


def make_placeholder(dir_: Path, num: int, seed: int, kind: str) -> list[Path]:
    dir_.mkdir(parents=True, exist_ok=True)
    files = []
    rng = np.random.default_rng(seed)
    for i in range(num):
        t = np.arange(int(6 * SR)) / SR
        if kind == "clean":  # speech-ish: few sine formants with slow AM
            f0 = rng.uniform(120, 220)
            x = 0.3 * np.sin(2*np.pi*f0*t) + 0.2*np.sin(2*np.pi*2.3*f0*t) + 0.1*np.sin(2*np.pi*3.7*f0*t)
            x *= (0.6 + 0.4*np.sin(2*np.pi*3*t))
        else:  # noise: filtered white noise
            from scipy.signal import butter, sosfilt
            x = rng.standard_normal(t.size).astype(np.float32) * 0.25
            sos = butter(4, [200, 6000], "bandpass", fs=SR, output="sos")
            x = sosfilt(sos, x).astype(np.float32)
        files.append(dir_ / f"{kind}_{i:04d}.wav")
        import soundfile as sf
        sf.write(files[-1], np.clip(x, -1, 1).astype(np.float32), SR, subtype="PCM_16")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=None)
    ap.add_argument("--noise", default=None)
    ap.add_argument("--out", default="data/eval/paired")
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synthetic", action="store_true", help="generate placeholder tones/noise")
    a = ap.parse_args()

    if a.synthetic:
        tmp = Path("data/_synthetic_src")
        clean_files = make_placeholder(tmp / "clean", max(a.num, 4), a.seed, "clean")
        noise_files = make_placeholder(tmp / "noise", 3, a.seed, "noise")
        clean_dir, noise_dir = tmp / "clean", tmp / "noise"
    else:
        if not a.clean or not a.noise:
            raise SystemExit("Give --clean and --noise dirs, or use --synthetic.")
        clean_dir, noise_dir = Path(a.clean), Path(a.noise)

    cleans = sorted(clean_dir.glob("*.wav"))
    noises = sorted(noise_dir.glob("*.wav"))
    snrs = [0, 5, 10, 15]
    for i in range(a.num):
        synth_pair(cleans[i % len(cleans)], noises[(i // len(cleans)) % len(noises)],
                   Path(a.out), i, float(snrs[i % len(snrs)]), a.seed)
    print(f"eval set: {a.num} pairs -> {a.out}")
    print("next: python -m eval.manifest build --dir data/eval/paired --out eval/manifests/paired_v1.json")


if __name__ == "__main__":
    main()
