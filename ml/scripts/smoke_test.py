"""End-to-end smoke test of the ml harness. No real data needed.

Generates a tiny synthetic eval set, builds a manifest, trains the tiny_unet
plumbing model for 1 epoch, exports it, evaluates DFN3 + tiny model, and
sanity-checks metric behavior (clean-vs-clean must beat clean-vs-noisy).

Run from ml/:  python -m scripts.smoke_test
Expect a few minutes (DFN3 download happens once, on first use).
"""

import json
import subprocess
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]


def sh(*args: str) -> None:
    print(f"\n>>> {' '.join(args)}")
    r = subprocess.run([sys.executable, "-m", *args], cwd=ML)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(args)}")


def main():
    print("== 1. synthetic eval set ==")
    sh("scripts.build_eval_set", "--synthetic", "--num", "4")

    print("== 2. manifest ==")
    sh("eval.manifest", "build", "--dir", "data/eval/paired",
       "--out", "eval/manifests/paired_v1.json")

    print("== 3. metric sanity ==")
    sys.path.insert(0, str(ML))
    import numpy as np
    import soundfile as sf
    from eval import metrics
    noisy, sr = sf.read(str(ML / "data/eval/paired/noisy_0000.wav"), dtype="float32")
    clean, _ = sf.read(str(ML / "data/eval/paired/clean_0000.wav"), dtype="float32")
    self_score = metrics.si_sdr(clean, clean)
    noisy_score = metrics.si_sdr(clean, noisy)
    assert self_score > noisy_score, f"metric sanity failed: {self_score} vs {noisy_score}"
    print(f"si_sdr(clean,clean)={self_score:.1f} > si_sdr(clean,noisy)={noisy_score:.1f}  ok")

    print("== 4. train tiny_unet (1 epoch, plumbing only) ==")
    sh("training.train", "--arch", "tiny_unet", "--epochs", "1",
       "--data", "data/eval/paired", "--batch", "1", "--segment-seconds", "2")

    print("== 5. export ==")
    ckpts = sorted((ML / "runs").glob("train_tiny_unet_*/ckpt.pt"))
    latest = ckpts[-1].parent
    sh("training.export", "--ckpt", str(latest / "ckpt.pt"), "--name", "tiny_smoke")

    print("== 6. eval DFN3 + exported tiny model ==")
    cfg = ML / "eval/adapters.yaml"
    text = cfg.read_text(encoding="utf-8")
    if "tiny_smoke" not in text:
        cfg.write_text(text + (
            "  tiny_smoke:\n    module: adapters.generic\n"
            "    args: { model_dir: models/tiny_smoke }\n"), encoding="utf-8")
    sh("eval.runner", "--manifest", "eval/manifests/paired_v1.json",
       "--runs", "dfn3", "tiny_smoke")

    print("\nSMOKE TEST PASSED")
    print("Note: synthetic data -> numbers are meaningless; this only proves plumbing.")


if __name__ == "__main__":
    main()
