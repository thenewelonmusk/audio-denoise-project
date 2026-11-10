"""Dataset manifests: build, hash, load.

A manifest pins the eval set: file list + sha256 of every file + a dataset_hash
(hash of the file list). All results reference dataset_hash, so numbers are only
comparable within the same version.

Usage: python -m eval.manifest build --dir data/eval/paired --out eval/manifests/paired_v1.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(paired_dir: str, out_path: str, version: str = "v1") -> dict:
    d = Path(paired_dir)
    noisy_files = sorted(d.glob("noisy_*.wav"))
    if not noisy_files:
        raise SystemExit(f"No noisy_*.wav files in {d}")
    entries = []
    for noisy in noisy_files:
        clean = noisy.parent / noisy.name.replace("noisy_", "clean_", 1)
        if not clean.exists():
            raise SystemExit(f"Missing pair for {noisy.name}: {clean.name}")
        info = sf.info(str(noisy))
        entries.append({
            "noisy": noisy.name,
            "clean": clean.name,
            "sha256_noisy": _sha256(noisy),
            "sha256_clean": _sha256(clean),
            "sample_rate": info.samplerate,
            "frames": info.frames,
        })
    # dataset hash: hash of the sorted (name, sha) pairs
    hasher = hashlib.sha256()
    for e in entries:
        hasher.update(f"{e['noisy']}:{e['sha256_noisy']};{e['clean']}:{e['sha256_clean']}".encode())
    manifest = {
        "version": version,
        "sample_rate": entries[0]["sample_rate"],
        "dataset_hash": hasher.hexdigest()[:16],
        "num_pairs": len(entries),
        "files": entries,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {out} ({manifest['num_pairs']} pairs, hash={manifest['dataset_hash']})")
    return manifest


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "show"])
    p.add_argument("--dir", dest="dir")
    p.add_argument("--out", dest="out")
    p.add_argument("--path", dest="path")
    a = p.parse_args()
    if a.cmd == "build":
        build(a.dir, a.out)
    else:
        m = load(a.path)
        print(f"{m['version']} hash={m['dataset_hash']} pairs={m['num_pairs']}")
