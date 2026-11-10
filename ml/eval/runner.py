"""Eval runner: run adapter(s) over a manifest, compute metrics, write results.

Usage (from ml/):
  python -m eval.runner --manifest eval/manifests/paired_v1.json --runs dfn3 dfn2

Results land in runs/eval/<timestamp>_<name>/result.json following the frozen
schema, plus per-file rows in rows.csv. A result JSON is only comparable with
results sharing the same dataset_hash and METRIC_VERSION.
"""

import argparse
import csv
import importlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.manifest import load as load_manifest
from eval import metrics


def _hardware() -> dict:
    import torch
    hw = {"platform": platform.platform(), "cpu": platform.processor(),
          "threads": str(__import__("os").cpu_count())}
    if torch.cuda.is_available():
        hw["gpu"] = torch.cuda.get_device_name(0)
    return hw


def run_one(run_name: str, module_name: str, args: dict, manifest: dict,
            data_dir: Path, out_dir: Path) -> dict:
    mod = importlib.import_module(module_name)
    sr = manifest["sample_rate"]
    rows = []
    est_dir = out_dir / "est"
    est_dir.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(manifest["files"]):
        noisy = data_dir / entry["noisy"]
        clean = data_dir / entry["clean"]
        est_path = est_dir / f"est_{entry['noisy']}"
        info = mod.enhance_file(str(noisy), str(est_path), **args)
        clean_arr, clean_sr = sf.read(str(clean), dtype="float32")
        est_arr, est_sr = sf.read(str(est_path), dtype="float32")
        if clean_arr.ndim > 1:
            clean_arr = clean_arr.mean(axis=-1)
        if est_arr.ndim > 1:
            est_arr = est_arr.mean(axis=-1)
        n = min(clean_arr.size, est_arr.size)
        m = metrics.compute(clean_arr[:n], est_arr[:n], clean_sr)
        rows.append({"file": entry["noisy"], **m,
                     "rtf": info.get("rtf"), "elapsed": info.get("elapsed_seconds")})
        print(f"  [{i+1}/{len(manifest['files'])}] {entry['noisy']}: {m}")

    agg = {}
    for key in ("si_sdr", "stoi", "pesq_torch_loss"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        agg[key] = round(float(np.mean(vals)), 4) if vals else None
    rtfs = [r["rtf"] for r in rows if r.get("rtf") is not None]
    agg["rtf_mean"] = round(float(np.mean(rtfs)), 4) if rtfs else None

    with open(out_dir / "rows.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "run_name": run_name,
        "module": module_name,
        "args": args,
        "dataset_hash": manifest["dataset_hash"],
        "manifest_version": manifest["version"],
        "metric_version": metrics.METRIC_VERSION,
        "metrics": agg,
        "rtf": {"file_mode_mean": agg.get("rtf_mean")},
        "hardware": _hardware(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="eval/adapters.yaml")
    ap.add_argument("--runs", nargs="+", required=True, help="run names from the config")
    ap.add_argument("--data-dir", default="data/eval/paired")
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    manifest = load_manifest(a.manifest)
    data_dir = Path(a.data_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {}
    for name in a.runs:
        spec = cfg["runs"][name]
        out_dir = Path("runs/eval") / f"{stamp}_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] module={spec['module']} args={spec['args']}")
        t0 = time.perf_counter()
        result = run_one(name, spec["module"], spec["args"], manifest, data_dir, out_dir)
        print(f"[{name}] done in {time.perf_counter()-t0:.1f}s -> {out_dir/'result.json'}")
        print(f"[{name}] {result['metrics']}")
        all_results[name] = result

    summary = Path("runs/eval") / f"summary_{stamp}.json"
    summary.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"summary -> {summary}")


if __name__ == "__main__":
    main()
