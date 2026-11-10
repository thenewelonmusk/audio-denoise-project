"""Metrics for the paired eval tier.

- si_sdr: fullband, 48 kHz (numpy, reference-free math, has clean ref)
- stoi:   computed at 16 kHz (STOI is defined for narrowband/wideband speech)
- pesq_torch_loss: torch-pesq objective at 16 kHz. LOWER IS BETTER (range [0, inf)).
  This is a differentiable PESQ approximation, not ITU-T P.862. Proper PESQ
  needs the `pesq` C package (no Windows wheels on Py 3.10) — run that on
  Linux/WSL if certified numbers are required.
- dnsmos: DNSMOS P.835 via onnxruntime. Requires model files in eval/dnsmos/
  (see scripts/download_dnsmos.py). Returns None if unavailable.
"""

import numpy as np
import torch
from pystoi import stoi

_EVAL_SR = 16000


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    import math
    import scipy.signal as sig
    if sr_from == sr_to:
        return x.astype(np.float32)
    g = math.gcd(sr_to, sr_from)
    return sig.resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref.astype(np.float64) - ref.mean()
    est = est.astype(np.float64) - est.mean()
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-10)
    proj, noise = alpha * ref, est - alpha * ref
    return float(10 * np.log10((proj @ proj + 1e-10) / (noise @ noise + 1e-10)))


def stoi_score(ref: np.ndarray, est: np.ndarray, fs: int) -> float:
    r, e = _resample(ref, fs, _EVAL_SR), _resample(est, fs, _EVAL_SR)
    n = min(r.size, e.size)
    return float(stoi(r[:n], e[:n], _EVAL_SR, extended=False))


def pesq_torch_loss(ref: np.ndarray, est: np.ndarray, fs: int) -> float:
    """torch-pesq objective. Internally resamples to 16 kHz and band-limits
    325-3250 Hz. LOWER IS BETTER (range [0, inf)). factor only scales the
    value; fixed at 1.0 so numbers are comparable across runs."""
    from torch_pesq import PesqLoss
    r = torch.from_numpy(ref.astype(np.float32)).unsqueeze(0)
    e = torch.from_numpy(est.astype(np.float32)).unsqueeze(0)
    n = min(r.size(1), e.size(1))
    with torch.no_grad():
        loss = PesqLoss(factor=1.0, sample_rate=fs)(r[:, :n], e[:, :n])
    return float(loss.mean().item())


def dnsmos(audio: np.ndarray, fs: int) -> dict | None:
    """DNSMOS P.835 (SIG/BAK/OVRL). Needs onnx models in eval/dnsmos/."""
    import onnxruntime
    from pathlib import Path
    d = Path(__file__).parent / "dnsmos"
    model_path = d / "model_v8.onnx"
    if not model_path.exists():
        return None
    raise NotImplementedError(
        "DNSMOS runner not ported yet: adapt the reference implementation from "
        "microsoft/DNS-Challenge (DNSMOS/dnsmos_local.py) to run model_v8.onnx."
    )


METRIC_VERSION = "v0.1"


def compute(ref: np.ndarray, est: np.ndarray, fs: int) -> dict:
    out = {
        "si_sdr": round(si_sdr(ref, est), 3),
        "stoi": round(stoi_score(ref, est, fs), 4),
        "pesq_torch_loss": round(pesq_torch_loss(ref, est, fs), 4),
    }
    dm = dnsmos(est, fs)
    if dm:
        out["dnsmos"] = dm
    return out
