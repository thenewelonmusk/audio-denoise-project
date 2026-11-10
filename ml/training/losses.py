"""Training losses.

Three experiment types, three loss situations:
1. scratch / fine-tune: si_sdr_loss + mrstft_loss on (clean target vs estimate)
2. distillation (response-based): same losses but the TARGET is the teacher's
   enhanced output (pre-generated wav dir), not the ground-truth clean.
   Feature/mask matching can be added later per-architecture; response
   distillation needs nothing special and usually captures most of the gain.
3. architecture search: losses are agnostic — any [B, T] -> [B, T] model works.
"""

import numpy as np
import torch
import torch.nn.functional as F

_MRSTFT_SIZES = (256, 512, 1024, 2048)

# --- loss registry: the extension point for custom training objectives ---
# Contract: fn(est, target, **kwargs) -> scalar tensor.
#   est, target: [B, T] float waveforms on device (raw audio; arch internals
#   are NOT visible here — for feature/matching losses, expose them via the
#   arch and add a dedicated train mode instead).
# Registered via --loss <name>, kwargs via --loss-kwargs JSON.

LOSS_REGISTRY: dict = {}


def register_loss(name: str):
    def deco(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return deco


def get_loss(name: str):
    if name not in LOSS_REGISTRY:
        raise KeyError(f"Unknown loss '{name}'. Registered: {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name]


# --- aux-loss registry: losses that need arch internals ---
# Contract: fn(est, target, aux, **kwargs) -> scalar tensor.
#   est, target: [B, T] waveforms as usual.
#   aux: dict = {"x": the input tensor (added by train.py)} + whatever the
#   arch's forward(x, return_aux=True) returned (document YOUR keys in the
#   arch). An arch opts in by accepting return_aux in its forward signature;
#   train.py detects this automatically (see stft_mask_net in archs.py).
# Feature-matching (FitNets) is intentionally absent: it needs the teacher
# model loaded in-loop plus projection layers per feature-shape pair - build
# it when you actually have two own-registry archs worth distilling between.

AUX_LOSS_REGISTRY: dict = {}


def register_aux_loss(name: str):
    def deco(fn):
        AUX_LOSS_REGISTRY[name] = fn
        return fn
    return deco


def get_aux_loss(name: str):
    if name not in AUX_LOSS_REGISTRY:
        raise KeyError(f"Unknown aux loss '{name}'. Registered: {sorted(AUX_LOSS_REGISTRY)}")
    return AUX_LOSS_REGISTRY[name]


def si_sdr_loss(est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    est = est - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    alpha = (target * est).sum(-1, keepdim=True) / ((target * target).sum(-1, keepdim=True) + 1e-10)
    proj, noise = alpha * target, est - alpha * target
    return -(10 * torch.log10((proj * proj).sum(-1) + 1e-10)
             - 10 * torch.log10((noise * noise).sum(-1) + 1e-10)).mean()


def mrstft_loss(est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = est.new_tensor(0.0)
    for n_fft in _MRSTFT_SIZES:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=est.device)
        e = torch.stft(est, n_fft, hop, window=window, return_complex=True)
        t = torch.stft(target, n_fft, hop, window=window, return_complex=True)
        loss = loss + F.l1_loss(e.abs(), t.abs())
        loss = loss + F.l1_loss(torch.log1p(e.abs()), torch.log1p(t.abs()))
    return loss / len(_MRSTFT_SIZES)


@register_loss("default")
def total_loss(est: torch.Tensor, target: torch.Tensor,
               si_sdr_weight: float = 1.0, stft_weight: float = 0.3) -> torch.Tensor:
    return si_sdr_weight * si_sdr_loss(est, target) + stft_weight * mrstft_loss(est, target)


@register_loss("si_sdr")
def si_sdr_only(est: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    return si_sdr_loss(est, target)


@register_loss("mrstft")
def mrstft_only(est: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    return mrstft_loss(est, target)


@register_aux_loss("stages")
def stages_loss(est: torch.Tensor, target: torch.Tensor, aux: dict | None = None,
                stage_weights: dict | None = None) -> torch.Tensor:
    """Multi-stage supervision for estimate-then-refine archs.

    aux contract: {"stages": {name: [B, T] intermediate waveform estimate}}.
    Sum of si_sdr between each stage and the same target. Weights via
    --loss-kwargs '{"stage_weights": {"noise_est": 0.5}}' (default 1.0).
    """
    assert aux and aux.get("stages"), "stages loss needs aux['stages'] from the arch"
    w = stage_weights or {}
    total = est.new_tensor(0.0)
    for name_, s in aux["stages"].items():
        total = total + w.get(name_, 1.0) * si_sdr_loss(s.float(), target.float())
    return total


@register_aux_loss("irm_mask")
def irm_mask_loss(est: torch.Tensor, target: torch.Tensor, aux: dict | None = None,
                  eps: float = 1e-8) -> torch.Tensor:
    """Ideal Ratio Mask supervision for STFT magnitude-masking archs.

    aux contract: {"mask": [B, F, T] post-sigmoid magnitude mask, "n_fft": int,
    "hop": int} ("x" is added by train.py). IRM = |clean|² / |noisy|² clamped
    to [0, 1]; MSE against the predicted mask. Both sides compute the STFT
    with torch defaults (hann window, center=True) - the arch must do the
    same or the frame shapes won't match.
    """
    assert aux and {"mask", "n_fft", "hop"} <= aux.keys(), \
        "irm_mask needs {'mask', 'n_fft', 'hop'} from the arch"
    x, n_fft, hop = aux["x"], aux["n_fft"], aux["hop"]
    win = torch.hann_window(n_fft, device=x.device)
    X = torch.stft(x.float(), n_fft, hop, window=win, return_complex=True).abs()
    C = torch.stft(target.float(), n_fft, hop, window=win, return_complex=True).abs()
    irm = (C.pow(2) / (X.pow(2) + eps)).clamp(0.0, 1.0)
    if irm.shape != aux["mask"].shape:
        raise ValueError(f"arch mask {tuple(aux['mask'].shape)} != IRM {tuple(irm.shape)} "
                         f"(check n_fft/hop and windowing match)")
    return F.mse_loss(aux["mask"].float(), irm)
