"""Architecture registry — the extension point for "changing architecture".

Add your model here, give it this contract, and the whole harness (train,
export, generic adapter, eval, future src/ engine) picks it up automatically:

    @register("my_arch")
    class MyDenoiser(nn.Module):
        def __init__(self, **config): ...   # config saved to metadata.json
        def forward(self, x):               # [B, T] float32 -> [B, T] float32
            ...
            return est

Contract details:
- Input/output are raw waveforms, same length, float32. native_sr defaults to
  48000 (declare a different one in metadata via ARCH_NATIVE_SR if the model
  is natively 16 kHz — resampling happens in the adapter).
- Keep the model small enough for 4 GB VRAM or use short segments in train.py.
- Optional: accept `return_aux: bool = False` in forward to expose internals
  for aux losses (see training/losses.py AUX_LOSS_REGISTRY). True must return
  (est, aux_dict); False keeps the plain [B, T] contract, so exported models
  work unchanged in the app. One worked example: stft_mask_net below.
"""

import torch
import torch.nn as nn

REGISTRY: dict[str, type[nn.Module]] = {}
ARCH_NATIVE_SR: dict[str, int] = {}  # arch name -> native sample rate (default 48000)


def register(name: str, native_sr: int = 48000):
    def deco(cls):
        REGISTRY[name] = cls
        ARCH_NATIVE_SR[name] = native_sr
        return cls
    return deco


def build(name: str, **config) -> nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"Unknown arch '{name}'. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**config)


@register("tiny_unet")
class TinyDenoiser(nn.Module):
    """Minimal 1D conv encoder-decoder. Plumbing reference, not a serious model.

    Proves the full loop: train -> export -> generic adapter -> eval. Replace
    with real architectures (GTCRN, custom CRN, distilled student, ...).
    """

    def __init__(self, width: int = 16, depth: int = 3):
        super().__init__()
        chs = [width * (2 ** i) for i in range(depth)]
        self.enc = nn.ModuleList()
        in_ch = 1
        for c in chs:
            self.enc.append(nn.Sequential(
                nn.Conv1d(in_ch, c, 15, stride=2, padding=7), nn.PReLU()))
            in_ch = c
        # Decoder mirrors the encoder: each block upsamples to its skip's
        # length, concatenates, and halves the channel count. The bottleneck
        # output itself is not concatenated; one final upsample restores
        # the original time resolution.
        self.dec = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.dec.append(nn.Sequential(
                nn.Conv1d(chs[i] + chs[i - 1], chs[i - 1], 5, padding=2), nn.PReLU()))
        self.out = nn.Conv1d(chs[0], 1, 5, padding=2)

    def forward(self, x):
        h = x.unsqueeze(1)
        skips = []
        for layer in self.enc:
            h = layer(h)
            skips.append(h)
        h = skips.pop()  # bottleneck, kept as-is
        for layer in self.dec:
            skip = skips.pop()
            h = nn.functional.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = layer(torch.cat([h, skip], dim=1))
        h = nn.functional.interpolate(h, size=x.shape[-1], mode="nearest")
        return self.out(h).squeeze(1)


@register("stft_mask_net")
class StftMaskNet(nn.Module):
    """Tiny STFT magnitude-mask denoiser — the worked example for the aux lane.

    Predicts a sigmoid magnitude mask over the noisy STFT (noisy phase reused),
    reconstructs via iSTFT, and exposes {"mask", "n_fft", "hop"} so
    training.train --loss irm_mask can supervise the mask directly. Real
    masking archs (GTCRN and friends) follow this exact shape; also serves as
    a second, spectral-domain arch for experiments.
    """

    def __init__(self, n_fft: int = 512, hop: int = 128, width: int = 24):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 5, padding=2), nn.PReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.PReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.PReLU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, x, return_aux: bool = False):
        win = torch.hann_window(self.n_fft, device=x.device)
        spec = torch.stft(x.float(), self.n_fft, self.hop, window=win,
                          return_complex=True)
        mask = torch.sigmoid(self.net(torch.log1p(spec.abs()).unsqueeze(1))).squeeze(1)
        est = torch.istft(spec * mask, self.n_fft, self.hop, window=win,
                          length=x.shape[-1])
        if return_aux:
            return est, {"mask": mask, "n_fft": self.n_fft, "hop": self.hop}
        return est
