"""
Pure Python / NumPy / SciPy implementation of DeepFilterLib (`libdf` 0.5.6)
for Python 3.12+ where prebuilt Rust wheels are unavailable on PyPI.
Mathematically matches `DeepFilterLib-0.5.6/src/lib.rs` and `deep_filter/src/lib.rs`.
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

import numpy as np
import scipy.signal as signal

MEAN_NORM_INIT = (-60.0, -90.0)
UNIT_NORM_INIT = (0.001, 0.0001)


def _freq2erb(freq_hz: float) -> float:
    return 9.265 * math.log1p(freq_hz / (24.7 * 9.265))


def _erb2freq(n_erb: float) -> float:
    return 24.7 * 9.265 * (math.exp(n_erb / 9.265) - 1.0)


def _erb_fb(sr: int, fft_size: int, nb_bands: int, min_nb_freqs: int) -> np.ndarray:
    nyq_freq = sr // 2
    freq_width = float(sr) / float(fft_size)
    erb_low = _freq2erb(0.0)
    erb_high = _freq2erb(float(nyq_freq))
    erb = [0] * nb_bands
    step = (erb_high - erb_low) / float(nb_bands)
    prev_freq = 0
    freq_over = 0
    for i in range(1, nb_bands + 1):
        f = _erb2freq(erb_low + i * step)
        fb = int(math.floor(f / freq_width + 0.5))
        nb_freqs = fb - prev_freq - freq_over
        if nb_freqs < min_nb_freqs:
            freq_over = min_nb_freqs - nb_freqs
            nb_freqs = min_nb_freqs
        else:
            freq_over = 0
        erb[i - 1] = nb_freqs
        prev_freq = fb
    erb[nb_bands - 1] += 1
    too_large = sum(erb) - (fft_size // 2 + 1)
    if too_large > 0:
        erb[nb_bands - 1] -= too_large
    return np.asarray(erb, dtype=np.uintp)


class DF:
    """DeepFilter state used for STFT analysis, synthesis, and ERB filterbank."""

    def __init__(
        self,
        sr: int,
        fft_size: int,
        hop_size: int,
        nb_bands: Optional[int] = 32,
        min_nb_erb_freqs: Optional[int] = 1,
    ) -> None:
        nb_bands = 32 if nb_bands is None else int(nb_bands)
        min_nb_erb_freqs = 1 if min_nb_erb_freqs is None else int(min_nb_erb_freqs)
        assert hop_size * 2 <= fft_size, "hop_size * 2 must be <= fft_size"

        self._sr = int(sr)
        self._frame_size = int(hop_size)
        self._window_size = int(fft_size)
        self._freq_size = fft_size // 2 + 1
        self._erb = _erb_fb(self._sr, self._window_size, nb_bands, min_nb_erb_freqs)

        window_size_h = fft_size / 2.0
        idx = np.arange(fft_size, dtype=np.float64)
        s = np.sin(0.5 * math.pi * (idx + 0.5) / window_size_h)
        self._window = np.sin(0.5 * math.pi * s * s).astype(np.float32)
        self._wnorm = float(1.0 / ((fft_size ** 2) / (2.0 * hop_size)))

        mem_len = fft_size - hop_size
        self._analysis_mem = None
        self._synthesis_mem = None

    def reset(self) -> None:
        if self._analysis_mem is not None:
            self._analysis_mem.fill(0.0)
        if self._synthesis_mem is not None:
            self._synthesis_mem.fill(0.0)

    def sr(self) -> int:
        return self._sr

    def fft_size(self) -> int:
        return self._window_size

    def hop_size(self) -> int:
        return self._frame_size

    def nb_erb(self) -> int:
        return int(self._erb.size)

    def erb_widths(self) -> np.ndarray:
        return self._erb.copy()

    def fft_window(self) -> np.ndarray:
        return self._window.copy()

    def analysis(self, input: np.ndarray, reset: Optional[bool] = True) -> np.ndarray:
        inp = np.ascontiguousarray(input, dtype=np.float32)
        if inp.ndim != 2:
            raise ValueError(f"Expected 2D input [C, T], got shape {inp.shape}")
        channels, total_t = inp.shape
        H = self._frame_size
        W = self._window_size
        F = self._freq_size
        M = W - H
        freq_steps = total_t // H
        output = np.zeros((channels, freq_steps, F), dtype=np.complex64)
        if freq_steps == 0:
            return output

        used_t = freq_steps * H
        do_reset = True if reset is None else bool(reset)

        if self._analysis_mem is None or self._analysis_mem.shape[0] != channels:
            self._analysis_mem = np.zeros((channels, M), dtype=np.float32)
        elif do_reset:
            self._analysis_mem.fill(0.0)

        for ch in range(channels):
            padded = np.concatenate([self._analysis_mem[ch], inp[ch, :used_t]])
            frames = np.lib.stride_tricks.sliding_window_view(padded, W)[::H]
            windowed = frames * self._window
            spec = (np.fft.rfft(windowed, n=W, axis=-1) * self._wnorm).astype(np.complex64)
            output[ch] = spec
            if M > 0:
                self._analysis_mem[ch, :] = padded[used_t : used_t + M]

        return output

    def synthesis(self, input: np.ndarray, reset: Optional[bool] = True) -> np.ndarray:
        inp = np.ascontiguousarray(input, dtype=np.complex64)
        if inp.ndim != 3:
            raise ValueError(f"Expected 3D input [C, T, F], got shape {inp.shape}")
        channels, freq_steps, _ = inp.shape
        H = self._frame_size
        W = self._window_size
        M = W - H
        out_steps = freq_steps * H
        output = np.zeros((channels, out_steps), dtype=np.float32)
        if freq_steps == 0:
            return output

        do_reset = True if reset is None else bool(reset)
        split = M - H

        if self._synthesis_mem is None or self._synthesis_mem.shape[0] != channels:
            self._synthesis_mem = np.zeros((channels, M), dtype=np.float32)
        elif do_reset:
            self._synthesis_mem.fill(0.0)

        for ch in range(channels):
            # Match Rust realfft unnormalized inverse FFT
            x_all = (np.fft.irfft(inp[ch], n=W, axis=-1) * W).astype(np.float32)
            x_all *= self._window

            if split == 0:
                # Standard 50% overlap (W == 2 * H): vectorize overlap-add
                prev = np.empty((freq_steps, H), dtype=np.float32)
                prev[0] = self._synthesis_mem[ch]
                if freq_steps > 1:
                    prev[1:] = x_all[:-1, H:]
                output[ch] = (x_all[:, :H] + prev).reshape(-1)
                self._synthesis_mem[ch, :] = x_all[-1, H:]
            else:
                mem = self._synthesis_mem[ch]
                for t in range(freq_steps):
                    x = x_all[t]
                    output[ch, t * H : (t + 1) * H] = x[:H] + mem[:H]
                    if split > 0:
                        mem[:split] = mem[H : H + split] + x[H : H + split]
                        mem[split:] = x[H + split :]

        return output


def erb(
    input: np.ndarray,
    erb_fb: Union[np.ndarray, List[int]],
    db: Optional[bool] = True,
) -> np.ndarray:
    inp = np.asarray(input, dtype=np.complex64)
    fb = np.asarray(erb_fb, dtype=np.int64).ravel()
    indim = inp.ndim
    if indim not in (2, 3, 4):
        raise ValueError(f"Dimension not supported for erb: {indim}")

    orig_prefix = inp.shape[:-1]
    f = inp.shape[-1]
    power = inp.real * inp.real + inp.imag * inp.imag  # float32

    indices = np.cumsum(fb)[:-1]
    inv_widths = (1.0 / fb).astype(np.float32)
    # Sum within each ERB band along the last axis
    splits = np.split(power, indices, axis=-1)
    out = np.stack([s.sum(axis=-1) for s in splits], axis=-1) * inv_widths

    use_db = True if db is None else bool(db)
    if use_db:
        out = np.log10(out + np.float32(1e-10)) * np.float32(10.0)

    return out.reshape(orig_prefix + (fb.size,)).astype(np.float32)


def erb_inv(
    input: np.ndarray,
    erb_fb: Union[np.ndarray, List[int]],
) -> np.ndarray:
    inp = np.asarray(input, dtype=np.float32)
    fb = np.asarray(erb_fb, dtype=np.int64).ravel()
    indim = inp.ndim
    if indim not in (2, 3, 4):
        raise ValueError(f"Dimension not supported for erb_inv: {indim}")
    if inp.shape[-1] != fb.size:
        raise ValueError(
            f"Number of erb bands do not match with input: {inp.shape[-1]}, {fb.size}"
        )
    return np.repeat(inp, fb, axis=-1).astype(np.float32)


def erb_norm(
    erb: np.ndarray,
    alpha: float,
    state: Optional[np.ndarray] = None,
) -> np.ndarray:
    # Input shape [C, T, F], state shape [C, F]
    x = np.array(erb, dtype=np.float32, copy=True)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array [C, T, F], got {x.shape}")
    ch, t_steps, nb_bands = x.shape
    alpha_f = float(alpha)

    if state is None:
        s0 = np.linspace(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], nb_bands, dtype=np.float32)
        s_init = np.broadcast_to(s0[None, :], (ch, nb_bands)).copy()
    else:
        s_init = np.array(state, dtype=np.float32, copy=True)

    if t_steps == 0:
        return x

    # Recurrence: s[t] = (1 - alpha) * x[t] + alpha * s[t - 1]
    b = np.array([1.0 - alpha_f], dtype=np.float64)
    a = np.array([1.0, -alpha_f], dtype=np.float64)
    zi = (s_init[:, None, :].astype(np.float64)) * alpha_f
    s_all, _ = signal.lfilter(b, a, x.astype(np.float64), axis=1, zi=zi)
    out = ((x - s_all.astype(np.float32)) / np.float32(40.0)).astype(np.float32)
    return out


def unit_norm_init(num_freq_bins: int) -> np.ndarray:
    return np.linspace(
        UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], int(num_freq_bins), dtype=np.float32
    ).reshape(1, int(num_freq_bins))


def unit_norm(
    spec: np.ndarray,
    alpha: float,
    state: Optional[np.ndarray] = None,
) -> np.ndarray:
    # Input shape [C, T, F], state shape [C, F]
    x = np.array(spec, dtype=np.complex64, copy=True)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D complex array [C, T, F], got {x.shape}")
    ch, t_steps, n_freqs = x.shape
    alpha_f = float(alpha)

    if state is None:
        s0 = np.linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], n_freqs, dtype=np.float32)
        s_init = np.broadcast_to(s0[None, :], (ch, n_freqs)).copy()
    else:
        s_init = np.array(state, dtype=np.float32, copy=True)

    if t_steps == 0:
        return x

    mag = np.abs(x).astype(np.float64)
    b = np.array([1.0 - alpha_f], dtype=np.float64)
    a = np.array([1.0, -alpha_f], dtype=np.float64)
    zi = (s_init[:, None, :].astype(np.float64)) * alpha_f
    s_all, _ = signal.lfilter(b, a, mag, axis=1, zi=zi)
    denom = np.sqrt(s_all.astype(np.float32))
    return (x / denom).astype(np.complex64)


__all__ = [
    "DF",
    "erb",
    "erb_inv",
    "erb_norm",
    "unit_norm",
    "unit_norm_init",
]
