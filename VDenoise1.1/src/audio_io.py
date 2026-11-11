"""
src/audio_io.py - Production-Grade Audio Input/Output and Signal Processing Layer.

Implements Interface Contracts CT-01 (inspect_and_prepare_audio) and CT-03 (summarize_timing),
alongside signal processing utilities including polyphase resampling, multi-channel downmixing,
scientific STFT spectrogram generation, and large-file chunking with Hanning window
Overlap-Add (OLA) reassembly.

Conforms strictly to PLAN.md, PROJECT.md, and Anti-AI-Slop DAW rackmount visual specifications.
"""

from __future__ import annotations

import logging
import math
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile as sf

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. CUSTOM AUDIO VALIDATION EXCEPTION
# ==============================================================================

class AudioValidationError(Exception):
    """
    Exception raised when an audio file fails integrity validation,
    such as being corrupt, 0 bytes, missing, or unreadable by audio decoders.
    """
    pass


# ==============================================================================
# 2. AUDIO FORMAT CONVERSION & NORMALIZATION UTILITIES
# ==============================================================================

def downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert multi-channel audio to single-channel (mono) float32 array.

    Parameters:
        audio: 1D or 2D numpy array representing audio samples.

    Returns:
        1D numpy array of dtype float32 containing mono audio.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        return np.zeros(0, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # Typical soundfile shape: (samples, channels) where channels <= 8
        if arr.shape[1] <= 8 and arr.shape[0] >= arr.shape[1]:
            return np.mean(arr, axis=1, dtype=np.float32)
        elif arr.shape[0] <= 8 and arr.shape[1] > 64:
            return np.mean(arr, axis=0, dtype=np.float32)
        else:
            return np.mean(arr, axis=1, dtype=np.float32)

    # Higher dimensional tensors: collapse trailing channel dimensions
    return np.mean(arr, axis=tuple(range(1, arr.ndim)), dtype=np.float32)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 48000) -> np.ndarray:
    """
    Resample 1D audio array to target sample rate using polyphase filtering.

    Parameters:
        audio: 1D numpy array of audio samples.
        orig_sr: Source sampling rate in Hz.
        target_sr: Target sampling rate in Hz (default: 48,000 Hz).

    Returns:
        Resampled 1D float32 numpy array matching target sample rate and exact duration.
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Sample rates must be positive integers (got orig_sr={orig_sr}, target_sr={target_sr})")

    audio_mono = downmix_to_mono(audio)
    if len(audio_mono) == 0:
        return np.zeros(0, dtype=np.float32)

    if orig_sr == target_sr:
        return audio_mono.astype(np.float32)

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    target_len = int(round(len(audio_mono) * float(target_sr) / float(orig_sr)))
    resampled = signal.resample_poly(audio_mono, up, down).astype(np.float32)

    # Ensure returned length matches target duration exactly
    if len(resampled) > target_len:
        resampled = resampled[:target_len]
    elif len(resampled) < target_len:
        resampled = np.pad(resampled, (0, target_len - len(resampled)), mode="constant")

    return resampled


# ==============================================================================
# 3. SCIENTIFIC SPECTROGRAM GENERATION
# ==============================================================================

def generate_spectrogram(
    audio: np.ndarray,
    sr: int = 48000,
    n_fft: int = 2048,
    hop_length: int = 512,
    title: str = "Spectrogram",
    vmin_db: float = -80.0,
    vmax_db: float = 0.0,
) -> matplotlib.figure.Figure:
    """
    Generate a scientific STFT magnitude spectrogram in decibels (dB).

    Employs dark technical DAW rackmount styling with magma colormap,
    time (s), and frequency (kHz) axes.

    Parameters:
        audio: 1D numpy array of audio samples.
        sr: Sampling rate in Hz.
        n_fft: FFT window size (samples).
        hop_length: Hop length between successive STFT frames (samples).
        title: Plot title.
        vmin_db: Lower decibel clipping threshold.
        vmax_db: Upper decibel clipping threshold.

    Returns:
        matplotlib.figure.Figure instance.
    """
    audio_mono = downmix_to_mono(audio)

    # Handle sub-FFT tiny audio by zero-padding to at least n_fft
    if len(audio_mono) < n_fft:
        pad_width = n_fft - len(audio_mono)
        audio_padded = np.pad(audio_mono, (0, max(0, pad_width)), mode="constant")
    else:
        audio_padded = audio_mono

    # Compute Short-Time Fourier Transform
    f, t, Zxx = signal.stft(
        audio_padded,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window="hann",
        boundary="zeros",
    )

    magnitude = np.abs(Zxx)
    mag_db = 20.0 * np.log10(magnitude + 1e-9)
    mag_db = np.nan_to_num(mag_db, nan=vmin_db, posinf=vmax_db, neginf=vmin_db)
    mag_db = np.clip(mag_db, vmin_db, vmax_db)

    # Plot styling: DAW Rackmount Studio Aesthetic (Anti-AI-Slop)
    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
    fig.patch.set_facecolor("#18181b")  # Zinc 900
    ax.set_facecolor("#09090b")        # Zinc 950

    f_khz = f / 1000.0
    mesh = ax.pcolormesh(
        t,
        f_khz,
        mag_db,
        shading="auto",
        cmap="magma",
        vmin=vmin_db,
        vmax=vmax_db,
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("dB", color="#a1a1aa", fontsize=9)
    cbar.ax.tick_params(colors="#a1a1aa", labelsize=8)
    cbar.outline.set_edgecolor("#27272a")

    ax.set_title(title, color="#f4f4f5", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Time (s)", color="#a1a1aa", fontsize=9)
    ax.set_ylabel("Frequency (kHz)", color="#a1a1aa", fontsize=9)
    ax.tick_params(colors="#a1a1aa", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.15, color="#71717a")

    for spine in ax.spines.values():
        spine.set_edgecolor("#27272a")

    fig.tight_layout()
    return fig


# ==============================================================================
# 4. CHUNKING & OVERLAP-ADD (OLA) UTILITIES
# ==============================================================================

def chunk_audio(
    audio: np.ndarray,
    sr: int = 48000,
    max_chunk_seconds: float = 60.0,
    overlap_seconds: float = 0.5,
) -> List[np.ndarray]:
    """
    Split audio into overlapping chunks for memory-safe long-file processing.

    Parameters:
        audio: 1D numpy array of audio samples.
        sr: Sampling rate in Hz.
        max_chunk_seconds: Maximum segment length in seconds (default: 60.0s).
        overlap_seconds: Overlapping margin in seconds (default: 0.5s).

    Returns:
        List of 1D numpy array chunks.
    """
    audio_mono = downmix_to_mono(audio)
    if len(audio_mono) == 0:
        return [audio_mono]

    duration = len(audio_mono) / float(sr)
    if duration <= max_chunk_seconds:
        return [audio_mono]

    chunk_len = int(round(max_chunk_seconds * sr))
    overlap_len = int(round(overlap_seconds * sr))

    if chunk_len <= overlap_len:
        raise ValueError(
            f"max_chunk_seconds ({max_chunk_seconds}) must be strictly greater than overlap_seconds ({overlap_seconds})"
        )

    hop = chunk_len - overlap_len
    chunks: List[np.ndarray] = []
    i = 0
    while i < len(audio_mono):
        end_idx = min(i + chunk_len, len(audio_mono))
        chunk = audio_mono[i:end_idx]
        chunks.append(chunk)
        if end_idx >= len(audio_mono):
            break
        i += hop

    return chunks


def reconstruct_audio_ola(
    chunks: List[np.ndarray],
    sr: int = 48000,
    overlap_seconds: float = 0.5,
    total_length: Optional[int] = None,
) -> np.ndarray:
    """
    Reconstruct continuous audio from overlapping chunks using Hanning cross-fade.

    Ensures zero boundary clicks/pops and preserves total duration accurately.

    Parameters:
        chunks: List of overlapping 1D audio chunks.
        sr: Sampling rate in Hz.
        overlap_seconds: Overlap length in seconds.
        total_length: Optional expected output sample length to enforce exact alignment.

    Returns:
        Reconstructed 1D float32 numpy array.
    """
    if not chunks:
        return np.zeros(0, dtype=np.float32)

    if len(chunks) == 1:
        res = chunks[0].astype(np.float32)
        if total_length is not None:
            if len(res) > total_length:
                res = res[:total_length]
            elif len(res) < total_length:
                res = np.pad(res, (0, total_length - len(res)), mode="constant")
        return res

    overlap_len = int(round(overlap_seconds * sr))
    m = np.arange(overlap_len, dtype=np.float32)
    # Energy preserving raised-cosine / Hann cross-fade window (w_out + w_in == 1.0)
    w_out = 0.5 * (1.0 + np.cos(np.pi * m / overlap_len))
    w_in = 0.5 * (1.0 - np.cos(np.pi * m / overlap_len))

    out = chunks[0].astype(np.float32)
    for next_chunk in chunks[1:]:
        next_c = next_chunk.astype(np.float32)
        curr_overlap = min(len(out), len(next_c), overlap_len)
        if curr_overlap <= 0:
            out = np.concatenate([out, next_c])
        elif curr_overlap < overlap_len:
            m_c = np.arange(curr_overlap, dtype=np.float32)
            w_o = 0.5 * (1.0 + np.cos(np.pi * m_c / curr_overlap))
            w_i = 0.5 * (1.0 - np.cos(np.pi * m_c / curr_overlap))
            fade = out[-curr_overlap:] * w_o + next_c[:curr_overlap] * w_i
            out = np.concatenate([out[:-curr_overlap], fade, next_c[curr_overlap:]])
        else:
            fade = out[-overlap_len:] * w_out + next_c[:overlap_len] * w_in
            out = np.concatenate([out[:-overlap_len], fade, next_c[overlap_len:]])

    if total_length is not None:
        if len(out) > total_length:
            out = out[:total_length]
        elif len(out) < total_length:
            out = np.pad(out, (0, total_length - len(out)), mode="constant")

    return out.astype(np.float32)


# ==============================================================================
# 5. AUDIO PERSISTENCE
# ==============================================================================

def save_audio_wav(output_path: str, audio: np.ndarray, sample_rate: int = 48000) -> str:
    """
    Save 1D audio array to a standardized 48kHz Mono 16-bit PCM WAV file.

    Parameters:
        output_path: Target destination path.
        audio: Audio samples array.
        sample_rate: Sampling rate in Hz (default: 48,000 Hz).

    Returns:
        Absolute string path to the saved WAV file.
    """
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio_mono = downmix_to_mono(audio)
    audio_mono = np.nan_to_num(audio_mono, nan=0.0, posinf=1.0, neginf=-1.0)
    audio_mono = np.clip(audio_mono, -1.0, 1.0)

    sf.write(str(out_path), audio_mono, sample_rate, subtype="PCM_16")
    return str(out_path)


# ==============================================================================
# 6. INTERFACE CONTRACT CT-01: INSPECT AND PREPARE AUDIO
# ==============================================================================

def inspect_and_prepare_audio(
    input_path: str,
    spec_title: str = "Input Spectrogram (Pre-Denoising)",
) -> Dict[str, Any]:
    """
    Contract CT-01: Ingest, validate, and standardize arbitrary input audio.

    Transforms uploaded files (WAV, MP3, FLAC, M4A, etc.) into a normalized
    48,000 Hz Mono float32 array, saves a cached WAV copy, and generates
    an initial STFT spectrogram figure.

    Parameters:
        input_path: Path to the input audio file.

    Returns:
        Dictionary conforming to CT-01 specification:
        {
            "path": str,               # Path to normalized 48kHz mono WAV file
            "sample_rate": 48000,      # Standardized sample rate
            "channels": 1,             # Standardized channels
            "duration_seconds": float, # Exact audio duration in seconds
            "frames": int,             # Total frame count
            "audio_array_mono": np.ndarray,  # 1D float32 audio samples
            "spec_fig": matplotlib.figure.Figure, # Matplotlib spectrogram
        }

    Raises:
        AudioValidationError: If the file is missing, empty (0 bytes), corrupt,
                              or undecodable.
    """
    path_obj = Path(input_path)

    # 1. Existence Check
    if not path_obj.exists() or not path_obj.is_file():
        raise AudioValidationError(f"File not found: {input_path}")

    # 2. File Size Check (0-byte rejection)
    try:
        size_bytes = path_obj.stat().st_size
    except OSError as err:
        raise AudioValidationError(f"Cannot access audio file metadata '{input_path}': {err}") from err

    if size_bytes == 0:
        raise AudioValidationError(f"Audio file is empty (0 bytes): {input_path}")

    # 3. File Decoding via SoundFile
    try:
        raw_audio, sr = sf.read(str(path_obj), dtype="float32")
    except Exception as exc:
        raise AudioValidationError(f"Corrupt or unsupported audio file '{input_path}': {exc}") from exc

    if raw_audio is None or len(raw_audio) == 0:
        raise AudioValidationError(f"Audio file contains 0 frames: {input_path}")

    # 4. Multi-channel Downmix to Mono
    audio_mono = downmix_to_mono(raw_audio)

    # 5. Polyphase Resampling to 48,000 Hz
    if sr != 48000:
        audio_mono = resample_audio(audio_mono, orig_sr=sr, target_sr=48000)
    else:
        audio_mono = audio_mono.astype(np.float32)

    # 6. NaN / Inf Sanitization & Safe Amplitude Normalization
    audio_mono = np.nan_to_num(audio_mono, nan=0.0, posinf=1.0, neginf=-1.0)
    max_abs = float(np.max(np.abs(audio_mono))) if len(audio_mono) > 0 else 0.0
    if max_abs > 1.0:
        audio_mono = audio_mono / max_abs

    # 7. Write Standardized 48kHz Mono WAV to Cache Directory
    cache_dir = Path(tempfile.gettempdir()) / "voice_denoise_prep_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    norm_path = cache_dir / f"prep_{uuid.uuid4().hex[:12]}.wav"
    save_audio_wav(str(norm_path), audio_mono, sample_rate=48000)

    # 8. Compute Exact Duration & Metadata
    frames = len(audio_mono)
    duration_seconds = float(frames) / 48000.0

    # 9. Generate Scientific STFT Spectrogram
    spec_fig = generate_spectrogram(audio_mono, sr=48000, title=spec_title)

    return {
        "path": str(norm_path),
        "sample_rate": 48000,
        "channels": 1,
        "duration_seconds": duration_seconds,
        "frames": frames,
        "audio_array_mono": audio_mono,
        "spec_fig": spec_fig,
    }


# ==============================================================================
# 7. INTERFACE CONTRACT CT-03: TIMING & RTF TELEMETRY
# ==============================================================================

def summarize_timing(elapsed_seconds: float, audio_duration_seconds: float) -> Dict[str, float]:
    """
    Contract CT-03: Compute performance telemetry and Real-Time Factor (RTF).

    Validates inputs and calculates:
        RTF = elapsed_seconds / audio_duration_seconds

    An RTF < 1.0 indicates faster-than-realtime execution.
    If audio_duration_seconds is 0.0, RTF defaults safely to 0.0.

    Parameters:
        elapsed_seconds: Processing elapsed wall-clock time in seconds.
        audio_duration_seconds: Duration of the processed audio in seconds.

    Returns:
        Dictionary conforming to CT-03 specification:
        {
            "elapsed_seconds": float,         # Rounded to 3 decimal places
            "audio_duration_seconds": float,  # Rounded to 3 decimal places
            "rtf": float,                     # Rounded to 3 decimal places
        }

    Raises:
        ValueError: If elapsed_seconds or audio_duration_seconds is negative.
    """
    if elapsed_seconds < 0.0 or audio_duration_seconds < 0.0:
        raise ValueError(
            f"Elapsed time and audio duration must be non-negative (got elapsed={elapsed_seconds}, duration={audio_duration_seconds})"
        )

    if audio_duration_seconds == 0.0:
        rtf = 0.0
    else:
        rtf = elapsed_seconds / audio_duration_seconds

    return {
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "audio_duration_seconds": round(float(audio_duration_seconds), 3),
        "rtf": round(float(rtf), 3),
    }
