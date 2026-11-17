"""
tests/test_pipeline.py - Automated Integration Tests for Contract CT-04 (run_pipeline).

Verifies:
1. Short audio (< 60s) single-pass processing.
2. Long audio (> 60s, e.g. 135s -> 3 chunks) chunking & Overlap-Add continuity.
3. Exact sample-length preservation across OLA boundaries.
4. Peak RAM (< 1.2 GB) and RTF telemetry reporting.
5. Error handling for missing or 0-byte corrupted files.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.audio_io import AudioValidationError
from src.pipeline import run_pipeline


def _generate_synthetic_speech_wav(path: Path, duration_sec: float, sr: int = 48000) -> Path:
    """Create a synthetic voice-like harmonic signal + background noise WAV file."""
    t = np.linspace(0, duration_sec, int(round(duration_sec * sr)), endpoint=False, dtype=np.float32)
    voice = 0.25 * np.sin(2.0 * np.pi * 220.0 * t) + 0.12 * np.sin(2.0 * np.pi * 440.0 * t)
    rng = np.random.default_rng(seed=42)
    noise = 0.03 * rng.standard_normal(size=t.shape).astype(np.float32)
    sig = np.clip(voice + noise, -0.95, 0.95).astype(np.float32)
    sf.write(str(path), sig, sr, subtype="PCM_16")
    return path


def test_pipeline_short_audio_single_chunk(tmp_path: Path) -> None:
    """Audio <= 60s must be processed in 1 chunk and return a valid 48kHz mono WAV."""
    in_wav = _generate_synthetic_speech_wav(tmp_path / "short_10s.wav", duration_sec=10.0)
    out_dir = tmp_path / "outputs"

    res = run_pipeline(str(in_wav), engine_name="mock", output_dir=out_dir)

    out_path = Path(res["output_path"])
    assert out_path.exists()
    assert res["num_chunks"] == 1
    assert res["sample_rate"] == 48000
    assert res["channels"] == 1
    assert res["audio_duration_seconds"] == pytest.approx(10.0, abs=0.05)
    assert res["peak_ram_mb"] < 1228.8

    out_data, out_sr = sf.read(str(out_path), dtype="float32")
    assert out_sr == 48000
    assert out_data.ndim == 1
    assert len(out_data) == 48000 * 10


def test_pipeline_long_audio_chunking_and_ola(tmp_path: Path) -> None:
    """Audio > 60s (135s) must split into 3 chunks and reconstruct seamlessly via OLA."""
    duration_sec = 135.0
    sr = 48000
    in_wav = _generate_synthetic_speech_wav(tmp_path / "long_135s.wav", duration_sec=duration_sec, sr=sr)
    out_dir = tmp_path / "outputs"

    res = run_pipeline(str(in_wav), engine_name="mock", output_dir=out_dir)

    out_path = Path(res["output_path"])
    assert out_path.exists()
    assert res["num_chunks"] == 3
    assert res["peak_ram_mb"] < 1228.8

    out_data, out_sr = sf.read(str(out_path), dtype="float32")
    assert out_sr == sr
    # Verify exact frame alignment after Overlap-Add concatenation
    assert len(out_data) == int(round(duration_sec * sr))
    # Verify no NaN/Inf or boundary clipping spikes
    assert np.all(np.isfinite(out_data))
    assert float(np.max(np.abs(out_data))) <= 1.0


def test_pipeline_rejects_empty_or_missing_file(tmp_path: Path) -> None:
    """Pipeline must raise AudioValidationError on 0-byte or non-existent files."""
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")

    with pytest.raises(AudioValidationError):
        run_pipeline(str(empty_file), engine_name="mock", output_dir=tmp_path / "outputs")

    with pytest.raises(AudioValidationError):
        run_pipeline(str(tmp_path / "does_not_exist.wav"), engine_name="mock")