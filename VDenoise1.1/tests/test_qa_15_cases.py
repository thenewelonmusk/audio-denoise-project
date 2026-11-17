"""
tests/test_qa_15_cases.py - Comprehensive 15-Case QA Verification Suite.

Covers all functional, boundary, performance, and exception scenarios required
in Section 6.3 & Section 9.1 of the project technical specification.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import gradio as gr
import imageio_ffmpeg
import numpy as np
import pytest
import soundfile as sf

from app import process_audio, process_stream_chunk, reset_stream_session
from src.audio_io import AudioValidationError, inspect_and_prepare_audio
from src.pipeline import run_pipeline

import matplotlib.pyplot as plt

@pytest.fixture(autouse=True)
def _cleanup_matplotlib_figures():
    yield
    plt.close("all")


def _make_voice_signal(duration_sec: float, sr: int = 48000, noise_std: float = 0.03) -> np.ndarray:
    """Generate deterministic speech-like harmonic waveform mixed with Gaussian noise."""
    t = np.linspace(0, duration_sec, int(round(duration_sec * sr)), endpoint=False, dtype=np.float32)
    voice = 0.25 * np.sin(2.0 * np.pi * 220.0 * t) + 0.12 * np.sin(2.0 * np.pi * 440.0 * t)
    rng = np.random.default_rng(seed=123)
    noise = noise_std * rng.standard_normal(size=t.shape).astype(np.float32)
    return np.clip(voice + noise, -0.95, 0.95).astype(np.float32)


def test_tc01_short_wav_under_60s(tmp_path: Path) -> None:
    """TC-01: Short WAV file (< 60s) processes in 1 single chunk."""
    in_file = tmp_path / "tc01_15s.wav"
    sf.write(str(in_file), _make_voice_signal(15.0), 48000, subtype="PCM_16")

    res = run_pipeline(str(in_file), engine_name="mock", output_dir=tmp_path / "out")
    assert res["num_chunks"] == 1
    assert Path(res["output_path"]).exists()


def test_tc02_mp3_format_transcoding(tmp_path: Path) -> None:
    """TC-02: Compressed MP3 file auto-transcodes to 48kHz Mono WAV."""
    src_wav = tmp_path / "src.wav"
    in_mp3 = tmp_path / "tc02.mp3"
    sf.write(str(src_wav), _make_voice_signal(5.0), 48000, subtype="PCM_16")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-i", str(src_wav), "-b:a", "128k", str(in_mp3)], check=True)

    res = run_pipeline(str(in_mp3), engine_name="mock", output_dir=tmp_path / "out")
    assert res["sample_rate"] == 48000
    assert res["channels"] == 1
    assert Path(res["output_path"]).suffix == ".wav"


def test_tc03_m4a_phone_recording_transcoding(tmp_path: Path) -> None:
    """TC-03: Mobile M4A (AAC) recording auto-decodes and denoises cleanly."""
    src_wav = tmp_path / "src.wav"
    in_m4a = tmp_path / "tc03.m4a"
    sf.write(str(src_wav), _make_voice_signal(5.0), 48000, subtype="PCM_16")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-i", str(src_wav), "-c:a", "aac", str(in_m4a)], check=True)

    res = run_pipeline(str(in_m4a), engine_name="mock", output_dir=tmp_path / "out")
    assert Path(res["output_path"]).exists()
    assert res["audio_duration_seconds"] == pytest.approx(5.0, abs=0.1)


def test_tc04_flac_lossless_format(tmp_path: Path) -> None:
    """TC-04: Lossless FLAC file is ingested and exported as standard WAV."""
    in_flac = tmp_path / "tc04.flac"
    sf.write(str(in_flac), _make_voice_signal(4.0), 48000)

    res = run_pipeline(str(in_flac), engine_name="mock", output_dir=tmp_path / "out")
    assert Path(res["output_path"]).exists()
    assert res["sample_rate"] == 48000


def test_tc05_stereo_to_mono_downmix(tmp_path: Path) -> None:
    """TC-05: 2-channel Stereo input is automatically downmixed to 1-channel Mono."""
    mono = _make_voice_signal(4.0)
    stereo = np.stack([mono, mono * 0.5], axis=1)
    in_stereo = tmp_path / "tc05_stereo.wav"
    sf.write(str(in_stereo), stereo, 48000, subtype="PCM_16")

    prep = inspect_and_prepare_audio(str(in_stereo))
    assert prep["channels"] == 1
    assert prep["audio_array_mono"].ndim == 1
    Path(prep["path"]).unlink(missing_ok=True)


def test_tc06_low_sample_rate_upsampling(tmp_path: Path) -> None:
    """TC-06: Narrowband 16kHz input is polyphase-resampled to 48kHz."""
    in_16k = tmp_path / "tc06_16k.wav"
    sf.write(str(in_16k), _make_voice_signal(4.0, sr=16000), 16000, subtype="PCM_16")

    res = run_pipeline(str(in_16k), engine_name="mock", output_dir=tmp_path / "out")
    out_data, out_sr = sf.read(res["output_path"], dtype="float32")
    assert out_sr == 48000
    assert len(out_data) == 48000 * 4


def test_tc07_medium_audio_two_chunks(tmp_path: Path) -> None:
    """TC-07: Audio > 60s (90s) splits into 2 chunks and merges into 1 output file."""
    in_90s = tmp_path / "tc07_90s.wav"
    sf.write(str(in_90s), _make_voice_signal(90.0), 48000, subtype="PCM_16")

    res = run_pipeline(str(in_90s), engine_name="mock", output_dir=tmp_path / "out")
    assert res["num_chunks"] == 2
    assert res["peak_ram_mb"] < 1228.8


def test_tc08_long_audio_three_chunks_memory_safe(tmp_path: Path) -> None:
    """TC-08: Long 150s (2.5 min) audio splits into 3 chunks with Peak RAM < 1.2GB."""
    in_150s = tmp_path / "tc08_150s.wav"
    sf.write(str(in_150s), _make_voice_signal(150.0), 48000, subtype="PCM_16")

    res = run_pipeline(str(in_150s), engine_name="mock", output_dir=tmp_path / "out")
    assert res["num_chunks"] == 3
    assert res["peak_ram_mb"] < 1228.8
    out_data, _ = sf.read(res["output_path"], dtype="float32")
    assert len(out_data) == 48000 * 150


def test_tc09_ola_boundary_no_click_artifact(tmp_path: Path) -> None:
    """TC-09: Overlap-Add splice at 60s boundary produces no click/pop step discontinuity."""
    in_70s = tmp_path / "tc09_70s.wav"
    sf.write(str(in_70s), _make_voice_signal(70.0, noise_std=0.0), 48000, subtype="PCM_16")

    res = run_pipeline(str(in_70s), engine_name="mock", output_dir=tmp_path / "out")
    out_data, _ = sf.read(res["output_path"], dtype="float32")
    # Inspect sample-to-sample jumps across the 59s - 61s splice region
    splice_region = out_data[59 * 48000 : 61 * 48000]
    max_step = float(np.max(np.abs(np.diff(splice_region))))
    assert max_step < 0.15


def test_tc10_heavy_noise_stability(tmp_path: Path) -> None:
    """TC-10: Heavy noise input (-5 dB equivalent) yields finite, non-clipping output."""
    in_noisy = tmp_path / "tc10_heavy_noise.wav"
    sf.write(str(in_noisy), _make_voice_signal(5.0, noise_std=0.35), 48000, subtype="PCM_16")

    res = run_pipeline(str(in_noisy), engine_name="mock", output_dir=tmp_path / "out")
    out_data, _ = sf.read(res["output_path"], dtype="float32")
    assert np.all(np.isfinite(out_data))
    assert float(np.max(np.abs(out_data))) <= 1.0


def test_tc11_empty_zero_byte_file_rejected(tmp_path: Path) -> None:
    """TC-11: Empty 0-byte file raises AudioValidationError cleanly."""
    empty_wav = tmp_path / "tc11_empty.wav"
    empty_wav.write_bytes(b"")

    with pytest.raises(AudioValidationError):
        run_pipeline(str(empty_wav), engine_name="mock", output_dir=tmp_path / "out")


def test_tc12_corrupt_fake_mp3_rejected(tmp_path: Path) -> None:
    """TC-12: Text file renamed to .mp3 raises AudioValidationError cleanly."""
    corrupt_mp3 = tmp_path / "tc12_fake.mp3"
    corrupt_mp3.write_text("NOT_A_VALID_AUDIO_STREAM", encoding="utf-8")

    with pytest.raises(AudioValidationError):
        run_pipeline(str(corrupt_mp3), engine_name="mock", output_dir=tmp_path / "out")


def test_tc13_missing_file_rejected(tmp_path: Path) -> None:
    """TC-13: Non-existent file path raises AudioValidationError."""
    with pytest.raises(AudioValidationError):
        run_pipeline(str(tmp_path / "non_existent.wav"), engine_name="mock")


def test_tc14_ui_contract_ct05_and_none_guard(tmp_path: Path) -> None:
    """TC-14: Gradio process_audio raises gr.Error on None and returns 4-tuple on valid input."""
    with pytest.raises(gr.Error):
        process_audio(None, engine_name="mock")

    in_wav = tmp_path / "tc14.wav"
    sf.write(str(in_wav), _make_voice_signal(3.0), 48000, subtype="PCM_16")
    out_path, stats, spec_before, spec_after = process_audio(str(in_wav), engine_name="mock")
    assert out_path is not None and Path(out_path).exists()
    assert "rtf" in stats
    assert spec_before is not None and spec_after is not None


def test_tc15_live_streaming_chunk_and_reset() -> None:
    """TC-15: Real-time 200ms streaming chunk inference and session state reset work cleanly."""
    state, _ = reset_stream_session(engine_name="mock")
    chunk = _make_voice_signal(0.2, sr=48000)

    out_tuple, next_state = process_stream_chunk((48000, chunk), state=state, engine_name="mock")
    assert out_tuple is not None
    out_sr, out_chunk = out_tuple
    assert out_sr == 48000
    assert len(out_chunk) == len(chunk)
    assert next_state.chunk_count == 1