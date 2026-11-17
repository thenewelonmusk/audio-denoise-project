"""
src/pipeline.py - End-to-End Audio Denoising Orchestration Pipeline (Contract CT-04).

Coordinates:
1. Multi-format audio ingestion & transcoding (WAV, MP3, M4A, AAC, FLAC, OGG).
2. Standardization to 48 kHz Mono float32 via CT-01 (inspect_and_prepare_audio).
3. Memory-safe chunking (<= 60s) and raised-cosine Overlap-Add (OLA) reconstruction.
4. Pluggable AI inference via EngineRegistry (CT-02).
5. Wall-clock RTF and Peak RSS RAM telemetry via BenchmarkSession (CT-03).
6. Automatic cleanup of intermediate temporary files.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import imageio_ffmpeg
import soundfile as sf

from src.audio_io import (
    AudioValidationError,
    chunk_audio,
    generate_spectrogram,
    inspect_and_prepare_audio,
    reconstruct_audio_ola,
    save_audio_wav,
)
from src.benchmark import BenchmarkSession, append_benchmark_csv
from src.config import (
    CHUNK_MAX_SECONDS,
    DEFAULT_ENGINE_ID,
    DEFAULT_OUTPUT_DIR,
    OVERLAP_SECONDS,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
)
from src.engine_interface import EngineRegistry

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _resolve_ffmpeg_executable() -> str:
    """Locate ffmpeg executable from system PATH or bundled imageio_ffmpeg."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise AudioValidationError(
            f"Không tìm thấy công cụ ffmpeg để giải mã định dạng âm thanh nén: {exc}"
        ) from exc


def transcode_to_wav_if_needed(input_path: str) -> tuple[str, Optional[str]]:
    """
    Convert compressed audio formats (MP3, M4A, AAC, FLAC, OGG) to a temporary
    48 kHz Mono PCM_16 WAV file using ffmpeg.

    Returns:
        (effective_wav_path, temp_converted_path_to_cleanup)
    """
    path_obj = Path(input_path)
    if not path_obj.exists() or not path_obj.is_file():
        raise AudioValidationError(f"File not found: {input_path}")

    if path_obj.stat().st_size == 0:
        raise AudioValidationError(f"Audio file is empty (0 bytes): {input_path}")

    if path_obj.suffix.lower() == ".wav":
        return str(path_obj), None

    ffmpeg_exe = _resolve_ffmpeg_executable()
    conv_dir = Path(tempfile.gettempdir()) / "voice_denoise_ffmpeg_cache"
    conv_dir.mkdir(parents=True, exist_ok=True)
    temp_wav = conv_dir / f"ffmpeg_conv_{uuid.uuid4().hex[:12]}.wav"

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(path_obj),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        str(TARGET_CHANNELS),
        str(temp_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not temp_wav.exists() or temp_wav.stat().st_size == 0:
        temp_wav.unlink(missing_ok=True)
        stderr_tail = (result.stderr or "").strip().splitlines()[-1:]
        detail = stderr_tail[0] if stderr_tail else "ffmpeg transcoding failed"
        raise AudioValidationError(f"Không thể giải mã tệp '{input_path}': {detail}")

    return str(temp_wav), str(temp_wav)


def run_pipeline(
    input_path: str,
    engine_name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    log_benchmark: bool = False,
) -> Dict[str, Any]:
    """
    Contract CT-04: Execute the full speech denoising pipeline on an audio file
    of arbitrary format and arbitrary duration.

    Parameters:
        input_path: Path to input audio file (WAV, MP3, M4A, FLAC, OGG,...).
        engine_name: Registered engine ID (defaults to config.DEFAULT_ENGINE_ID).
        output_dir: Destination folder for the single output WAV file (defaults to outputs/).
        progress_callback: Optional callback fn(fraction: float, message: str).
        log_benchmark: If True, append performance metrics to results/benchmark.csv.

    Returns:
        Dictionary conforming to Contract CT-04:
        {
            "output_path": str,
            "elapsed_seconds": float,
            "audio_duration_seconds": float,
            "rtf": float,
            "peak_ram_mb": float,
            "ram_delta_mb": float,
            "sample_rate": int,
            "channels": int,
            "num_chunks": int,
            "engine_name": str,
            "spec_before": matplotlib.figure.Figure,
            "spec_after": matplotlib.figure.Figure,
        }
    """

    def _report(frac: float, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(frac, msg)

    converted_wav_path: Optional[str] = None
    prep_wav_path: Optional[str] = None

    try:
        with BenchmarkSession() as bench:
            # 1. Transcode non-WAV inputs
            _report(0.03, "Đang kiểm tra và giải mã định dạng tệp âm thanh...")
            effective_path, converted_wav_path = transcode_to_wav_if_needed(input_path)

            # 2. Validate, normalize to 48kHz Mono, and render input spectrogram (CT-01)
            _report(0.08, "Đang chuẩn hóa 48 kHz Mono và phân tích phổ đầu vào...")
            prep_info = inspect_and_prepare_audio(effective_path, spec_title="")
            prep_wav_path = prep_info["path"]
            audio_mono = prep_info["audio_array_mono"]
            duration = float(prep_info["duration_seconds"])
            sr = int(prep_info["sample_rate"])
            spec_before = prep_info["spec_fig"]

            # 3. Resolve AI Engine (CT-02)
            resolved_engine_id = engine_name.strip() if engine_name else DEFAULT_ENGINE_ID
            engine = EngineRegistry.get_engine(resolved_engine_id)

            # Prepare unique single output file path
            target_out_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
            target_out_dir.mkdir(parents=True, exist_ok=True)
            final_output_path = target_out_dir / f"clean_{uuid.uuid4().hex[:12]}.wav"

            # 4. Inference: Direct file pass (<= 60s) vs Chunking + Overlap-Add (> 60s)
            if duration > CHUNK_MAX_SECONDS:
                _report(0.20, f"Tệp dài ({duration:.1f}s > 60s): Đang phân đoạn tối đa 60s/khối...")
                chunks = chunk_audio(
                    audio_mono,
                    sr=sr,
                    max_chunk_seconds=CHUNK_MAX_SECONDS,
                    overlap_seconds=OVERLAP_SECONDS,
                )
                num_chunks = len(chunks)
                clean_chunks = []

                for idx, chunk in enumerate(chunks):
                    frac = 0.20 + 0.65 * ((idx + 1) / num_chunks)
                    _report(frac, f"Đang khử nhiễu phân đoạn {idx + 1}/{num_chunks} ({engine.name})...")
                    clean_chunks.append(engine.process_array(chunk, sample_rate=sr))

                _report(0.88, "Đang ghép nối mượt mà Overlap-Add (Hanning Cross-fade)...")
                cleaned_audio = reconstruct_audio_ola(
                    clean_chunks,
                    sr=sr,
                    overlap_seconds=OVERLAP_SECONDS,
                    total_length=len(audio_mono),
                )
                saved_path = save_audio_wav(str(final_output_path), cleaned_audio, sample_rate=sr)
            else:
                num_chunks = 1
                _report(0.35, f"Đang khử nhiễu trực tiếp bằng mô hình {engine.name}...")
                engine.process_file(prep_wav_path, str(final_output_path))
                saved_path = str(final_output_path.resolve())
                cleaned_audio, _ = sf.read(saved_path, dtype="float32")

            # 5. Render Output Spectrogram
            _report(0.94, "Đang kết xuất biểu đồ phổ Spectrogram sau khử nhiễu...")
            spec_after = generate_spectrogram(cleaned_audio, sr=sr, title="")

        # 6. Collect RTF & Peak RAM Telemetry (CT-03)
        perf = bench.build_report(audio_duration_seconds=duration)

        result: Dict[str, Any] = {
            "output_path": saved_path,
            "elapsed_seconds": perf["elapsed_seconds"],
            "audio_duration_seconds": perf["audio_duration_seconds"],
            "rtf": perf["rtf"],
            "peak_ram_mb": perf["peak_ram_mb"],
            "ram_delta_mb": perf["ram_delta_mb"],
            "sample_rate": sr,
            "channels": TARGET_CHANNELS,
            "num_chunks": num_chunks,
            "engine_name": engine.name,
            "spec_before": spec_before,
            "spec_after": spec_after,
        }

        if log_benchmark:
            append_benchmark_csv(
                {
                    "input_file": Path(input_path).name,
                    "engine_name": engine.name,
                    "audio_duration_seconds": perf["audio_duration_seconds"],
                    "num_chunks": num_chunks,
                    "elapsed_seconds": perf["elapsed_seconds"],
                    "rtf": perf["rtf"],
                    "peak_ram_mb": perf["peak_ram_mb"],
                    "ram_delta_mb": perf["ram_delta_mb"],
                    "within_rtf_target": perf["within_rtf_target"],
                    "within_ram_target": perf["within_ram_target"],
                }
            )

        _report(1.0, "Hoàn tất quy trình khử nhiễu!")
        return result

    finally:
        # Always remove temporary transcoded/prepared audio files
        if converted_wav_path:
            try:
                Path(converted_wav_path).unlink(missing_ok=True)
            except OSError:
                pass
        if prep_wav_path:
            try:
                Path(prep_wav_path).unlink(missing_ok=True)
            except OSError:
                pass