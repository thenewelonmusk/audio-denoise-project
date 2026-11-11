"""
app.py - Acoustic Studio: Professional AI Audio Denoiser Workstation.

Layer 3: Presentation & User Experience Layer.
Provides a modern, high-contrast Digital Audio Workstation (DAW) interface
built with Gradio 6.31.0 and adhering strictly to Anti-AI-Slop design guidelines.

Strict Architectural Boundary:
Zero direct imports of PyTorch, torchaudio, ONNX, or deep learning inference libraries.
All audio processing is orchestrated via:
- src.audio_io: audio ingestion, resampling, downmixing, chunking, OLA, spectrograms
- src.engine_interface: abstract pluggable engine contracts & factory registry
- src.theme: custom DAW studio theme and CSS tokens
"""

from collections.abc import Mapping
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import gradio as gr
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from src.audio_io import (
    AudioValidationError,
    chunk_audio,
    downmix_to_mono,
    generate_spectrogram,
    inspect_and_prepare_audio,
    reconstruct_audio_ola,
    save_audio_wav,
    summarize_timing,
)
from src.engine_interface import (
    AudioDenoiseEngine,
    EngineError,
    EngineRegistry,
    get_engine,
)
from src.theme import (
    AudioWorkstationTheme,
    DAW_CUSTOM_CSS,
    get_daw_theme,
)

logger = logging.getLogger(__name__)

OUTPUT_CACHE_DIR_NAME = "voice_denoise_app_outputs"
FFMPEG_CACHE_DIR_NAME = "voice_denoise_ffmpeg_cache"
PREP_CACHE_DIR_NAME = "voice_denoise_prep_cache"
OUTPUT_FILE_GLOB = "clean_*.wav"
FFMPEG_FILE_GLOB = "ffmpeg_conv_*.wav"
PREP_FILE_GLOB = "prep_*.wav"


def get_output_cache_dir() -> Path:
    """Return the temporary directory used to store denoised output WAV files."""
    return Path(tempfile.gettempdir()) / OUTPUT_CACHE_DIR_NAME


def _unlink_matching_files(
    target_dir: Path,
    pattern: str,
    *,
    recursive: bool = False,
    prune_empty_subdirs: bool = False,
) -> int:
    """Delete regular files matching pattern inside target_dir without failing on OS errors."""
    if not target_dir.exists() or not target_dir.is_dir():
        return 0

    deleted_count = 0
    candidate_parents = set()
    try:
        iterator = target_dir.rglob(pattern) if recursive else target_dir.glob(pattern)
        for file_path in iterator:
            if not file_path.is_file():
                continue
            try:
                file_path.unlink(missing_ok=True)
                deleted_count += 1
                if prune_empty_subdirs and file_path.parent != target_dir:
                    candidate_parents.add(file_path.parent)
            except OSError as err:
                logger.warning("Không thể xóa tệp tạm cũ '%s': %s", file_path, err)
    except OSError as err:
        logger.warning("Không thể quét thư mục tạm '%s': %s", target_dir, err)

    if prune_empty_subdirs:
        for parent_dir in sorted(candidate_parents, key=lambda p: len(p.parts), reverse=True):
            try:
                if parent_dir.exists() and parent_dir.is_dir() and not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
            except OSError:
                pass

    return deleted_count


def cleanup_stale_temp_files(temp_root: Optional[Path] = None) -> Dict[str, int]:
    """
    Scan and remove leftover temporary audio artifacts from prior sessions.

    Sweeps:
    - clean_*.wav in temp_root / voice_denoise_app_outputs
    - clean_*.wav directly in temp_root (non-recursive)
    - clean_*.wav in Gradio's temp cache (GRADIO_TEMP_DIR or temp_root / gradio)
    - leftover ffmpeg_conv_*.wav and prep_*.wav from interrupted runs

    Returns:
        Dictionary with per-target deletion counts and 'total'.
    """
    root_dir = temp_root if temp_root is not None else Path(tempfile.gettempdir())
    gradio_env = os.environ.get("GRADIO_TEMP_DIR", "").strip()
    gradio_dir = Path(gradio_env) if (temp_root is None and gradio_env) else (root_dir / "gradio")

    scan_specs: List[Tuple[str, Path, str, bool, bool]] = [
        ("outputs", root_dir / OUTPUT_CACHE_DIR_NAME, OUTPUT_FILE_GLOB, False, False),
        ("temp_root", root_dir, OUTPUT_FILE_GLOB, False, False),
        ("ffmpeg_cache", root_dir / FFMPEG_CACHE_DIR_NAME, FFMPEG_FILE_GLOB, False, False),
        ("prep_cache", root_dir / PREP_CACHE_DIR_NAME, PREP_FILE_GLOB, False, False),
    ]
    if gradio_dir.resolve() != root_dir.resolve():
        scan_specs.insert(2, ("gradio_cache", gradio_dir, OUTPUT_FILE_GLOB, True, True))

    counts: Dict[str, int] = {"gradio_cache": 0}
    total = 0
    for key, target_dir, pattern, recursive, prune_subdirs in scan_specs:
        removed = _unlink_matching_files(
            target_dir,
            pattern,
            recursive=recursive,
            prune_empty_subdirs=prune_subdirs,
        )
        counts[key] = removed
        total += removed

    counts["total"] = total
    if total > 0:
        logger.info("Đã dọn %d tệp tạm từ các phiên trước (%s).", total, counts)
    return counts


# ==============================================================================
# 1. TELEMETRY CONTAINER & HTML FORMATTERS (DAW METERS & CARDS)
# ==============================================================================

class TelemetryStats(str, Mapping):
    """
    Hybrid string and dictionary container for Gradio telemetry.

    Subclasses str so Gradio gr.HTML renders the HTML markup seamlessly,
    while providing dictionary key access (Contract CT-05) for programmatic
    evaluation harnesses and test assertions.
    """

    def __new__(cls, html_str: str, data: Optional[Dict[str, Any]] = None):
        inst = super().__new__(cls, html_str)
        inst._data = dict(data) if data is not None else {}
        return inst

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item in self._data:
                return self._data[item]
            raise KeyError(item)
        return super().__getitem__(item)

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, item: Any) -> bool:
        if item in self._data:
            return True
        if isinstance(item, str):
            return super().__contains__(item)
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get metric by key with default value."""
        return self._data.get(key, default)

    def keys(self):
        """Return view of metrics keys."""
        return self._data.keys()

    def values(self):
        """Return view of metrics values."""
        return self._data.values()

    def items(self):
        """Return view of metrics (key, value) pairs."""
        return self._data.items()

    def to_dict(self) -> Dict[str, Any]:
        """Return copy of underlying metrics dictionary."""
        return dict(self._data)

    def copy(self) -> Dict[str, Any]:
        """Return copy of underlying metrics dictionary."""
        return dict(self._data)


def format_telemetry_html(
    timing: dict[str, float],
    sample_rate: int = 48000,
    engine_name: str = "mock",
) -> str:
    """Format processing metrics into hardware rackmount telemetry cards."""
    elapsed = timing.get("elapsed_seconds", 0.0)
    duration = timing.get("audio_duration_seconds", 0.0)
    rtf = timing.get("rtf", 0.0)

    if rtf < 0.3:
        rtf_class = "status-good"
        status_label = "SIÊU NHANH"
    elif rtf < 1.0:
        rtf_class = "status-good"
        status_label = "REALTIME"
    else:
        rtf_class = "status-warn"
        status_label = "CHẬM HƠN REALTIME"

    meter_pct = min(100, max(5, int(min(rtf, 2.0) / 2.0 * 100)))

    return f"""
<div class="telemetry-grid">
    <div class="telemetry-card">
        <span class="telemetry-label">Thời gian Xử lý</span>
        <span class="telemetry-value">{elapsed:.3f}<span class="telemetry-unit">s</span></span>
    </div>
    <div class="telemetry-card">
        <span class="telemetry-label">Thời lượng Âm thanh</span>
        <span class="telemetry-value">{duration:.2f}<span class="telemetry-unit">s</span></span>
    </div>
    <div class="telemetry-card">
        <span class="telemetry-label">Hệ số Thời gian thực (RTF)</span>
        <span class="telemetry-value {rtf_class}">{rtf:.3f}<span class="telemetry-unit">({status_label})</span></span>
        <div class="audio-meter-bar"><div class="audio-meter-fill" style="width: {meter_pct}%;"></div></div>
    </div>
    <div class="telemetry-card">
        <span class="telemetry-label">Tần số Lấy mẫu & Động cơ</span>
        <span class="telemetry-value">{sample_rate // 1000}<span class="telemetry-unit">kHz / {engine_name.upper()}</span></span>
    </div>
</div>
"""


def get_initial_telemetry_html() -> str:
    """Return default hardware cards for initial UI state."""
    return """
<div class="telemetry-grid">
    <div class="telemetry-card">
        <span class="telemetry-label">Thời gian Xử lý</span>
        <span class="telemetry-value">--<span class="telemetry-unit">s</span></span>
    </div>
    <div class="telemetry-card">
        <span class="telemetry-label">Thời lượng Âm thanh</span>
        <span class="telemetry-value">--<span class="telemetry-unit">s</span></span>
    </div>
    <div class="telemetry-card">
        <span class="telemetry-label">Hệ số Thời gian thực (RTF)</span>
        <span class="telemetry-value">--<span class="telemetry-unit">(--)</span></span>
        <div class="audio-meter-bar"><div class="audio-meter-fill" style="width: 0%;"></div></div>
    </div>
    <div class="telemetry-card">
        <span class="telemetry-label">Tần số Lấy mẫu & Động cơ</span>
        <span class="telemetry-value">48<span class="telemetry-unit">kHz / CHỜ TỆP</span></span>
    </div>
</div>
"""


def get_initial_stream_telemetry_html(engine_name: Optional[str] = None) -> str:
    """Return default telemetry display for streaming mode."""
    engine_label = f" ({engine_name.upper()})" if engine_name else ""
    return f"""
<div style="font-family: 'JetBrains Mono', Consolas, monospace; font-size: 12px; color: #a1a1aa; padding: 8px 12px; background: #141518; border: 1px solid #27272a; border-radius: 4px;">
    ● Trạng thái Streaming{engine_label}: Sẵn sàng nhận tín hiệu từ Microphone (Độ trễ mục tiêu: &lt; 15ms)
</div>
"""


# ==============================================================================
# 2. ENGINE SELECTION REGISTRY & METADATA HELPERS
# ==============================================================================

DEFAULT_ENGINE = "dfnet2"

ENGINE_METADATA: Dict[str, Dict[str, str]] = {
    "dfnet2": {
        "name": "DeepFilterNet2 (DFNET2)",
        "type": "Mạng Nơ-ron Lọc Âm Thanh Phổ Phức (Full-Band 48kHz)",
        "sample_rate": "48 kHz (Khuyên dùng)",
        "status": "Hoạt động (Active / Default)",
        "description": "Mô hình học sâu lọc sâu 48kHz trên miền tần số phức STFT & ERB. Khử sạch tạp âm đời sống, bảo toàn formant và ngữ điệu giọng nói.",
    },
    "mock": {
        "name": "Mock DSP Engine",
        "type": "Xử lý Tín hiệu Số Cổ điển (Butterworth Bandpass)",
        "sample_rate": "48 kHz / Đa tần số",
        "status": "Hoạt động (Active / CPU)",
        "description": "Bộ lọc dải thông Butterworth 100Hz - 7.5kHz chạy trực tiếp trên CPU, siêu nhẹ (< 0.1ms/chunk), dùng kiểm thử đường ống không cần GPU.",
    },
    "rnnoise": {
        "name": "RNNoise Recurrent Denoiser",
        "type": "Mạng Nơ-ron Hồi quy (Recurrent Neural Network GRU)",
        "sample_rate": "48 kHz",
        "status": "Dự phòng (Standby)",
        "description": "Mô hình RNN lọc tạp âm kinh điển, thiết kế chuyên biệt cho đàm thoại và thoại hội nghị thời gian thực độ trễ thấp.",
    },
    "demucs": {
        "name": "Demucs Waveform Enhancer",
        "type": "Mạng Tích chập Miền Thời gian (Waveform Domain ConvNet)",
        "sample_rate": "48 kHz",
        "status": "Dự phòng (Standby)",
        "description": "Mô hình khôi phục tín hiệu âm thanh trực tiếp trên miền thời gian, tối ưu tái tạo chi tiết dạng sóng.",
    },
}


def resolve_engine_metadata(engine_name: str) -> Dict[str, Any]:
    """Resolve metadata for an engine from dynamic registry first, then fallback table."""
    canonical = engine_name.strip().lower() if engine_name else DEFAULT_ENGINE
    reg_meta = EngineRegistry.get_engine_metadata(canonical)
    if reg_meta.get("source_file"):
        return reg_meta
    if canonical in ENGINE_METADATA:
        return dict(ENGINE_METADATA[canonical])
    return reg_meta


def get_available_engines() -> List[str]:
    """Return list of engine identifiers with DEFAULT_ENGINE prioritized first."""
    all_engines = list(EngineRegistry.list_available_engines())
    if DEFAULT_ENGINE in all_engines:
        all_engines.remove(DEFAULT_ENGINE)
        return [DEFAULT_ENGINE] + sorted(all_engines)
    if all_engines:
        return sorted(all_engines)
    return [DEFAULT_ENGINE]


def get_engine_choices() -> List[Tuple[str, str]]:
    """Return (label, value) pairs for the engine selection dropdown."""
    engines = get_available_engines()
    choices = []
    for eng in engines:
        meta = resolve_engine_metadata(eng)
        display_name = meta.get("name", eng.upper())
        if eng.lower() == DEFAULT_ENGINE.lower():
            label = f"{display_name} [Mặc định]"
        else:
            label = display_name
        choices.append((label, eng))
    return choices


def get_engine_info_html(engine_name: str) -> str:
    """Render informative technical card for the selected engine."""
    canonical = engine_name.strip().lower() if engine_name else DEFAULT_ENGINE
    meta = resolve_engine_metadata(canonical)
    is_active = canonical in EngineRegistry.list_available_engines()
    status_class = "active" if is_active else "standby"
    status_text = meta.get("status", "Sẵn sàng")
    source_file = meta.get("source_file", "")
    source_html = (
        f'<div><strong>Tệp nguồn:</strong> <code>models/{source_file}</code></div>'
        if source_file
        else ""
    )

    return f"""
<div class="daw-engine-meta-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span style="font-weight: 700; color: #22d3ee; font-size: 13px;">{meta['name']}</span>
        <span class="daw-engine-status-tag {status_class}">{status_text}</span>
    </div>
    <div><strong>Kiến trúc:</strong> {meta['type']}</div>
    <div><strong>Tần số chuẩn:</strong> {meta['sample_rate']}</div>
    {source_html}
    <div style="margin-top: 4px; color: #a1a1aa; font-size: 11px;">{meta['description']}</div>
</div>
"""


def on_engine_change(engine_name: str) -> Tuple[str, Any, str]:
    """
    Handle engine selection change from UI:
    1. Update the technical engine information panel.
    2. Reset streaming filter state for the new engine.
    3. Update streaming telemetry banner to reflect the new engine.
    """
    canonical = engine_name.strip().lower() if engine_name else DEFAULT_ENGINE
    info_html = get_engine_info_html(canonical)
    try:
        engine = EngineRegistry.get_engine(canonical)
        new_state = engine.reset_stream_state()
    except Exception:
        new_state = None

    stream_telemetry_html = f"""
<div style="font-family: 'JetBrains Mono', Consolas, monospace; font-size: 12px; color: #34d399; padding: 8px 12px; background: #141518; border: 1px solid #059669; border-radius: 4px;">
    ● Động cơ hoạt động: <strong>{canonical.upper()}</strong> (Sẵn sàng nhận tín hiệu từ Microphone)
</div>
"""
    gr.Info(f"Đã chuyển động cơ khử nhiễu sang: {canonical.upper()}")
    return info_html, new_state, stream_telemetry_html


# ==============================================================================
# 2. CORE PROCESSING LOGIC (CONTRACT CT-05 & STREAMING)
# ==============================================================================

def _resolve_ffmpeg_executable() -> str:
    """Resolve ffmpeg binary path from system PATH or imageio_ffmpeg package."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise AudioValidationError(
            f"Không tìm thấy công cụ ffmpeg để chuyển đổi định dạng âm thanh: {exc}"
        ) from exc


def convert_to_wav_ffmpeg(input_path: str) -> str:
    """
    Convert a non-WAV audio file (e.g. .m4a, .mp3, .aac, .ogg, .flac) to .wav using ffmpeg.

    Parameters:
        input_path: Path to the input audio file.

    Returns:
        Absolute string path to the newly generated .wav file.

    Raises:
        AudioValidationError: If the file does not exist, is empty, or ffmpeg fails to decode.
    """
    path_obj = Path(input_path)
    if not path_obj.exists() or not path_obj.is_file():
        raise AudioValidationError(f"File not found: {input_path}")

    try:
        size_bytes = path_obj.stat().st_size
    except OSError as err:
        raise AudioValidationError(f"Cannot access audio file metadata '{input_path}': {err}") from err

    if size_bytes == 0:
        raise AudioValidationError(f"Audio file is empty (0 bytes): {input_path}")

    ffmpeg_exe = _resolve_ffmpeg_executable()
    conv_dir = Path(tempfile.gettempdir()) / FFMPEG_CACHE_DIR_NAME
    conv_dir.mkdir(parents=True, exist_ok=True)
    out_wav_path = conv_dir / f"ffmpeg_conv_{uuid.uuid4().hex[:12]}.wav"

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(path_obj),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "48000",
        "-ac",
        "1",
        str(out_wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out_wav_path.exists() or out_wav_path.stat().st_size == 0:
        try:
            if out_wav_path.exists():
                out_wav_path.unlink()
        except OSError:
            pass
        stderr_tail = (result.stderr or "").strip().splitlines()[-1:]
        detail = stderr_tail[0] if stderr_tail else "ffmpeg conversion failed"
        raise AudioValidationError(
            f"Corrupt or unsupported audio file '{input_path}': {detail}"
        )

    return str(out_wav_path)


def process_audio(
    input_path: Optional[str],
    engine_name: Optional[str] = None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], Union[str, TelemetryStats], Optional[plt.Figure], Optional[plt.Figure]]:
    """
    Contract CT-05: Orchestrate file-mode audio denoising pipeline.

    1. Ingests & validates audio format via inspect_and_prepare_audio (CT-01),
       automatically converting non-WAV inputs (e.g. .m4a, .mp3, .ogg) to .wav via ffmpeg.
    2. Resolves pluggable AI engine via EngineRegistry.get_engine().
    3. Handles long audio (> 60s) via chunking and Overlap-Add (OLA) reassembly.
    4. Computes execution duration and RTF metrics via summarize_timing (CT-03).
    5. Produces Before and After STFT Spectrograms.
    6. Closes all matplotlib figures to prevent memory accumulation in Gcf.
    7. Cleans up any temporary ffmpeg converted files.
    8. Returns 4-tuple: (output_wav_path, telemetry_stats, spec_before_fig, spec_after_fig).
    """
    if not input_path:
        raise gr.Error("Vui lòng tải lên tệp âm thanh hợp lệ (WAV, MP3, M4A, FLAC, OGG) trước khi bắt đầu xử lý.")

    start_time = time.perf_counter()
    converted_wav_path: Optional[str] = None

    try:
        path_obj = Path(input_path)
        if path_obj.suffix.lower() != ".wav":
            progress(0.02, desc="Đang chuyển đổi tệp sang định dạng WAV bằng ffmpeg...")
            try:
                converted_wav_path = convert_to_wav_ffmpeg(input_path)
            except AudioValidationError as err:
                raise gr.Error(f"Lỗi kiểm tra tệp âm thanh: {err}") from err
            except Exception as err:
                raise gr.Error(f"Không thể chuyển đổi tệp âm thanh bằng ffmpeg: {err}") from err
            target_path = converted_wav_path
        else:
            target_path = input_path

        progress(0.05, desc="Đang kiểm tra & chuẩn hóa định dạng âm thanh...")
        try:
            prep_info = inspect_and_prepare_audio(
                target_path, spec_title="Biểu đồ Phổ Âm thanh Gốc (Before Spectrogram)"
            )
        except AudioValidationError as err:
            raise gr.Error(f"Lỗi kiểm tra tệp âm thanh: {err}") from err
        except Exception as err:
            raise gr.Error(f"Không thể đọc hoặc giải mã tệp âm thanh: {err}") from err

        prep_wav_path = prep_info["path"]
        audio_mono = prep_info["audio_array_mono"]
        duration = prep_info["duration_seconds"]
        sr = prep_info["sample_rate"]
        spec_before_fig = prep_info["spec_fig"]

        resolved_engine = engine_name if engine_name else os.environ.get("DENOISE_ENGINE", None)
        try:
            engine = EngineRegistry.get_engine(resolved_engine)
        except Exception as err:
            raise gr.Error(f"Lỗi khởi tạo động cơ khử nhiễu AI ({resolved_engine or 'mặc định'}): {err}") from err

        cache_dir = get_output_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / f"clean_{uuid.uuid4().hex[:12]}.wav"

        if duration > 60.0:
            progress(0.2, desc="Tệp dài > 60s: Đang phân đoạn và khử nhiễu từng khối (RAM)...")
            chunks = chunk_audio(audio_mono, sr=sr, max_chunk_seconds=60.0, overlap_seconds=0.5)
            clean_chunks = []
            for idx, chunk in enumerate(chunks):
                frac = 0.2 + 0.6 * ((idx + 1) / len(chunks))
                progress(frac, desc=f"Đang xử lý phân đoạn {idx + 1}/{len(chunks)}...")
                clean_chunks.append(engine.process_array(chunk, sample_rate=sr))
            
            progress(0.85, desc="Tái cấu trúc âm thanh Overlap-Add (OLA)...")
            cleaned_audio = reconstruct_audio_ola(
                clean_chunks,
                sr=sr,
                overlap_seconds=0.5,
                total_length=len(audio_mono),
            )
            save_audio_wav(str(output_path), cleaned_audio, sample_rate=sr)
        else:
            progress(0.3, desc="Đang chạy mô hình khử nhiễu âm thanh...")
            engine.process_file(prep_wav_path, str(output_path))
            cleaned_audio, _ = sf.read(str(output_path), dtype="float32")

        progress(0.9, desc="Đang kết xuất biểu đồ phổ Spectrogram...")
        elapsed = time.perf_counter() - start_time
        timing_stats = summarize_timing(elapsed, duration)

        # After Scientific STFT Spectrogram
        spec_after_fig = generate_spectrogram(
            cleaned_audio,
            sr=sr,
            title="Biểu đồ Phổ Sau Khử Nhiễu (After Spectrogram)",
        )

        stats_markdown = format_telemetry_html(timing_stats, sr, engine.name)

        stats_dict = {
            "audio_duration_sec": round(float(duration), 3),
            "processing_time_sec": round(float(elapsed), 3),
            "rtf": round(float(timing_stats.get("rtf", 0.0)), 3),
            "sample_rate": sr,
            "channels": 1,
            "elapsed_seconds": timing_stats.get("elapsed_seconds", round(float(elapsed), 3)),
            "audio_duration_seconds": timing_stats.get("audio_duration_seconds", round(float(duration), 3)),
            "engine_name": engine.name,
        }
        telemetry_stats = TelemetryStats(stats_markdown, stats_dict)

        progress(1.0, desc="Khử nhiễu âm thanh thành công!")

        # Free all figures from matplotlib global Gcf accumulation
        plt.close("all")

        return str(output_path), telemetry_stats, spec_before_fig, spec_after_fig

    except gr.Error:
        plt.close("all")
        raise
    except EngineError as err:
        plt.close("all")
        raise gr.Error(f"Động cơ khử nhiễu báo lỗi: {err}") from err
    except Exception as err:
        plt.close("all")
        raise gr.Error(f"Lỗi không xác định trong quá trình xử lý: {err}") from err
    finally:
        if converted_wav_path:
            try:
                Path(converted_wav_path).unlink(missing_ok=True)
            except Exception:
                pass
        # Always cleanup the temporary file created by inspect_and_prepare_audio
        try:
            Path(prep_wav_path).unlink(missing_ok=True)
        except Exception:
            pass


def process_stream_chunk(
    audio_chunk: Any,
    state: Any = None,
    engine_name: Optional[str] = None,
) -> Tuple[Optional[Tuple[int, np.ndarray]], Any]:
    """
    Handle real-time microphone audio chunk inference with state continuity.

    Parameters:
        audio_chunk: Tuple of (sample_rate, np.ndarray) or raw np.ndarray from WebRTC mic stream.
        state: Previous filter memory state.
        engine_name: Name of AI engine to use (default resolves to dfnet2 / env setting).

    Returns:
        Tuple containing:
            - Tuple (sample_rate, cleaned_audio_array) for playback
            - Updated state object for the next streaming chunk
    """
    if audio_chunk is None:
        return None, state

    if isinstance(audio_chunk, tuple):
        sr, chunk = audio_chunk
    else:
        sr = 48000
        chunk = audio_chunk

    if chunk is None or (isinstance(chunk, np.ndarray) and chunk.size == 0):
        return None, state

    # Downmix multi-channel to mono if necessary
    if isinstance(chunk, np.ndarray) and chunk.ndim > 1:
        chunk = downmix_to_mono(chunk)

    orig_dtype = chunk.dtype if isinstance(chunk, np.ndarray) else None

    resolved_engine = engine_name if engine_name else os.environ.get("DENOISE_ENGINE", None)
    engine = EngineRegistry.get_engine(resolved_engine)
    cleaned_chunk, new_state = engine.process_stream_chunk(chunk, sample_rate=sr, state=state)

    # Return matching format: if input was integer, return int16 to avoid Gradio audio warnings
    if orig_dtype is not None and np.issubdtype(orig_dtype, np.integer):
        clean_norm = np.clip(cleaned_chunk, -1.0, 1.0)
        clean_out = (clean_norm * 32767.0).astype(np.int16)
    else:
        clean_out = cleaned_chunk.astype(np.float32)

    return (sr, clean_out), new_state


def reset_stream_session(engine_name: Optional[str] = None) -> Tuple[Any, str]:
    """Reset streaming filter state and buffer."""
    resolved_engine = engine_name if engine_name else os.environ.get("DENOISE_ENGINE", None)
    engine = EngineRegistry.get_engine(resolved_engine)
    new_state = engine.reset_stream_state()
    feedback_html = f"""
<div style="font-family: 'JetBrains Mono', Consolas, monospace; font-size: 12px; color: #10b981; padding: 8px 12px; background: #141518; border: 1px solid #059669; border-radius: 4px;">
    ✓ Bộ đệm streaming đã được đặt lại thành công ({engine.name.upper()}). Phiên mới sẵn sàng.
</div>
"""
    return new_state, feedback_html


# ==============================================================================
# 3. GRADIO APPLICATION WORKSTATION BUILDER
# ==============================================================================

def create_app() -> gr.Blocks:
    """
    Build the complete Gradio DAW audio workstation application.

    Returns:
        gr.Blocks instance configured with File & Live Streaming modes.
    """
    with gr.Blocks(title="ACOUSTIC STUDIO // AI AUDIO DENOISER") as demo:
        # Header Banner Rack
        with gr.Row(elem_classes=["daw-header"]):
            with gr.Column(scale=3):
                gr.HTML("""
                <div class="daw-header-title">
                    <h1>ACOUSTIC STUDIO // AI AUDIO DENOISER</h1>
                    <p>Hệ thống Khử nhiễu Âm thanh Thời gian thực & Phân tích Phổ Tần số Chuyên sâu</p>
                </div>
                """)
            with gr.Row(scale=1, elem_classes=["daw-header-badges"]):
                engine_btn = gr.Button(
                    "ENGINE: READY",
                    elem_id="daw-engine-ready-btn",
                    elem_classes=["daw-badge", "daw-engine-btn"],
                    size="sm",
                    scale=0,
                    min_width=0,
                )
                gr.HTML(
                    '<span class="daw-badge daw-badge-mono">48 kHz MONO</span>',
                    elem_classes=["daw-badge-wrapper"],
                )

        # Collapsible Engine Selection Rack (Toggled by "ENGINE: READY" button)
        engine_rack_state = gr.State(False)
        with gr.Column(visible=False, elem_id="daw-engine-rack", elem_classes=["daw-engine-rack"]) as engine_rack:
            with gr.Row(elem_classes=["daw-engine-rack-header"]):
                gr.HTML("""
                <div class="daw-engine-rack-title">
                    ⚙️ BẢNG ĐIỀU KHIỂN ĐỘNG CƠ KHỬ NHIỄU // AI DENOISE ENGINE SELECTOR
                </div>
                """)
                close_rack_btn = gr.Button(
                    "✕ Đóng",
                    size="sm",
                    variant="secondary",
                    scale=0,
                    min_width=84,
                    elem_classes=["daw-close-rack-btn"],
                )

            with gr.Row():
                with gr.Column(scale=3):
                    initial_choices = get_engine_choices()
                    initial_engine = (
                        DEFAULT_ENGINE
                        if any(val == DEFAULT_ENGINE for _, val in initial_choices)
                        else (initial_choices[0][1] if initial_choices else DEFAULT_ENGINE)
                    )
                    selected_engine = gr.Dropdown(
                        choices=initial_choices,
                        value=initial_engine,
                        label="LỰA CHỌN ĐỘNG CƠ KHỬ NHIỄU (ACTIVE ENGINE)",
                        info="Chọn mô hình khử nhiễu sẽ được áp dụng cho cả Xử lý Tệp (File Mode) và Khử Nhiễu Trực Tiếp (Live Streaming).",
                        interactive=True,
                    )
                with gr.Column(scale=2):
                    engine_info_box = gr.HTML(value=get_engine_info_html(initial_engine))

        # Shared streaming session state
        stream_state = gr.State(None)

        def toggle_engine_rack(is_visible: bool):
            return gr.update(visible=not is_visible), not is_visible

        engine_btn.click(
            fn=toggle_engine_rack,
            inputs=[engine_rack_state],
            outputs=[engine_rack, engine_rack_state],
        )

        close_rack_btn.click(
            fn=lambda: (gr.update(visible=False), False),
            inputs=None,
            outputs=[engine_rack, engine_rack_state],
        )

        with gr.Tabs():
            # ------------------------------------------------------------------
            # TAB 1: File Batch Processing Mode
            # ------------------------------------------------------------------
            with gr.Tab("Xử lý Tệp (File Mode)"):
                with gr.Row(equal_height=True, elem_classes=["daw-audio-comparison-row"]):
                    with gr.Column(scale=1):
                        input_audio = gr.Audio(
                            label="Âm thanh Gốc (Before) — Upload WAV, MP3, FLAC, OGG, M4A",
                            sources=["upload", "microphone"],
                            type="filepath",
                            elem_classes=["daw-audio-panel", "daw-audio-file-panel"],
                        )
                    with gr.Column(scale=1):
                        output_audio = gr.Audio(
                            label="Âm thanh Đã Khử Nhiễu (After)",
                            type="filepath",
                            interactive=False,
                            elem_classes=["daw-audio-panel", "daw-audio-file-panel"],
                        )

                process_btn = gr.Button("Khử Nhiễu Âm Thanh", variant="primary", elem_classes=["daw-process-btn"])

                # Telemetry Rack
                telemetry_panel = gr.HTML(value=get_initial_telemetry_html())

                # Spectrogram Comparison
                with gr.Row():
                    spec_before = gr.Plot(label="Biểu đồ Phổ Tín hiệu Gốc (Before Spectrogram)")
                    spec_after = gr.Plot(label="Biểu đồ Phổ Sau Khử Nhiễu (After Spectrogram)")

                process_btn.click(
                    fn=process_audio,
                    inputs=[input_audio, selected_engine],
                    outputs=[output_audio, telemetry_panel, spec_before, spec_after],
                )

            # ------------------------------------------------------------------
            # TAB 2: Live Streaming Denoising Mode
            # ------------------------------------------------------------------
            with gr.Tab("Khử Nhiễu Trực Tiếp (Live Streaming)"):
                with gr.Row(equal_height=True, elem_classes=["daw-audio-comparison-row"]):
                    with gr.Column(scale=1):
                        mic_input = gr.Audio(
                            label="Microphone Luồng Vào (Live Stream In)",
                            sources=["microphone"],
                            streaming=True,
                            type="numpy",
                            elem_classes=["daw-audio-panel", "daw-audio-stream-panel"],
                        )
                    with gr.Column(scale=1):
                        live_output = gr.Audio(
                            label="Âm thanh Đã Khử Nhiễu (Live Output)",
                            streaming=True,
                            autoplay=True,
                            elem_classes=["daw-audio-panel", "daw-audio-stream-panel"],
                        )

                reset_btn = gr.Button("Đặt lại Bộ đệm (Reset Stream)", variant="secondary", elem_classes=["daw-process-btn"])

                stream_telemetry = gr.HTML(value=get_initial_stream_telemetry_html(DEFAULT_ENGINE))

                # Streaming Event Connection
                mic_input.stream(
                    fn=process_stream_chunk,
                    inputs=[mic_input, stream_state, selected_engine],
                    outputs=[live_output, stream_state],
                    stream_every=0.2,
                )

                reset_btn.click(
                    fn=reset_stream_session,
                    inputs=[selected_engine],
                    outputs=[stream_state, stream_telemetry],
                )

        # Connect engine change listener
        selected_engine.change(
            fn=on_engine_change,
            inputs=[selected_engine],
            outputs=[engine_info_box, stream_state, stream_telemetry],
        )

    return demo


if __name__ == "__main__":
    # Rà soát thư mục models/ mỗi lần chạy app:
    # - Xoá các model đã đặt tên nhưng không còn thấy file .py tương ứng
    # - Yêu cầu người dùng đặt tên hiển thị nếu phát hiện file model .py mới
    EngineRegistry.sync_models(interactive=True)
    demo = create_app()
    os.environ.setdefault("DENOISE_ENGINE", DEFAULT_ENGINE)
    theme, css = get_daw_theme()
    cleanup_summary = cleanup_stale_temp_files()
    if cleanup_summary["total"] > 0:
        print(
            f"[Cleanup] Đã dọn {cleanup_summary['total']} tệp tạm từ phiên trước "
            f"(outputs={cleanup_summary['outputs']}, gradio={cleanup_summary['gradio_cache']}, "
            f"ffmpeg={cleanup_summary['ffmpeg_cache']}, prep={cleanup_summary['prep_cache']}, "
            f"temp_root={cleanup_summary['temp_root']})."
        )
    demo.launch(theme=theme, css=css)
else:
    demo = create_app()