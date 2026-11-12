"""
models/dfn3_vivos.py - Fine-Tuned DeepFilterNet3 (VIVOS v1) AI Audio Denoising Engine.

Pluggable AudioDenoiseEngine implementation (Contract CT-02) bridging the fine-tuned
DeepFilterNet3 checkpoint located in ``final_models/dfn3_ft_vivos_v1`` into the
application's modular engine architecture.

Activate with:
    $env:DENOISE_ENGINE="dfn3_ft_vivos_v1"; python app.py   (PowerShell)
    # or using alias:
    $env:DENOISE_ENGINE="dfn3_vivos"; python app.py

Design notes:
- PyTorch and ``df`` are imported lazily inside initialize(), so importing or AST-scanning
  this module never pulls in deep learning libraries until the engine is instantiated.
- Audio I/O uses soundfile (not torchaudio) to avoid backend issues.
- File mode: whole file is downmixed to mono, resampled to 48 kHz, padded to a multiple
  of ``df_state.hop_size()``, enhanced in one pass, and trimmed to the exact input length.
- Streaming mode: uses a rolling input context window (``history + new_chunk``) and
  returns only the samples corresponding to ``new_chunk``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal
import soundfile as sf

from src.engine_interface import (
    AudioDenoiseEngine,
    EngineInitializationError,
    EngineProcessingError,
    EngineRegistry,
    EngineValidationError,
    FileProcessResult,
)

logger = logging.getLogger(__name__)

MODEL_SR = 48000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_REL_PATH = Path("final_models") / "dfn3_ft_vivos_v1"


# ==============================================================================
# 1. STREAMING STATE
# ==============================================================================

@dataclass
class DFN3VivosStreamState:
    """Session state for streaming: rolling input history at the caller's sample rate."""
    history: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sample_rate: int = MODEL_SR
    chunk_count: int = 0
    total_samples: int = 0
    total_elapsed_ms: float = 0.0
    last_chunk_latency_ms: float = 0.0

    def reset(self) -> None:
        self.history = np.zeros(0, dtype=np.float32)
        self.chunk_count = 0
        self.total_samples = 0
        self.total_elapsed_ms = 0.0
        self.last_chunk_latency_ms = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_count": self.chunk_count,
            "total_samples": self.total_samples,
            "sample_rate": self.sample_rate,
            "history_samples": int(self.history.size),
            "total_elapsed_ms": round(self.total_elapsed_ms, 3),
            "last_chunk_latency_ms": round(self.last_chunk_latency_ms, 3),
        }


def _resample(data: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Polyphase resample of a 1D float32 array."""
    if sr_from == sr_to or data.size == 0:
        return data.astype(np.float32, copy=False)
    g = math.gcd(sr_to, sr_from)
    return signal.resample_poly(data, sr_to // g, sr_from // g).astype(np.float32)


def _resolve_model_dir(custom_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the directory containing config.ini and checkpoints/ for dfn3_ft_vivos_v1."""
    candidates: List[Path] = []
    if custom_dir is not None and str(custom_dir).strip():
        p = Path(custom_dir)
        candidates.append(p if p.is_absolute() else (PROJECT_ROOT / p))

    env_dir = os.environ.get("DFN3_VIVOS_MODEL_DIR", "").strip()
    if env_dir:
        p_env = Path(env_dir)
        candidates.append(p_env if p_env.is_absolute() else (PROJECT_ROOT / p_env))

    candidates.extend([
        PROJECT_ROOT / DEFAULT_MODEL_REL_PATH,
        PROJECT_ROOT / "models" / "dfn3_ft_vivos_v1",
        PROJECT_ROOT / "ml" / "final_models" / "dfn3_ft_vivos_v1",
    ])

    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand.resolve()

    return (PROJECT_ROOT / DEFAULT_MODEL_REL_PATH).resolve()


# ==============================================================================
# 2. ENGINE
# ==============================================================================

@EngineRegistry.register("dfn3_ft_vivos_v1", aliases=["dfn3_vivos", "vivos", "bridge"])
class DFN3VivosEngine(AudioDenoiseEngine):
    """DeepFilterNet3 fine-tuned on Vietnamese VIVOS + UrbanSound8K (48 kHz mono)."""

    ENGINE_ID = "dfn3_ft_vivos_v1"
    DISPLAY_NAME = "DeepFilterNet3 VIVOS Fine-Tuned (DFN3-VIVOS-V1)"
    ENGINE_TYPE = "Mạng Nơ-ron Lọc Âm Thanh Phổ Phức Tinh Chỉnh Tiếng Việt (Full-Band 48kHz)"
    SAMPLE_RATE_LABEL = "48 kHz (Khuyên dùng)"
    STATUS_LABEL = "Hoạt động (Active / Fine-Tuned)"
    DESCRIPTION = (
        "Mô hình DeepFilterNet3 tinh chỉnh trên dữ liệu tiếng Việt VIVOS + UrbanSound8K "
        "(SI-SDR 17.51 → 23.37 dB). Khử sạch tạp âm đời sống và bảo toàn âm sắc, thanh điệu tiếng Việt."
    )

    def __init__(
        self,
        model_base_dir: Optional[Union[str, Path]] = None,
        context_seconds: float = 1.0,
        post_filter: bool = False,
        atten_lim_db: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_base_dir: Path to the fine-tuned DeepFilterNet3 model directory
                (defaults to ``final_models/dfn3_ft_vivos_v1``).
            context_seconds: Streaming history window length in seconds.
            post_filter: Enable DeepFilterNet's extra post-filter (stronger suppression).
            atten_lim_db: Optional attenuation limit in dB passed to ``df.enhance.enhance``.
        """
        super().__init__(**kwargs)
        self.model_dir: Path = _resolve_model_dir(model_base_dir)
        self.model_name: str = self.model_dir.name
        self.context_seconds: float = max(0.0, float(context_seconds))
        self.post_filter: bool = bool(post_filter)
        self.atten_lim_db: Optional[float] = (
            float(atten_lim_db) if atten_lim_db is not None else None
        )

        self._model: Any = None
        self._df_state: Any = None
        self._torch: Any = None
        self._enhance_fn: Any = None
        self._hop_size: int = 480
        self._model_metadata: Dict[str, Any] = {}
        self._infer_lock = threading.Lock()

    # ---- properties ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "dfn3_ft_vivos_v1"

    @property
    def description(self) -> str:
        version = self._model_metadata.get("version", "0.1.0")
        return (
            f"DeepFilterNet3 VIVOS Fine-Tuned v{version} "
            f"(dir='{self.model_dir.name}', 48 kHz mono)"
        )

    @property
    def supported_sample_rates(self) -> List[int]:
        # Other rates are resampled internally, so these are all accepted.
        return [8000, 16000, 22050, 32000, 44100, 48000]

    # ---- lifecycle ----------------------------------------------------------

    def initialize(self, **kwargs: Any) -> None:
        """Validate local checkpoint directory, load PyTorch, and initialize DeepFilterNet3 (idempotent)."""
        if self._is_initialized and self._model is not None:
            return

        if not self.model_dir.exists() or not self.model_dir.is_dir():
            raise EngineInitializationError(
                f"Không tìm thấy thư mục mô hình tinh chỉnh tại: '{self.model_dir}'",
                engine_name=self.name,
            )

        config_path = self.model_dir / "config.ini"
        ckpt_dir = self.model_dir / "checkpoints"
        if not config_path.is_file():
            raise EngineInitializationError(
                f"Thiếu tệp cấu hình 'config.ini' trong thư mục mô hình: '{self.model_dir}'",
                engine_name=self.name,
            )
        if not ckpt_dir.is_dir() or not any(ckpt_dir.iterdir()):
            raise EngineInitializationError(
                f"Thiếu checkpoint trọng số trong thư mục: '{ckpt_dir}'",
                engine_name=self.name,
            )

        meta_path = self.model_dir / "metadata.json"
        if meta_path.is_file():
            try:
                raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(raw_meta, dict):
                    self._model_metadata = raw_meta
            except Exception as exc:
                logger.warning("Không thể đọc metadata.json tại '%s': %s", meta_path, exc)

        try:
            import sys
            import types
            import torch

            # Compatibility shim: deepfilternet 0.5.6 imports
            # `torchaudio.backend.common.AudioMetaData`, removed in torchaudio >= 2.2.
            if "torchaudio.backend.common" not in sys.modules:
                try:
                    import torchaudio.backend.common  # type: ignore  # noqa: F401
                except ImportError:
                    backend_mod = sys.modules.setdefault(
                        "torchaudio.backend", types.ModuleType("torchaudio.backend")
                    )
                    common_mod = types.ModuleType("torchaudio.backend.common")
                    common_mod.AudioMetaData = object  # type: ignore[attr-defined]
                    backend_mod.common = common_mod  # type: ignore[attr-defined]
                    sys.modules["torchaudio.backend.common"] = common_mod

            if "libdf" not in sys.modules:
                try:
                    import libdf  # type: ignore  # noqa: F401
                except ImportError:
                    from src import _libdf_compat
                    sys.modules["libdf"] = _libdf_compat

            from df.enhance import enhance, init_df
        except ImportError as exc:
            raise EngineInitializationError(
                "Thiếu thư viện DeepFilterNet/PyTorch. Cài bằng: "
                "pip install torch torchaudio deepfilternet "
                f"(chi tiết: {exc})",
                engine_name=self.name,
                cause=exc,
            ) from exc

        try:
            model, df_state, _ = init_df(
                model_base_dir=str(self.model_dir),
                post_filter=self.post_filter,
                log_file=None,
            )
        except Exception as exc:
            raise EngineInitializationError(
                f"Không thể tải mô hình tinh chỉnh '{self.model_dir}': {exc}",
                engine_name=self.name,
                cause=exc,
            ) from exc

        model_sr = int(df_state.sr())
        if model_sr != MODEL_SR:
            raise EngineInitializationError(
                f"Mô hình có sample rate {model_sr} Hz, engine yêu cầu {MODEL_SR} Hz.",
                engine_name=self.name,
            )

        self._torch = torch
        self._enhance_fn = enhance
        self._model = model
        self._df_state = df_state
        self._hop_size = int(df_state.hop_size()) if hasattr(df_state, "hop_size") else 480
        self._is_initialized = True
        logger.info("Initialized DFN3VivosEngine (model_dir='%s')", self.model_dir)

    # ---- core inference -----------------------------------------------------

    def _enhance_48k(self, audio: np.ndarray) -> np.ndarray:
        """Run fine-tuned DeepFilterNet3 on a 1D float32 array at 48 kHz; returns exact same length."""
        if not self._is_initialized:
            self.initialize()

        orig_len = int(audio.size)
        if orig_len == 0:
            return np.zeros(0, dtype=np.float32)

        audio_1d = np.ascontiguousarray(np.nan_to_num(audio), dtype=np.float32)
        hop = max(1, self._hop_size)
        pad = (-orig_len) % hop
        if pad:
            audio_1d = np.pad(audio_1d, (0, pad), mode="constant")

        torch = self._torch
        tensor = torch.from_numpy(audio_1d).unsqueeze(0)  # CPU tensor; df.enhance handles device
        with self._infer_lock, torch.no_grad():
            out = self._enhance_fn(
                self._model,
                self._df_state,
                tensor,
                atten_lim_db=self.atten_lim_db,
            )

        out_np = out.squeeze(0).cpu().numpy().astype(np.float32)
        if out_np.size >= orig_len:
            return out_np[:orig_len]
        padded = np.zeros(orig_len, dtype=np.float32)
        padded[: out_np.size] = out_np
        return padded

    # ---- array / file mode --------------------------------------------------

    def process_array(self, audio: np.ndarray, sample_rate: int = MODEL_SR) -> np.ndarray:
        """Process an in-memory audio array and return a 48 kHz mono float32 array."""
        if audio is None or not isinstance(audio, np.ndarray):
            raise EngineValidationError(
                "Input audio must be a non-null NumPy ndarray",
                engine_name=self.name,
            )
        if sample_rate <= 0:
            raise EngineValidationError(
                f"Invalid sample rate: {sample_rate}. Must be positive.",
                engine_name=self.name,
            )

        if not self._is_initialized:
            self.initialize()

        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=-1)
        elif arr.ndim == 0:
            arr = arr.reshape(-1)

        if arr.size == 0:
            return np.zeros(0, dtype=np.float32)

        if sample_rate != MODEL_SR:
            arr = _resample(arr, sample_rate, MODEL_SR)

        cleaned = self._enhance_48k(arr)
        return np.clip(np.nan_to_num(cleaned), -1.0, 1.0).astype(np.float32)

    def process_file(self, input_path: str, output_path: str) -> FileProcessResult:
        """Process a full audio file synchronously (Contract CT-02)."""
        t0 = time.perf_counter()
        in_p, out_p = Path(input_path), Path(output_path)

        if not in_p.exists() or not in_p.is_file():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")

        try:
            data, sr = sf.read(str(in_p), dtype="float32")
        except Exception as exc:
            raise EngineProcessingError(
                f"Failed to read input audio file '{input_path}': {exc}",
                engine_name=self.name,
            ) from exc

        total_frames = int(data.shape[0]) if data.ndim > 1 else int(data.size)
        duration = total_frames / float(sr) if sr > 0 else 0.0

        cleaned = self.process_array(data, sr)

        try:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_p), cleaned, MODEL_SR, subtype="PCM_16")
        except Exception as exc:
            raise EngineProcessingError(
                f"Failed to write output audio file '{output_path}': {exc}",
                engine_name=self.name,
            ) from exc

        elapsed = time.perf_counter() - t0
        rtf = elapsed / duration if duration > 0 else 0.0
        return FileProcessResult(
            status="success",
            output_path=str(out_p.resolve()),
            metrics={
                "elapsed_seconds": round(elapsed, 4),
                "audio_duration_seconds": round(duration, 4),
                "rtf": round(rtf, 4),
                "sample_rate": MODEL_SR,
                "channels": 1,
                "model": self.model_name,
                "model_version": self._model_metadata.get("version", "0.1.0"),
                "base_arch": self._model_metadata.get("config", {}).get("base", "DeepFilterNet3"),
                "samples_processed": int(cleaned.size),
            },
            message=f"DeepFilterNet3 VIVOS fine-tuned denoising completed (RTF={rtf:.3f})",
        )

    # ---- streaming mode -----------------------------------------------------

    def reset_stream_state(self, sample_rate: int = MODEL_SR) -> DFN3VivosStreamState:
        return DFN3VivosStreamState(sample_rate=sample_rate)

    def process_stream_chunk(
        self,
        chunk: np.ndarray,
        sample_rate: int = MODEL_SR,
        state: Optional[Any] = None,
    ) -> Tuple[np.ndarray, DFN3VivosStreamState]:
        t0 = time.perf_counter()
        chunk_clean = self.validate_stream_chunk(chunk, sample_rate)

        if not isinstance(state, DFN3VivosStreamState):
            state = self.reset_stream_state(sample_rate)
        elif state.sample_rate != sample_rate:
            state.reset()
            state.sample_rate = sample_rate

        if chunk_clean.size == 0:
            return chunk_clean, state

        # Enhance (history + chunk) at 48 kHz, keep only the part for this chunk.
        window = np.concatenate([state.history, chunk_clean])
        try:
            window_48k = _resample(window, sample_rate, MODEL_SR)
            enhanced_48k = self._enhance_48k(window_48k)
            enhanced = _resample(enhanced_48k, MODEL_SR, sample_rate)
        except EngineInitializationError:
            raise
        except Exception as exc:
            raise EngineProcessingError(
                f"DeepFilterNet3 VIVOS streaming inference failed: {exc}",
                engine_name=self.name,
                is_streaming=True,
            ) from exc

        n = chunk_clean.size
        if enhanced.size >= n:
            out_chunk = enhanced[-n:]
        else:
            out_chunk = np.zeros(n, dtype=np.float32)
            out_chunk[-enhanced.size:] = enhanced
        out_chunk = np.clip(np.nan_to_num(out_chunk), -1.0, 1.0).astype(np.float32)

        # Roll the (raw, un-enhanced) input history forward.
        max_hist = int(self.context_seconds * sample_rate)
        state.history = window[-max_hist:].copy() if max_hist > 0 else np.zeros(0, np.float32)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        state.chunk_count += 1
        state.total_samples += n
        state.total_elapsed_ms += elapsed_ms
        state.last_chunk_latency_ms = elapsed_ms
        return out_chunk, state
