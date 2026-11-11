"""
models/dfnet_engine.py - DeepFilterNet2 AI Audio Denoising Engine.

Real neural-network implementation of AudioDenoiseEngine (Contract CT-02) backed
by the pretrained DeepFilterNet2 model (package ``deepfilternet``).

Activate with:
    $env:DENOISE_ENGINE="dfnet2"; python app.py        (PowerShell)

Design notes:
- PyTorch and ``df`` are imported lazily inside initialize(), so importing this
  module never pulls in deep learning libraries until the engine is created.
- Audio I/O is done with soundfile (not torchaudio) to avoid backend issues.
- File mode: whole file is resampled to 48 kHz mono and enhanced in one pass.
- Streaming mode: DeepFilterNet's public API is offline, so streaming uses a
  sliding context window. Each call enhances ``history + new_chunk`` and returns
  only the samples that correspond to the new chunk. Quality is slightly lower
  at chunk edges than in file mode, and compute per call grows with the window.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# ==============================================================================
# 1. STREAMING STATE
# ==============================================================================

@dataclass
class DFNetStreamState:
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


# ==============================================================================
# 2. ENGINE
# ==============================================================================

@EngineRegistry.register("dfnet2", aliases=["deepfilter", "df2"])
class DeepFilterNetEngine(AudioDenoiseEngine):
    """DeepFilterNet2 full-band (48 kHz) neural speech enhancement engine."""

    def __init__(
        self,
        model_name: str = "DeepFilterNet2",
        context_seconds: float = 1.0,
        post_filter: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_name: Pretrained model identifier passed to ``df.init_df``.
            context_seconds: Streaming history window length in seconds.
            post_filter: Enable DeepFilterNet's extra post-filter (stronger suppression).
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.context_seconds = max(0.0, float(context_seconds))
        self.post_filter = bool(post_filter)

        self._model: Any = None
        self._df_state: Any = None
        self._torch: Any = None
        self._enhance_fn: Any = None
        self._infer_lock = threading.Lock()

    # ---- properties ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "dfnet2"

    @property
    def description(self) -> str:
        return f"DeepFilterNet2 neural denoiser (model='{self.model_name}', 48 kHz mono)"

    @property
    def supported_sample_rates(self) -> List[int]:
        # Other rates are resampled internally, so these are all accepted.
        return [8000, 16000, 22050, 32000, 44100, 48000]

    # ---- lifecycle ----------------------------------------------------------

    def initialize(self, **kwargs: Any) -> None:
        """Load PyTorch and the pretrained model (idempotent)."""
        if self._is_initialized and self._model is not None:
            return

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
                model_base_dir=self.model_name,
                post_filter=self.post_filter,
                log_file=None,
                default_model=self.model_name,
            )
        except Exception as exc:
            raise EngineInitializationError(
                f"Không thể tải mô hình '{self.model_name}': {exc}",
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
        self._is_initialized = True
        logger.info("Initialized DeepFilterNetEngine (model='%s')", self.model_name)

    # ---- core inference -----------------------------------------------------

    def _enhance_48k(self, audio: np.ndarray) -> np.ndarray:
        """Run DeepFilterNet on a 1D float32 array at 48 kHz; returns same length."""
        if not self._is_initialized:
            self.initialize()
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        with self._infer_lock, torch.no_grad():
            out = self._enhance_fn(self._model, self._df_state, tensor)
        out_np = out.squeeze(0).cpu().numpy().astype(np.float32)
        if out_np.size >= audio.size:
            return out_np[: audio.size]
        padded = np.zeros(audio.size, dtype=np.float32)
        padded[: out_np.size] = out_np
        return padded

    # ---- file mode ----------------------------------------------------------

    def process_array(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        if not self._is_initialized:
            self.initialize()
        
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        
        # Resample
        if sample_rate != MODEL_SR:
            audio = _resample(audio, sample_rate, MODEL_SR)
            
        cleaned = self._enhance_48k(audio)
        return np.clip(np.nan_to_num(cleaned), -1.0, 1.0).astype(np.float32)

    # ---- file mode ----------------------------------------------------------

    def process_file(self, input_path: str, output_path: str) -> FileProcessResult:
        t0 = time.perf_counter()
        in_p, out_p = Path(input_path), Path(output_path)

        if not in_p.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")
        
        data, sr = sf.read(str(in_p), dtype="float32")
        duration = data.size / sr

        cleaned = self.process_array(data, sr)

        try:
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
                "samples_processed": int(data.size),
            },
            message=f"DeepFilterNet2 denoising completed (RTF={rtf:.3f})",
        )

    # ---- streaming mode -----------------------------------------------------

    def reset_stream_state(self, sample_rate: int = MODEL_SR) -> DFNetStreamState:
        return DFNetStreamState(sample_rate=sample_rate)

    def process_stream_chunk(
        self,
        chunk: np.ndarray,
        sample_rate: int = MODEL_SR,
        state: Optional[Any] = None,
    ) -> Tuple[np.ndarray, DFNetStreamState]:
        t0 = time.perf_counter()
        chunk_clean = self.validate_stream_chunk(chunk, sample_rate)

        if not isinstance(state, DFNetStreamState):
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
                f"DeepFilterNet streaming inference failed: {exc}",
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
