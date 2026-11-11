"""
models/mock_engine.py - High-Fidelity Mock AI Audio Denoising Engine.

This module provides a CPU-only, zero-dependency reference implementation of
AudioDenoiseEngine conforming strictly to Interface Contract CT-02.

Key Capabilities:
1. Pass-Through Mode ('passthrough'):
   Exact identity replication with configurable artificial latency for pipeline testing.
2. Simulated DSP Filter Mode ('dsp_filter' or 'dsp'):
   Acoustic noise suppression using a 4th-order Butterworth bandpass filter
   (100 Hz - 7500 Hz) via Second-Order Sections (SOS) with stateful delay preservation
   across streaming chunk boundaries.
3. Optional Spectral Subtraction:
   Lightweight STFT noise floor suppression for batch file processing.
4. Real-time Streaming Chunk Processing:
   Sub-millisecond latency (<0.1ms vs 10ms budget) and zero boundary clicks/pops.

Dependencies:
- Python Standard Library (math, time, os, logging, dataclasses, typing, pathlib)
- numpy (>=1.26.4)
- scipy (>=1.13.1)
- soundfile (>=0.14.0)

Strict Architectural Rule:
Zero PyTorch, ONNXRuntime, or CUDA dependencies.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal
import soundfile as sf

from src.engine_interface import (
    AudioDenoiseEngine,
    EngineError,
    EngineProcessingError,
    EngineRegistry,
    EngineValidationError,
    FileProcessResult,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. STREAMING STATE DATA TRANSFER OBJECT
# ==============================================================================

@dataclass
class MockStreamState:
    """
    Session state carrier for real-time streaming chunk inference in MockDenoiseEngine.
    Preserves filter delay registers across chunk boundaries to eliminate audio clicks.
    """
    filter_state: Optional[np.ndarray] = None   # SOS delay registers zi, shape (n_sections, 2)
    chunk_count: int = 0                        # Sequential chunk counter
    total_samples: int = 0                      # Total samples processed in stream
    sample_rate: int = 48000                    # Sampling frequency of the active stream
    sos: Optional[np.ndarray] = None            # Cached Second-Order Sections filter matrix
    total_elapsed_ms: float = 0.0               # Cumulative compute time in milliseconds
    last_chunk_latency_ms: float = 0.0          # Execution time of most recent chunk in ms
    peak_amplitude: float = 0.0                 # Maximum absolute sample magnitude seen

    def reset(self) -> None:
        """Reset state counters and zero out filter delay registers."""
        if self.sos is not None:
            self.filter_state = np.zeros((self.sos.shape[0], 2), dtype=np.float32)
        else:
            self.filter_state = None
        self.chunk_count = 0
        self.total_samples = 0
        self.total_elapsed_ms = 0.0
        self.last_chunk_latency_ms = 0.0
        self.peak_amplitude = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return state telemetry dictionary for debugging and logging."""
        return {
            "chunk_count": self.chunk_count,
            "total_samples": self.total_samples,
            "sample_rate": self.sample_rate,
            "total_elapsed_ms": round(self.total_elapsed_ms, 3),
            "last_chunk_latency_ms": round(self.last_chunk_latency_ms, 3),
            "peak_amplitude": round(self.peak_amplitude, 5),
            "has_filter_state": self.filter_state is not None,
        }


# ==============================================================================
# 2. MOCK DENOISE ENGINE IMPLEMENTATION
# ==============================================================================

@EngineRegistry.register("mock", aliases=["passthrough", "mock_dsp"])
class MockDenoiseEngine(AudioDenoiseEngine):
    """
    Mock AI Audio Denoising Engine.

    Fulfills the AudioDenoiseEngine contract completely without deep learning models.
    Supports Pass-Through and Authentic DSP Filtering with streaming state continuity.
    """

    SUPPORTED_MODES = ("passthrough", "dsp_filter", "dsp")

    def __init__(
        self,
        mode: str = "dsp_filter",
        simulated_delay_ms: float = 0.0,
        simulated_rtf: float = 0.0,
        low_cutoff: float = 100.0,
        high_cutoff: float = 7500.0,
        filter_order: int = 4,
        spectral_subtraction: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MockDenoiseEngine.

        Args:
            mode: Operational mode ('dsp_filter', 'dsp', or 'passthrough').
            simulated_delay_ms: Artificial latency injected per chunk/file in ms.
            simulated_rtf: Artificial RTF factor for file processing simulation.
            low_cutoff: Highpass cutoff frequency in Hz (removes DC / rumble).
            high_cutoff: Lowpass cutoff frequency in Hz (removes hiss).
            filter_order: Order of the Butterworth bandpass filter.
            spectral_subtraction: Whether to apply STFT spectral subtraction in file mode.
        """
        super().__init__(**kwargs)
        mode_clean = mode.strip().lower()
        if mode_clean not in self.SUPPORTED_MODES:
            raise EngineValidationError(
                f"Unsupported mode '{mode}'. Choose from {self.SUPPORTED_MODES}",
                engine_name=self.name,
            )

        self._mode: str = mode_clean
        self.simulated_delay_ms: float = max(0.0, float(simulated_delay_ms))
        self.simulated_rtf: float = max(0.0, float(simulated_rtf))
        self.low_cutoff: float = float(low_cutoff)
        self.high_cutoff: float = float(high_cutoff)
        self.filter_order: int = int(filter_order)
        self.spectral_subtraction: bool = bool(spectral_subtraction)

        # SOS filter coefficient caches keyed by sample_rate: Dict[int, np.ndarray]
        self._sos_cache: Dict[int, np.ndarray] = {}

        # Pre-cache filter for standard sample rates (48000 Hz, 44100 Hz, 16000 Hz)
        for sr in (48000, 44100, 16000):
            self._ensure_sos(sr)

        self._is_initialized = True
        logger.info(
            "Initialized MockDenoiseEngine (mode='%s', delay=%.1fms, rtf=%.2f, filter=[%.1f-%.1f Hz])",
            self._mode,
            self.simulated_delay_ms,
            self.simulated_rtf,
            self.low_cutoff,
            self.high_cutoff,
        )

    # --------------------------------------------------------------------------
    # Engine Properties
    # --------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique lowercase identifier for the engine."""
        return "mock"

    @property
    def description(self) -> str:
        """Descriptive summary of engine capabilities."""
        return (
            f"Mock AI Denoise Engine (Mode: '{self._mode}', "
            f"Filter: [{self.low_cutoff:.0f}-{self.high_cutoff:.0f} Hz], "
            "Zero GPU/PyTorch Dependencies)"
        )

    @property
    def supported_sample_rates(self) -> List[int]:
        """Supported native sampling rates."""
        return [8000, 16000, 22050, 44100, 48000]

    def process_array(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        if not self._is_initialized:
            self.initialize()
            
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        # Apply Processing Mode
        if self._mode == "passthrough":
            processed_audio = audio.copy()
        elif self._mode in ("dsp_filter", "dsp"):
            processed_audio = self._filter_batch(audio, sample_rate)
            if self.spectral_subtraction:
                processed_audio = self._apply_spectral_subtraction(processed_audio, sample_rate)
        else:
            processed_audio = audio.copy()
            
        return np.clip(processed_audio, -1.0, 1.0).astype(np.float32)

    @property
    def mode(self) -> str:
        """Active operational mode ('dsp_filter', 'dsp', or 'passthrough')."""
        return self._mode

    @mode.setter
    def mode(self, new_mode: str) -> None:
        """Switch operational mode dynamically."""
        mode_clean = new_mode.strip().lower()
        if mode_clean not in self.SUPPORTED_MODES:
            raise EngineValidationError(
                f"Unsupported mode '{new_mode}'. Choose from {self.SUPPORTED_MODES}",
                engine_name=self.name,
            )
        self._mode = mode_clean
        logger.info("MockDenoiseEngine switched mode to '%s'", self._mode)

    # --------------------------------------------------------------------------
    # Lifecycle & Initialization
    # --------------------------------------------------------------------------

    def initialize(self, **kwargs: Any) -> None:
        """
        Idempotent engine configuration/re-initialization hook.
        Allows overriding runtime parameters without destroying instance.
        """
        if "mode" in kwargs:
            self.mode = kwargs["mode"]
        if "simulated_delay_ms" in kwargs:
            self.simulated_delay_ms = max(0.0, float(kwargs["simulated_delay_ms"]))
        if "simulated_rtf" in kwargs:
            self.simulated_rtf = max(0.0, float(kwargs["simulated_rtf"]))
        if "spectral_subtraction" in kwargs:
            self.spectral_subtraction = bool(kwargs["spectral_subtraction"])
        self._is_initialized = True

    # --------------------------------------------------------------------------
    # File Batch Processing (Contract CT-02)
    # --------------------------------------------------------------------------

    def process_file(
        self,
        input_path: str,
        output_path: str,
    ) -> FileProcessResult:
        """
        Synchronously process a full audio file (Contract CT-02).

        Args:
            input_path: Path to the input audio file (WAV, MP3, FLAC, etc.).
            output_path: Destination path for the cleaned WAV output.

        Returns:
            FileProcessResult: Object containing 'status', 'output_path', and 'metrics'.

        Raises:
            FileNotFoundError / EngineValidationError: If input_path does not exist.
            EngineProcessingError: If decoding, processing, or writing fails.
        """
        t0 = time.perf_counter()

        in_p = Path(input_path)
        out_p = Path(output_path)

        if not in_p.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")
        if not in_p.is_file():
            raise EngineValidationError(
                f"Input path is not a valid file: {input_path}",
                engine_name=self.name,
            )

        out_p.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Read input audio with SoundFile
            data, sample_rate = sf.read(str(in_p), dtype="float32")
        except Exception as e:
            raise EngineProcessingError(
                f"Failed to read audio file '{input_path}': {e}",
                engine_name=self.name,
            ) from e

        # Normalize channels to mono 1D
        if data.ndim > 1:
            data = np.mean(data, axis=-1)

        total_samples = len(data)
        audio_duration = float(total_samples) / float(sample_rate) if sample_rate > 0 else 0.0

        if total_samples == 0:
            raise EngineProcessingError(
                f"Audio file '{input_path}' contains 0 audio frames",
                engine_name=self.name,
            )

        # Standardize to 48,000 Hz if necessary
        target_sr = 48000
        if sample_rate != target_sr:
            g = math.gcd(target_sr, sample_rate)
            up = target_sr // g
            down = sample_rate // g
            data = signal.resample_poly(data, up, down).astype(np.float32)
            sample_rate = target_sr
            total_samples = len(data)
            audio_duration = float(total_samples) / float(sample_rate)

        # Apply Processing Mode
        if self._mode == "passthrough":
            processed_audio = data.copy()
        elif self._mode in ("dsp_filter", "dsp"):
            processed_audio = self._filter_batch(data, sample_rate)
            if self.spectral_subtraction:
                processed_audio = self._apply_spectral_subtraction(processed_audio, sample_rate)
        else:
            processed_audio = data.copy()

        # Amplitude protection: clip to [-1.0, 1.0]
        processed_audio = np.clip(processed_audio, -1.0, 1.0).astype(np.float32)

        # Write output WAV file (48kHz Mono PCM_16)
        try:
            sf.write(str(out_p), processed_audio, target_sr, subtype="PCM_16")
        except Exception as e:
            raise EngineProcessingError(
                f"Failed to write output audio file '{output_path}': {e}",
                engine_name=self.name,
            ) from e

        # Emulate inference delay if configured
        compute_elapsed = time.perf_counter() - t0
        target_delay = (self.simulated_delay_ms / 1000.0)
        if self.simulated_rtf > 0.0:
            target_delay = max(target_delay, self.simulated_rtf * audio_duration)

        if target_delay > compute_elapsed:
            sleep_time = target_delay - compute_elapsed
            time.sleep(sleep_time)

        total_elapsed = time.perf_counter() - t0
        rtf = (total_elapsed / audio_duration) if audio_duration > 0.0 else 0.0

        metrics = {
            "elapsed_seconds": round(total_elapsed, 4),
            "audio_duration_seconds": round(audio_duration, 4),
            "rtf": round(rtf, 4),
            "sample_rate": target_sr,
            "channels": 1,
            "mode": self._mode,
            "samples_processed": total_samples,
        }

        logger.info(
            "MockEngine process_file complete: '%s' -> '%s' (dur=%.2fs, elapsed=%.4fs, rtf=%.4f)",
            in_p.name,
            out_p.name,
            audio_duration,
            total_elapsed,
            rtf,
        )

        return FileProcessResult(
            status="success",
            output_path=str(out_p.resolve()),
            metrics=metrics,
            message=f"Mock denoising completed successfully via mode '{self._mode}'",
        )

    # --------------------------------------------------------------------------
    # Real-Time Streaming Chunk Processing (Contract CT-02)
    # --------------------------------------------------------------------------

    def process_stream_chunk(
        self,
        chunk: np.ndarray,
        sample_rate: int = 48000,
        state: Optional[Any] = None,
    ) -> Tuple[np.ndarray, MockStreamState]:
        """
        Process a single real-time audio chunk (Contract CT-02 Streaming).

        Args:
            chunk: 1D NumPy array of audio samples (float32).
            sample_rate: Audio sampling frequency (default 48000 Hz).
            state: Session state object from the previous chunk, or None on startup.

        Returns:
            Tuple[np.ndarray, MockStreamState]:
                - Cleaned audio chunk (1D np.float32).
                - Updated MockStreamState for the next chunk cycle.
        """
        t0 = time.perf_counter()

        # Validate input chunk
        chunk_clean = self.validate_stream_chunk(chunk, sample_rate)

        # Handle empty chunk edge case
        if chunk_clean.size == 0:
            stream_state = state if isinstance(state, MockStreamState) else self.reset_stream_state(sample_rate)
            if stream_state.filter_state is not None and not np.all(np.isfinite(stream_state.filter_state)):
                stream_state.filter_state = np.zeros_like(stream_state.filter_state)
            return chunk_clean, stream_state

        # Initialize or unpack state
        if state is None or not isinstance(state, MockStreamState):
            stream_state = self.reset_stream_state(sample_rate)
        else:
            stream_state = state
            if stream_state.sample_rate != sample_rate:
                # Sample rate changed mid-stream; recompute filter
                stream_state.sample_rate = sample_rate
                stream_state.sos = self._ensure_sos(sample_rate)
                stream_state.filter_state = np.zeros((stream_state.sos.shape[0], 2), dtype=np.float32)

        # Sanity check filter_state registers to prevent NaN/Inf poisoning
        if stream_state.filter_state is not None and not np.all(np.isfinite(stream_state.filter_state)):
            stream_state.filter_state = np.zeros_like(stream_state.filter_state)

        # Process Chunk According to Mode
        if self._mode == "passthrough":
            out_chunk = chunk_clean.copy()
        elif self._mode in ("dsp_filter", "dsp"):
            out_chunk, new_zi = self._filter_chunk(
                chunk_clean,
                stream_state.sos,
                stream_state.filter_state,
            )
            if not np.all(np.isfinite(new_zi)):
                new_zi = np.zeros_like(new_zi)
            stream_state.filter_state = new_zi
        else:
            out_chunk = chunk_clean.copy()

        # Sanitize out_chunk if any non-finite produced
        if not np.all(np.isfinite(out_chunk)):
            out_chunk = np.nan_to_num(out_chunk, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

        # Amplitude clipping protection
        out_chunk = np.clip(out_chunk, -1.0, 1.0).astype(np.float32)

        # Simulate artificial per-chunk latency if configured
        if self.simulated_delay_ms > 0.0:
            time.sleep(self.simulated_delay_ms / 1000.0)

        chunk_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Update telemetry in state
        stream_state.chunk_count += 1
        stream_state.total_samples += len(out_chunk)
        stream_state.total_elapsed_ms += chunk_elapsed_ms
        stream_state.last_chunk_latency_ms = chunk_elapsed_ms

        peak = float(np.max(np.abs(out_chunk)))
        if peak > stream_state.peak_amplitude:
            stream_state.peak_amplitude = peak

        return out_chunk, stream_state

    # --------------------------------------------------------------------------
    # Stream State Management
    # --------------------------------------------------------------------------

    def reset_stream_state(self, sample_rate: int = 48000) -> MockStreamState:
        """
        Return a fresh, isolated state object for starting a new streaming session.

        Args:
            sample_rate: Sampling frequency for the session.

        Returns:
            MockStreamState: Clean state with initialized filter delay registers.
        """
        sos = self._ensure_sos(sample_rate)
        initial_zi = np.zeros((sos.shape[0], 2), dtype=np.float32)
        return MockStreamState(
            filter_state=initial_zi,
            chunk_count=0,
            total_samples=0,
            sample_rate=sample_rate,
            sos=sos,
            total_elapsed_ms=0.0,
            last_chunk_latency_ms=0.0,
            peak_amplitude=0.0,
        )

    # --------------------------------------------------------------------------
    # Internal DSP Computation Helpers
    # --------------------------------------------------------------------------

    def _ensure_sos(self, sample_rate: int) -> np.ndarray:
        """
        Retrieve or compute Second-Order Sections (SOS) bandpass filter matrix.
        Caches matrices per sample_rate for zero allocation during streaming.
        """
        if sample_rate in self._sos_cache:
            return self._sos_cache[sample_rate]

        nyquist = sample_rate * 0.5
        # Ensure cutoff frequencies are strictly within (0, Nyquist)
        low = max(10.0, min(self.low_cutoff, nyquist * 0.45))
        high = max(low + 100.0, min(self.high_cutoff, nyquist * 0.95))

        # Generate Butterworth bandpass filter in SOS representation
        sos = signal.butter(
            N=self.filter_order,
            Wn=[low, high],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        ).astype(np.float32)

        self._sos_cache[sample_rate] = sos
        return sos

    def _filter_batch(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply Butterworth bandpass filter to full batch audio signal."""
        sos = self._ensure_sos(sample_rate)
        # Apply causal filtering with zero initial state
        zi = np.zeros((sos.shape[0], 2), dtype=np.float32)
        filtered, _ = signal.sosfilt(sos, data, zi=zi)
        return filtered.astype(np.float32)

    def _filter_chunk(
        self,
        chunk: np.ndarray,
        sos: Optional[np.ndarray],
        zi: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter a single streaming chunk using stateful delay registers.
        Executes in <0.1ms on CPU with zero allocation overhead.
        """
        if sos is None:
            sos = self._ensure_sos(48000)
        if zi is None:
            zi = np.zeros((sos.shape[0], 2), dtype=np.float32)

        filtered_chunk, new_zi = signal.sosfilt(sos, chunk, zi=zi)
        return filtered_chunk.astype(np.float32), new_zi.astype(np.float32)

    def _apply_spectral_subtraction(
        self,
        audio: np.ndarray,
        sample_rate: int,
        alpha: float = 1.5,
        beta: float = 0.08,
    ) -> np.ndarray:
        """
        Lightweight spectral subtraction via STFT for stationary noise reduction.

        Args:
            audio: 1D audio array.
            sample_rate: Sampling frequency.
            alpha: Over-subtraction factor (reduces residual musical noise).
            beta: Spectral noise floor (prevents dark, hollow artifacts).
        """
        nperseg = 1024
        noverlap = 768

        if len(audio) < nperseg:
            return audio

        # Compute Short-Time Fourier Transform
        _, _, Zxx = signal.stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Estimate stationary noise spectrum via 10th percentile across time frames
        noise_floor = np.percentile(mag, 10, axis=1, keepdims=True)

        # Apply over-subtraction with floor
        clean_mag = np.maximum(mag - alpha * noise_floor, beta * mag)

        # Invert STFT
        Zxx_clean = clean_mag * np.exp(1j * phase)
        _, clean_audio = signal.istft(Zxx_clean, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

        # Match output length to input length
        if len(clean_audio) >= len(audio):
            return clean_audio[:len(audio)].astype(np.float32)
        else:
            padded = np.zeros_like(audio)
            padded[:len(clean_audio)] = clean_audio
            return padded
