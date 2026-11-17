"""
src/benchmark.py - Performance Telemetry, RTF & Peak RAM Benchmarking (Contract CT-03).

Measures wall-clock execution latency, Real-Time Factor (RTF), and peak process
Resident Set Size (RSS) memory usage during audio denoising pipelines.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

from src.config import (
    DEFAULT_BENCHMARK_CSV,
    TARGET_MAX_RAM_MB,
    TARGET_MAX_RTF,
)


def get_process_rss_mb() -> float:
    """Return current process Resident Set Size (RSS) memory in Megabytes (MB)."""
    if psutil is None:
        return 0.0
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


def summarize_timing(elapsed_seconds: float, audio_duration_seconds: float) -> Dict[str, float]:
    """
    Contract CT-03: Compute performance telemetry and Real-Time Factor (RTF).

    Parameters:
        elapsed_seconds: Processing wall-clock time in seconds.
        audio_duration_seconds: Duration of the processed audio in seconds.

    Returns:
        {
            "elapsed_seconds": float,
            "audio_duration_seconds": float,
            "rtf": float,
        }
    """
    if elapsed_seconds < 0.0 or audio_duration_seconds < 0.0:
        raise ValueError(
            f"Elapsed time and audio duration must be non-negative "
            f"(got elapsed={elapsed_seconds}, duration={audio_duration_seconds})"
        )

    rtf = (elapsed_seconds / audio_duration_seconds) if audio_duration_seconds > 0.0 else 0.0
    return {
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "audio_duration_seconds": round(float(audio_duration_seconds), 3),
        "rtf": round(float(rtf), 3),
    }


class BenchmarkSession:
    """
    Context manager that tracks wall-clock latency and polls peak RSS memory
    at 50ms intervals to capture true memory spikes during chunk inference.
    """

    def __init__(self, poll_interval_sec: float = 0.05) -> None:
        self.poll_interval_sec = poll_interval_sec
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.baseline_ram_mb: float = 0.0
        self.peak_ram_mb: float = 0.0
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    def _poll_memory(self) -> None:
        while not self._stop_event.is_set():
            current_mb = get_process_rss_mb()
            if current_mb > self.peak_ram_mb:
                self.peak_ram_mb = current_mb
            self._stop_event.wait(self.poll_interval_sec)

    def __enter__(self) -> BenchmarkSession:
        self.baseline_ram_mb = get_process_rss_mb()
        self.peak_ram_mb = self.baseline_ram_mb
        self.start_time = time.perf_counter()

        if psutil is not None:
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._poll_memory,
                name="ram-benchmark-poller",
                daemon=True,
            )
            self._monitor_thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.perf_counter()
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=0.5)
        final_mb = get_process_rss_mb()
        if final_mb > self.peak_ram_mb:
            self.peak_ram_mb = final_mb

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time if self.end_time > 0.0 else time.perf_counter()
        return max(0.0, end - self.start_time)

    def build_report(self, audio_duration_seconds: float) -> Dict[str, Any]:
        """Combine CT-03 timing stats with Peak RAM telemetry."""
        timing = summarize_timing(self.elapsed_seconds, audio_duration_seconds)
        peak_mb = round(self.peak_ram_mb, 2)
        delta_mb = round(max(0.0, self.peak_ram_mb - self.baseline_ram_mb), 2)
        return {
            **timing,
            "baseline_ram_mb": round(self.baseline_ram_mb, 2),
            "peak_ram_mb": peak_mb,
            "ram_delta_mb": delta_mb,
            "within_rtf_target": timing["rtf"] < TARGET_MAX_RTF,
            "within_ram_target": peak_mb < TARGET_MAX_RAM_MB,
        }


def append_benchmark_csv(
    record: Dict[str, Any],
    csv_path: Optional[Path] = None,
) -> Path:
    """Append a single benchmark run row to results/benchmark.csv."""
    target_csv = Path(csv_path) if csv_path is not None else DEFAULT_BENCHMARK_CSV
    target_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "input_file",
        "engine_name",
        "audio_duration_seconds",
        "num_chunks",
        "elapsed_seconds",
        "rtf",
        "peak_ram_mb",
        "ram_delta_mb",
        "within_rtf_target",
        "within_ram_target",
    ]
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **{k: record.get(k, "") for k in fieldnames if k != "timestamp"},
    }

    write_header = not target_csv.exists() or target_csv.stat().st_size == 0
    with target_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return target_csv