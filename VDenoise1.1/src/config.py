"""
src/config.py - Centralized System Configuration & Architectural Constants.

Defines unified parameters for audio standardization, 60-second chunking,
Overlap-Add (OLA) windowing, performance targets, and directory paths.
"""

from __future__ import annotations

import os
from pathlib import Path

# ==============================================================================
# 1. DIRECTORY PATHS
# ==============================================================================
# VDenoise1.1/
APP_ROOT: Path = Path(__file__).resolve().parent.parent
# audio-denoise-project-app_dfn_dev/
REPO_ROOT: Path = APP_ROOT.parent

DEFAULT_OUTPUT_DIR: Path = APP_ROOT / "outputs"
DEFAULT_RESULTS_DIR: Path = REPO_ROOT / "results"
DEFAULT_BENCHMARK_CSV: Path = DEFAULT_RESULTS_DIR / "benchmark.csv"
FINETUNED_MODEL_DIR: Path = REPO_ROOT / "ml" / "final_models" / "dfn3_ft_vivos_v1"

# ==============================================================================
# 2. AUDIO & CHUNKING PARAMETERS (CT-01 & CT-04)
# ==============================================================================
TARGET_SAMPLE_RATE: int = 48000
TARGET_CHANNELS: int = 1
CHUNK_MAX_SECONDS: float = 60.0
OVERLAP_SECONDS: float = 0.5

SUPPORTED_AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".wma",
)

# ==============================================================================
# 3. ENGINE & PERFORMANCE BENCHMARK THRESHOLDS
# ==============================================================================
DEFAULT_ENGINE_ID: str = os.environ.get("DENOISE_ENGINE", "dfn3_ft_vivos_v1")
FALLBACK_ENGINE_ID: str = "dfnet2"

# Target thresholds committed in technical specification
TARGET_MAX_RTF: float = 0.30
TARGET_MAX_RAM_MB: float = 1228.8  # 1.2 GB