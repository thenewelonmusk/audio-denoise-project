"""
Voice Denoising Engine Abstraction Layer and Reference Implementation.
"""

from typing import Any
from src.engine_interface import (
    AudioDenoiseEngine,
    EngineError,
    EngineInitializationError,
    EngineMetadata,
    EngineNotFoundError,
    EngineProcessingError,
    EngineRegistry,
    EngineValidationError,
    FileProcessResult,
    StreamChunkResult,
    get_engine,
    list_available_engines,
    register_engine,
)

def __getattr__(name: str) -> Any:
    # Lazily import reference mock engine from models/ to keep src/ decoupled
    if name in {"MockDenoiseEngine", "MockStreamState"}:
        import models.mock_engine
        return getattr(models.mock_engine, name)

    # Lazily import audio_io which depends on matplotlib/scipy/soundfile
    if name in {
        "AudioValidationError",
        "chunk_audio",
        "downmix_to_mono",
        "generate_spectrogram",
        "inspect_and_prepare_audio",
        "reconstruct_audio_ola",
        "resample_audio",
        "save_audio_wav",
        "summarize_timing",
    }:
        import src.audio_io
        return getattr(src.audio_io, name)
    
    # Lazily import theme which depends on gradio
    if name in {
        "AudioWorkstationTheme",
        "DAW_CUSTOM_CSS",
        "get_daw_theme",
    }:
        import src.theme
        return getattr(src.theme, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AudioDenoiseEngine",
    "EngineRegistry",
    "get_engine",
    "list_available_engines",
    "register_engine",
    "FileProcessResult",
    "StreamChunkResult",
    "EngineMetadata",
    "EngineError",
    "EngineNotFoundError",
    "EngineInitializationError",
    "EngineProcessingError",
    "EngineValidationError",
    "MockDenoiseEngine",
    "MockStreamState",
    "AudioValidationError",
    "chunk_audio",
    "downmix_to_mono",
    "generate_spectrogram",
    "inspect_and_prepare_audio",
    "reconstruct_audio_ola",
    "resample_audio",
    "save_audio_wav",
    "summarize_timing",
    "AudioWorkstationTheme",
    "DAW_CUSTOM_CSS",
    "get_daw_theme",
]
