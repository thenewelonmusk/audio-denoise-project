"""
src/engine_interface.py - Standard Audio Denoising Engine Interface & Registry.

This module establishes the Layer 1 abstraction boundary for the Audio Denoising
system. It defines:
1. Standard exception hierarchy (EngineError and specialized subclasses).
2. Data Transfer Objects (FileProcessResult, StreamChunkResult, EngineMetadata).
3. AudioDenoiseEngine Abstract Base Class (Contract CT-02).
4. Thread-safe EngineRegistry for plug-and-play factory instantiation.
5. 1-line configuration switcher via environment variable DENOISE_ENGINE.

Strict architectural rule: This module MUST NOT import PyTorch, ONNX, or any
deep learning runtime libraries.
"""

from __future__ import annotations

import abc
import ast
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type

import numpy as np

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. EXCEPTION HIERARCHY
# ==============================================================================

class EngineError(Exception):
    """Base exception for all audio denoise engine errors."""
    def __init__(self, message: str, engine_name: Optional[str] = None):
        self.engine_name = engine_name
        prefix = f"[{engine_name}] " if engine_name else ""
        super().__init__(f"{prefix}{message}")


class EngineNotFoundError(EngineError, KeyError):
    """Raised when an unregistered engine identifier is requested from EngineRegistry."""
    def __init__(self, engine_name: str, available_engines: Optional[List[str]] = None):
        avail_str = f" Available engines: {sorted(available_engines)}" if available_engines else ""
        super().__init__(
            f"Engine '{engine_name}' not found in registry.{avail_str}",
            engine_name=engine_name,
        )


class EngineInitializationError(EngineError):
    """Raised when an engine fails during initialize() or is missing dependencies."""
    def __init__(self, message: str, engine_name: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(message, engine_name=engine_name)
        self.cause = cause


class EngineProcessingError(EngineError):
    """Raised when an error occurs during file or stream chunk processing."""
    def __init__(self, message: str, engine_name: Optional[str] = None, is_streaming: bool = False):
        super().__init__(message, engine_name=engine_name)
        self.is_streaming = is_streaming


class EngineValidationError(EngineError, ValueError):
    """Raised when input parameters (audio array, sample rate, paths) violate preconditions."""
    def __init__(self, message: str, engine_name: Optional[str] = None):
        super().__init__(message, engine_name=engine_name)


# ==============================================================================
# 2. DATA TRANSFER OBJECTS (DTOs)
# ==============================================================================

@dataclass(frozen=True)
class FileProcessResult(Mapping):
    """
    Immutable result DTO returned by AudioDenoiseEngine.process_file().

    Implements the Mapping protocol for backward-compatibility with Contract CT-02
    dict consumers:
        result['status'], result['output_path'], result['metrics']
    while also supporting clean attribute access:
        result.status, result.output_path, result.metrics
    """
    status: str = "success"
    output_path: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert DTO into a standard serializable dictionary."""
        return {
            "status": self.status,
            "output_path": self.output_path,
            "metrics": dict(self.metrics),
            "message": self.message,
        }

    # Mapping protocol implementation
    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key not in d:
            raise KeyError(f"Key '{key}' not in FileProcessResult")
        return d[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class StreamChunkResult:
    """
    Result DTO for streaming chunk inference.
    """
    chunk: np.ndarray
    sample_rate: int
    state: Any
    latency_ms: float = 0.0

    def to_tuple(self) -> Tuple[np.ndarray, Any]:
        """Return the standard (cleaned_chunk, new_state) tuple."""
        return self.chunk, self.state


@dataclass(frozen=True)
class EngineMetadata:
    """
    Descriptive metadata for registered engines.
    """
    name: str
    display_name: str
    version: str
    description: str
    supported_sample_rates: List[int] = field(default_factory=lambda: [48000])
    requires_gpu: bool = False
    is_available: bool = True
    unavailable_reason: Optional[str] = None


# ==============================================================================
# 3. ABSTRACT BASE CLASS: AudioDenoiseEngine
# ==============================================================================

class AudioDenoiseEngine(abc.ABC):
    """
    Abstract Base Class defining the canonical contract for all Audio Denoising Engines.

    All engine implementations (Mock, DeepFilterNet2, RNNoise, Demucs) must subclass
    AudioDenoiseEngine and implement process_file and process_stream_chunk.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._is_initialized: bool = False
        self._init_kwargs: Dict[str, Any] = kwargs

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Unique lowercase identifier for the engine (e.g. 'mock', 'dfnet2').
        """
        pass

    @property
    def description(self) -> str:
        """Human-readable description of the engine."""
        return self.__doc__ or f"{self.name} audio denoise engine"

    @property
    def supported_sample_rates(self) -> List[int]:
        """List of sample rates supported natively by the engine. Default is [48000]."""
        return [48000]

    @property
    def is_initialized(self) -> bool:
        """Indicates whether initialize() has completed successfully."""
        return self._is_initialized

    def initialize(self, **kwargs: Any) -> None:
        """
        Initialize the engine (allocate buffers, warm up models, verify dependencies).
        This method is idempotent: calling it multiple times has no side effects.
        """
        self._is_initialized = True

    @abc.abstractmethod
    def process_file(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Process a full audio file synchronously (Contract CT-02).

        Args:
            input_path: Path to the input audio file (WAV 48kHz Mono).
            output_path: Target path for the enhanced WAV output.

        Returns:
            Dict or FileProcessResult containing 'status', 'output_path', and 'metrics'.

        Raises:
            EngineValidationError: If input_path does not exist or paths are invalid.
            EngineProcessingError: If processing or file I/O fails.
        """
        pass

    def process_array(
        self, audio: np.ndarray, sample_rate: int = 48000
    ) -> np.ndarray:
        """
        Process a full audio array synchronously in memory.

        Default implementation processes via temporary file.
        Subclasses should override for direct RAM processing.
        """
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            sf.write(tmp_in.name, audio, sample_rate)
            tmp_in_path = tmp_in.name
            
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            tmp_out_path = tmp_out.name
            
        try:
            self.process_file(tmp_in_path, tmp_out_path)
            data, _ = sf.read(tmp_out_path)
            return data.astype(np.float32)
        finally:
            if os.path.exists(tmp_in_path):
                os.remove(tmp_in_path)
            if os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)

    @abc.abstractmethod
    def process_stream_chunk(
        self,
        chunk: np.ndarray,
        sample_rate: int,
        state: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Any]:
        """
        Process a single real-time audio chunk (Contract CT-02 Streaming).

        Args:
            chunk: 1D NumPy array of audio samples (float32).
            sample_rate: Audio sampling rate (e.g. 48000).
            state: Filter state / memory from previous chunk (None for the first chunk).

        Returns:
            Tuple[np.ndarray, Any]:
                - Cleaned audio chunk (1D float32 array).
                - Updated state object for the next streaming chunk.

        Raises:
            EngineValidationError: If chunk shape, type, or sample_rate is unsupported.
            EngineProcessingError: If chunk inference fails.
        """
        pass

    def reset_stream_state(self) -> Any:
        """
        Return a fresh, empty state object for starting a new streaming session.
        Default implementation returns None.
        """
        return None

    def validate_stream_chunk(
        self,
        chunk: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Defensive helper method for streaming chunk validation and normalization.
        Converts multi-channel to mono and casts to float32 in [-1.0, 1.0].
        """
        if chunk is None or not isinstance(chunk, np.ndarray):
            raise EngineValidationError("Streaming chunk must be a non-null NumPy ndarray", engine_name=self.name)

        if sample_rate <= 0:
            raise EngineValidationError(
                f"Invalid sample rate: {sample_rate}. Must be positive.",
                engine_name=self.name,
            )

        if not np.issubdtype(chunk.dtype, np.number):
            raise EngineValidationError(
                f"Unsupported chunk dtype '{chunk.dtype}'. Audio chunk must be a numeric array.",
                engine_name=self.name,
            )

        # Downmix multi-channel to mono if necessary before size check
        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=-1)

        if chunk.ndim == 0:
            chunk = chunk.reshape(-1)

        if chunk.size == 0:
            return chunk.astype(np.float32)

        # Convert integer formats to float32 [-1.0, 1.0]
        try:
            if np.issubdtype(chunk.dtype, np.integer):
                max_val = float(np.iinfo(chunk.dtype).max)
                chunk = (chunk.astype(np.float32) / max_val)
            elif chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)
        except (ValueError, TypeError) as exc:
            raise EngineValidationError(
                f"Failed to convert chunk dtype '{chunk.dtype}' to float32: {exc}",
                engine_name=self.name,
            ) from exc

        # Sanitize NaN/Inf to prevent filter state corruption
        if not np.all(np.isfinite(chunk)):
            chunk = np.nan_to_num(chunk, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

        # Check sample rate compatibility
        if sample_rate not in self.supported_sample_rates:
            logger.warning(
                "[%s] Sample rate %d Hz not in supported rates %s; performance may degrade.",
                self.name, sample_rate, self.supported_sample_rates
            )

        return chunk

    def __enter__(self) -> AudioDenoiseEngine:
        if not self.is_initialized:
            self.initialize(**self._init_kwargs)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


# ==============================================================================
# 4. ENGINE REGISTRY & FACTORY
# ==============================================================================

class EngineRegistry:
    """
    Thread-safe registry and factory for AudioDenoiseEngine implementations.

    Supports:
    - Decorator-based class registration: @EngineRegistry.register('mock')
    - Lazy-loading registration via module import path
    - Instance caching for low-latency reuse across UI events
    - 1-line configuration switching via environment variable DENOISE_ENGINE
    """

    _lock: threading.RLock = threading.RLock()
    _registry: Dict[str, Type[AudioDenoiseEngine]] = {}
    _lazy_registry: Dict[str, str] = {}  # key -> 'module.submodule.ClassName'
    _aliases: Dict[str, str] = {}        # alias -> canonical_key
    _instances: Dict[Tuple[str, frozenset[Tuple[str, Any]]], AudioDenoiseEngine] = {}
    _metadata_registry: Dict[str, Dict[str, Any]] = {}
    _source_files: Dict[str, Path] = {}

    IGNORED_MODEL_FILES = {
        "__init__.py",
    }

    _DEFAULT_SEED_MODELS: Dict[str, Dict[str, Any]] = {
        "dfnet_engine.py": {
            "engine_id": "dfnet2",
            "class_name": "DeepFilterNetEngine",
            "display_name": "DeepFilterNet2 (DFNET2)",
            "aliases": ["deepfilter", "df2"],
            "type": "Mạng Nơ-ron Lọc Âm Thanh Phổ Phức (Full-Band 48kHz)",
            "sample_rate": "48 kHz (Khuyên dùng)",
            "status": "Hoạt động (Active / Default)",
            "description": "Mô hình học sâu lọc sâu 48kHz trên miền tần số phức STFT & ERB. Khử sạch tạp âm đời sống, bảo toàn formant và ngữ điệu giọng nói.",
        },
        "mock_engine.py": {
            "engine_id": "mock",
            "class_name": "MockDenoiseEngine",
            "display_name": "Mock DSP Engine",
            "aliases": ["passthrough", "mock_dsp"],
            "type": "Xử lý Tín hiệu Số Cổ điển (Butterworth Bandpass)",
            "sample_rate": "48 kHz / Đa tần số",
            "status": "Hoạt động (Active / CPU)",
            "description": "Bộ lọc dải thông Butterworth 100Hz - 7.5kHz chạy trực tiếp trên CPU, siêu nhẹ (< 0.1ms/chunk), dùng kiểm thử đường ống không cần GPU.",
        },
    }

    @classmethod
    def register(
        cls,
        name: str,
        aliases: Optional[List[str]] = None,
        force: bool = False,
    ) -> Callable[[Type[AudioDenoiseEngine]], Type[AudioDenoiseEngine]]:
        """
        Decorator to register an AudioDenoiseEngine subclass.
        """
        def decorator(subclass: Type[AudioDenoiseEngine]) -> Type[AudioDenoiseEngine]:
            cls.register_class(name, subclass, aliases=aliases, force=force)
            return subclass
        return decorator

    @classmethod
    def register_class(
        cls,
        name: str,
        engine_cls: Type[AudioDenoiseEngine],
        aliases: Optional[List[str]] = None,
        force: bool = False,
    ) -> None:
        """Register an engine class directly under a canonical name."""
        with cls._lock:
            canonical = name.strip().lower()
            if not issubclass(engine_cls, AudioDenoiseEngine):
                raise TypeError(f"Class '{engine_cls.__name__}' must inherit from AudioDenoiseEngine")
            if canonical in cls._registry and not force:
                logger.debug("Engine '%s' already registered, skipping.", canonical)
                return

            if aliases:
                for alias in aliases:
                    alias_clean = alias.strip().lower()
                    if (alias_clean in cls._registry or alias_clean in cls._lazy_registry) and not force:
                        raise ValueError(f"Alias '{alias_clean}' collides with canonical engine name")

            cls._registry[canonical] = engine_cls
            cls._lazy_registry.pop(canonical, None)

            if aliases:
                for alias in aliases:
                    alias_clean = alias.strip().lower()
                    cls._aliases[alias_clean] = canonical

            logger.info("Registered engine '%s' (%s)", canonical, engine_cls.__name__)

    @classmethod
    def register_lazy(
        cls,
        name: str,
        import_path: str,
        aliases: Optional[List[str]] = None,
        force: bool = False,
    ) -> None:
        """
        Register an engine by its module import path without importing it immediately.
        Enables zero startup import overhead for heavy AI engines.
        """
        with cls._lock:
            canonical = name.strip().lower()
            if canonical in cls._registry and not force:
                return

            if aliases:
                for alias in aliases:
                    alias_clean = alias.strip().lower()
                    if (alias_clean in cls._registry or alias_clean in cls._lazy_registry) and not force:
                        raise ValueError(f"Alias '{alias_clean}' collides with canonical engine name")

            cls._lazy_registry[canonical] = import_path
            if aliases:
                for alias in aliases:
                    alias_clean = alias.strip().lower()
                    cls._aliases[alias_clean] = canonical

            logger.debug("Lazily registered engine '%s' -> '%s'", canonical, import_path)

    @classmethod
    def _resolve_canonical_name(cls, name: str) -> str:
        """Resolve an engine name or alias to its canonical key."""
        clean = name.strip().lower()
        return cls._aliases.get(clean, clean)

    @classmethod
    def _load_lazy_engine(cls, canonical: str) -> Type[AudioDenoiseEngine]:
        """Dynamically import and register a lazy engine."""
        import_path = cls._lazy_registry[canonical]
        module_path, class_name = import_path.rsplit(".", 1)
        try:
            try:
                mod = importlib.import_module(module_path)
            except ModuleNotFoundError:
                src_file = cls._source_files.get(canonical)
                if src_file is not None and src_file.exists():
                    spec = importlib.util.spec_from_file_location(module_path, str(src_file))
                    if spec is None or spec.loader is None:
                        raise ImportError(f"Cannot create module spec from '{src_file}'")
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_path] = mod
                    spec.loader.exec_module(mod)
                else:
                    raise
            engine_cls = getattr(mod, class_name)
            if not issubclass(engine_cls, AudioDenoiseEngine):
                raise TypeError(f"Target '{import_path}' does not inherit from AudioDenoiseEngine")
            cls._registry[canonical] = engine_cls
            return engine_cls
        except Exception as exc:
            raise EngineInitializationError(
                f"Failed to load lazy engine '{canonical}' from '{import_path}': {exc}",
                engine_name=canonical,
                cause=exc,
            ) from exc

    @classmethod
    def get_engine_class(cls, name: str) -> Type[AudioDenoiseEngine]:
        """
        Retrieve the engine class for the given name or alias.
        """
        with cls._lock:
            if not name or not name.strip():
                raise EngineNotFoundError(name or "", available_engines=cls.list_available_engines())

            canonical = cls._resolve_canonical_name(name)
            if canonical in cls._registry:
                return cls._registry[canonical]
            if canonical in cls._lazy_registry:
                return cls._load_lazy_engine(canonical)

            raise EngineNotFoundError(name, available_engines=cls.list_available_engines())

    @classmethod
    def get_engine(
        cls,
        name: Optional[str] = None,
        cache_instance: bool = True,
        **kwargs: Any,
    ) -> AudioDenoiseEngine:
        """
        Factory function returning an initialized AudioDenoiseEngine instance.

        If name is None, resolves the engine name in priority order:
        1. Environment variable DENOISE_ENGINE
        2. Fallback default: 'mock'
        """
        with cls._lock:
            if name is None:
                name = os.environ.get("DENOISE_ENGINE", "mock")

            if not name or not name.strip():
                raise EngineNotFoundError(name or "", available_engines=cls.list_available_engines())

            canonical = cls._resolve_canonical_name(name)
            cache_key = (canonical, frozenset(kwargs.items()))

            if cache_instance and cache_key in cls._instances:
                inst = cls._instances[cache_key]
                if not inst.is_initialized:
                    inst.initialize()
                return inst

            engine_cls = cls.get_engine_class(canonical)
            instance = engine_cls(**kwargs)

        # Initialize instance outside of the registry lock to prevent deadlocks
        try:
            instance.initialize()
        except Exception as exc:
            if isinstance(exc, EngineError):
                raise
            raise EngineInitializationError(
                f"Initialization of engine '{canonical}' failed: {exc}",
                engine_name=canonical,
                cause=exc,
            ) from exc

        if cache_instance:
            with cls._lock:
                cls._instances[cache_key] = instance

        return instance

    @classmethod
    def list_available_engines(cls) -> List[str]:
        """Return a sorted list of all registered canonical engine keys."""
        with cls._lock:
            all_keys = set(cls._registry.keys()).union(cls._lazy_registry.keys())
            return sorted(all_keys)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister an engine and clear its cached instance."""
        with cls._lock:
            canonical = cls._resolve_canonical_name(name)
            cls._registry.pop(canonical, None)
            cls._lazy_registry.pop(canonical, None)
            cls._metadata_registry.pop(canonical, None)
            cls._source_files.pop(canonical, None)
            cls._instances = {
                k: v for k, v in cls._instances.items() if k[0] != canonical
            }
            # Remove any aliases pointing to canonical
            cls._aliases = {k: v for k, v in cls._aliases.items() if v != canonical}

    @classmethod
    def clear(cls) -> None:
        """Reset all registries and instances (used primarily in test fixtures)."""
        with cls._lock:
            cls._registry.clear()
            cls._lazy_registry.clear()
            cls._aliases.clear()
            cls._instances.clear()
            cls._metadata_registry.clear()
            cls._source_files.clear()
            cls._register_defaults()

    @classmethod
    def _extract_engine_info_from_ast(
        cls,
        py_file: Path,
        tree: ast.Module,
    ) -> Optional[Dict[str, Any]]:
        """
        Inspect AST of a Python file in models/ to determine if it defines an AudioDenoiseEngine.
        Returns a metadata dictionary for the discovered engine class, or None if not an engine module.
        """
        module_doc = (ast.get_docstring(tree) or "").strip()

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue

            # 1. Check if class inherits from AudioDenoiseEngine
            inherits_engine = False
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "AudioDenoiseEngine":
                    inherits_engine = True
                elif isinstance(base, ast.Attribute) and base.attr == "AudioDenoiseEngine":
                    inherits_engine = True

            # 2. Check decorators for @EngineRegistry.register(...) or @register_engine(...)
            decorator_engine_id: Optional[str] = None
            decorator_aliases: List[str] = []
            has_register_decorator = False

            for dec in node.decorator_list:
                if isinstance(dec, ast.Call):
                    func = dec.func
                    is_reg = (
                        (isinstance(func, ast.Attribute) and func.attr == "register")
                        or (isinstance(func, ast.Name) and func.id == "register_engine")
                    )
                    if is_reg:
                        has_register_decorator = True
                        if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                            decorator_engine_id = dec.args[0].value.strip().lower()
                        for kw in dec.keywords:
                            if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                decorator_engine_id = kw.value.value.strip().lower()
                            elif kw.arg == "aliases" and isinstance(kw.value, (ast.List, ast.Tuple)):
                                for elt in kw.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        decorator_aliases.append(elt.value.strip().lower())

            if not inherits_engine and not has_register_decorator:
                continue

            # 3. Inspect class body for `name` property, `description` property, and class constants
            prop_name: Optional[str] = None
            prop_desc: Optional[str] = None
            class_attrs: Dict[str, str] = {}

            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                            class_attrs[target.id] = item.value.value.strip()
                elif isinstance(item, ast.FunctionDef):
                    if item.name == "name":
                        for stmt in item.body:
                            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                                prop_name = stmt.value.value.strip().lower()
                    elif item.name == "description":
                        for stmt in item.body:
                            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                                prop_desc = stmt.value.value.strip()

            stem = py_file.stem
            fallback_id = stem[:-7] if stem.endswith("_engine") and len(stem) > 7 else stem
            engine_id = (
                decorator_engine_id
                or prop_name
                or class_attrs.get("ENGINE_ID", "").lower()
                or fallback_id.lower()
            )

            class_doc = (ast.get_docstring(node) or "").strip()
            first_doc_line = (class_doc.splitlines()[0] if class_doc else "") or (
                module_doc.splitlines()[0] if module_doc else ""
            )
            description = (
                class_attrs.get("DESCRIPTION")
                or prop_desc
                or first_doc_line
                or f"Động cơ khử nhiễu âm thanh AI ({node.name})."
            )

            seed = cls._DEFAULT_SEED_MODELS.get(py_file.name, {})
            default_display_name = (
                class_attrs.get("DISPLAY_NAME")
                or seed.get("display_name")
                or f"{node.name} ({engine_id.upper()})"
            )

            return {
                "filename": py_file.name,
                "file_path": py_file,
                "engine_id": engine_id,
                "class_name": node.name,
                "aliases": decorator_aliases or list(seed.get("aliases", [])),
                "default_display_name": default_display_name,
                "type": class_attrs.get("ENGINE_TYPE") or seed.get("type", "AI Speech Enhancement Engine (48kHz)"),
                "sample_rate": class_attrs.get("SAMPLE_RATE_LABEL") or seed.get("sample_rate", "48 kHz"),
                "status": class_attrs.get("STATUS_LABEL") or seed.get("status", "Hoạt động (Active)"),
                "description": seed.get("description", description),
            }

        return None

    @classmethod
    def _scan_model_engine_files(cls, models_dir: Path) -> Dict[str, Dict[str, Any]]:
        """
        Scan all .py files in models_dir and return discovered engine definitions keyed by filename.
        """
        discovered: Dict[str, Dict[str, Any]] = {}
        if not models_dir.exists() or not models_dir.is_dir():
            return discovered

        for py_file in sorted(models_dir.glob("*.py")):
            if py_file.name in cls.IGNORED_MODEL_FILES or py_file.name.startswith("_"):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except Exception as exc:
                logger.warning("Skipping invalid Python file '%s' during model scan: %s", py_file, exc)
                continue

            info = cls._extract_engine_info_from_ast(py_file, tree)
            if info is not None:
                discovered[py_file.name] = info

        return discovered

    @classmethod
    def sync_models(
        cls,
        models_dir: Optional[Path] = None,
        registry_file: Optional[Path] = None,
        interactive: bool = False,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Scan `models/` directory for model (.py) files and synchronize the available model list:
        1. If any previously named model's .py file was deleted from `models/`, remove it from
           the registry file and unregister it from EngineRegistry.
        2. If a new .py model file is detected in `models/`, prompt the user to name it
           (displaying the corresponding .py filename) when `interactive=True`, and persist
           it to `models_registry.json`.
        3. Register all active discovered models lazily into EngineRegistry and update metadata.
        """
        with cls._lock:
            target_models_dir = (
                Path(models_dir).resolve()
                if models_dir is not None
                else (Path(__file__).resolve().parent.parent / "models")
            )
            target_registry_file = (
                Path(registry_file).resolve()
                if registry_file is not None
                else (target_models_dir / "models_registry.json")
            )

            discovered = cls._scan_model_engine_files(target_models_dir)
            saved_registry: Dict[str, Dict[str, Any]] = {}
            registry_changed = False

            if target_registry_file.exists():
                try:
                    raw = json.loads(target_registry_file.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        saved_registry = raw
                except Exception as exc:
                    logger.warning("Could not read '%s': %s", target_registry_file, exc)
                    saved_registry = {}
            else:
                # Seed initial default models (dfnet_engine.py, mock_engine.py) if present
                for seed_file, seed_meta in cls._DEFAULT_SEED_MODELS.items():
                    if seed_file in discovered:
                        saved_registry[seed_file] = dict(seed_meta)
                        registry_changed = True

            # Step 1: Prune models whose .py file is no longer in models_dir
            for saved_file in list(saved_registry.keys()):
                if saved_file not in discovered:
                    removed_info = saved_registry.pop(saved_file)
                    removed_id = str(removed_info.get("engine_id", "")).strip().lower()
                    removed_display = removed_info.get("display_name", removed_id or saved_file)
                    if removed_id:
                        cls.unregister(removed_id)
                    registry_changed = True
                    output_fn(
                        f"[QUÉT MODEL] Đã xoá model '{removed_display}' "
                        f"do không còn thấy file 'models/{saved_file}'."
                    )

            # Step 2: Reconcile discovered .py files (existing vs new)
            active_records: Dict[str, Dict[str, Any]] = {}

            for filename, info in discovered.items():
                engine_id = info["engine_id"]
                if filename in saved_registry:
                    saved_entry = saved_registry[filename]
                    old_id = str(saved_entry.get("engine_id", "")).strip().lower()
                    if old_id and old_id != engine_id:
                        cls.unregister(old_id)

                    record = {
                        "engine_id": engine_id,
                        "class_name": info["class_name"],
                        "display_name": saved_entry.get("display_name") or info["default_display_name"],
                        "aliases": info["aliases"],
                        "type": saved_entry.get("type") or info["type"],
                        "sample_rate": saved_entry.get("sample_rate") or info["sample_rate"],
                        "status": saved_entry.get("status") or info["status"],
                        "description": saved_entry.get("description") or info["description"],
                    }
                    if saved_registry[filename] != record:
                        saved_registry[filename] = record
                        registry_changed = True
                    active_records[filename] = record
                else:
                    default_name = info["default_display_name"]
                    if interactive:
                        output_fn("\n" + "=" * 78)
                        output_fn("[PHÁT HIỆN MODEL MỚI] Tìm thấy file model mới trong thư mục models/:")
                        output_fn(f"  • Tên file .py : models/{filename}")
                        output_fn(f"  • Tên Class    : {info['class_name']}")
                        output_fn(f"  • Mã Engine ID : {engine_id}")
                        output_fn("-" * 78)
                        prompt_text = (
                            f"Nhập tên hiển thị cho model của file '{filename}' "
                            f"(Nhấn Enter để dùng mặc định '{default_name}'): "
                        )
                        try:
                            user_input = input_fn(prompt_text).strip()
                        except (EOFError, KeyboardInterrupt):
                            user_input = ""
                        chosen_name = user_input if user_input else default_name
                        output_fn(f"✓ Đã lưu model '{chosen_name}' (file: models/{filename}).")
                        output_fn("=" * 78 + "\n")

                        record = {
                            "engine_id": engine_id,
                            "class_name": info["class_name"],
                            "display_name": chosen_name,
                            "aliases": info["aliases"],
                            "type": info["type"],
                            "sample_rate": info["sample_rate"],
                            "status": info["status"],
                            "description": info["description"],
                        }
                        saved_registry[filename] = record
                        active_records[filename] = record
                        registry_changed = True
                    else:
                        # Non-interactive pass: register in-memory with default name without
                        # marking as user-named in models_registry.json yet.
                        active_records[filename] = {
                            "engine_id": engine_id,
                            "class_name": info["class_name"],
                            "display_name": default_name,
                            "aliases": info["aliases"],
                            "type": info["type"],
                            "sample_rate": info["sample_rate"],
                            "status": info["status"],
                            "description": info["description"],
                        }

            # Step 3: Persist updated registry JSON if modified
            if registry_changed:
                try:
                    target_registry_file.parent.mkdir(parents=True, exist_ok=True)
                    target_registry_file.write_text(
                        json.dumps(saved_registry, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                except Exception as exc:
                    logger.warning("Failed to write model registry file '%s': %s", target_registry_file, exc)

            # Step 4: Register active models into EngineRegistry & metadata store
            for filename, record in active_records.items():
                eng_id = record["engine_id"]
                stem = Path(filename).stem
                cls_name = record["class_name"]
                aliases = record.get("aliases", [])
                cls._source_files[eng_id] = discovered[filename]["file_path"]
                cls.register_lazy(
                    eng_id,
                    f"models.{stem}.{cls_name}",
                    aliases=aliases,
                    force=True,
                )
                cls._metadata_registry[eng_id] = {
                    "name": record["display_name"],
                    "display_name": record["display_name"],
                    "engine_id": eng_id,
                    "class_name": cls_name,
                    "source_file": filename,
                    "type": record["type"],
                    "sample_rate": record["sample_rate"],
                    "status": record["status"],
                    "description": record["description"],
                }

            return dict(active_records)

    @classmethod
    def get_engine_metadata(cls, name: str) -> Dict[str, Any]:
        """Return descriptive UI metadata dictionary for a registered engine."""
        with cls._lock:
            canonical = cls._resolve_canonical_name(name) if name else "dfnet2"
            if canonical in cls._metadata_registry:
                return dict(cls._metadata_registry[canonical])
            return {
                "name": canonical.upper(),
                "display_name": canonical.upper(),
                "engine_id": canonical,
                "class_name": "",
                "source_file": "",
                "type": "AI Speech Enhancement Engine",
                "sample_rate": "48 kHz",
                "status": "Sẵn sàng (Ready)",
                "description": "Động cơ khử nhiễu giọng nói.",
            }

    @classmethod
    def _register_defaults(cls) -> None:
        """Register standard engine keys and synchronize with models/ directory."""
        cls.sync_models(interactive=False)
        # Fallback if neither default file was found in custom environment
        if "mock" not in cls._registry and "mock" not in cls._lazy_registry:
            cls.register_lazy("mock", "models.mock_engine.MockDenoiseEngine", aliases=["passthrough", "mock_dsp"])
        if "dfnet2" not in cls._registry and "dfnet2" not in cls._lazy_registry:
            cls.register_lazy("dfnet2", "models.dfnet_engine.DeepFilterNetEngine", aliases=["deepfilter", "df2"])

    @classmethod
    def is_engine_available(cls, name: str) -> bool:
        """Check if an engine can be successfully initialized or imported."""
        with cls._lock:
            try:
                canonical = cls._resolve_canonical_name(name)
                engine_cls = cls.get_engine_class(canonical)
                return True
            except Exception:
                return False


# Initialize standard defaults upon module import
EngineRegistry._register_defaults()


# ==============================================================================
# 5. GLOBAL CONVENIENCE HELPERS (1-Line Engine Switching)
# ==============================================================================

def get_engine(name: Optional[str] = None, **kwargs: Any) -> AudioDenoiseEngine:
    """
    Global factory helper for 1-line engine loading:
        engine = get_engine()
        engine = get_engine('mock')
        engine = get_engine('dfnet2')
    """
    return EngineRegistry.get_engine(name=name, **kwargs)


def list_available_engines() -> List[str]:
    """Return all available engine keys."""
    return EngineRegistry.list_available_engines()


def register_engine(
    name: str,
    aliases: Optional[List[str]] = None,
    force: bool = False,
) -> Callable[[Type[AudioDenoiseEngine]], Type[AudioDenoiseEngine]]:
    """Decorator to register an engine class."""
    return EngineRegistry.register(name, aliases=aliases, force=force)
