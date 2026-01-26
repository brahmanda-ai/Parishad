"""
Backend implementations for Parishad model runners.

This package provides pluggable backend implementations for different LLM providers:
- LlamaCppBackend: Local GGUF models via llama-cpp-python
- OpenAIBackend: OpenAI API (and compatible endpoints)
- OllamaBackend: Ollama local server (OpenAI-compatible)
- OllamaNativeBackend: Ollama using native API
- TransformersBackend: HuggingFace Transformers models
- HuggingFaceBackend: HuggingFace Inference API (cloud)
- HuggingFaceChatBackend: HuggingFace chat completion API
- MlxBackend: Apple Silicon (M1/M2/M3/M4) via MLX
"""

from __future__ import annotations

# Base classes and types - always available
from .base import (
    BackendError,
    BackendConfig,
    BackendResult,
    ModelBackend,
    BaseBackend,
)

# Conditional imports for optional backends
_LLAMA_CPP_AVAILABLE = False
_OPENAI_AVAILABLE = False
_OLLAMA_AVAILABLE = False
_OLLAMA_NATIVE_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False
_HUGGINGFACE_AVAILABLE = False
_MLX_AVAILABLE = False


# LlamaCpp
try:
    from .llama_cpp import LlamaCppBackend
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    LlamaCppBackend = None  # type: ignore

# OpenAI
try:
    from .openai_api import OpenAIBackend, OllamaBackend
    _OPENAI_AVAILABLE = True
    _OLLAMA_AVAILABLE = True
except ImportError:
    OpenAIBackend = None  # type: ignore
    OllamaBackend = None  # type: ignore

# Transformers
try:
    from .transformers_hf import TransformersBackend
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    TransformersBackend = None  # type: ignore

# HuggingFace Inference API
try:
    from .huggingface import HuggingFaceBackend, HuggingFaceChatBackend
    _HUGGINGFACE_AVAILABLE = True
except ImportError:
    HuggingFaceBackend = None  # type: ignore
    HuggingFaceChatBackend = None  # type: ignore

# Ollama Native
try:
    from .ollama import OllamaNativeBackend
    _OLLAMA_NATIVE_AVAILABLE = True
except ImportError:
    OllamaNativeBackend = None  # type: ignore

# MLX
try:
    from .mlx_lm import MlxBackend
    _MLX_AVAILABLE = True
except ImportError:
    MlxBackend = None  # type: ignore




def is_backend_available(name: str) -> bool:
    """Check if a backend's dependencies are installed."""
    if name == "llama_cpp":
        try:
            import llama_cpp
            return True
        except ImportError:
            return False
    elif name == "openai":
        try:
            import openai
            return True
        except ImportError:
            return False
    elif name == "transformers":
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False
    elif name == "ollama":
        try:
            import openai
            return True
        except ImportError:
            return False
    elif name == "mlx":
        try:
            import mlx_lm
            return True
        except ImportError:
            return False

    elif name == "ollama_native":
        try:
            import requests
            return True
        except ImportError:
            return False
    elif name in ("huggingface", "huggingface_chat"):
        try:
            import huggingface_hub
            return True
        except ImportError:
            return False
    return False


def get_available_backends() -> list[str]:
    """Get a list of currently available backends."""
    backends = []
    
    if is_backend_available("llama_cpp"):
        backends.append("llama_cpp")
    if is_backend_available("openai"):
        backends.append("openai")
    if is_backend_available("transformers"):
        backends.append("transformers")
    if is_backend_available("ollama"):
        backends.append("ollama")
    if is_backend_available("mlx"):
        backends.append("mlx")
    if is_backend_available("ollama_native"):
        backends.append("ollama_native")
    if is_backend_available("huggingface"):
        backends.append("huggingface")
        backends.append("huggingface_chat")
        
    return backends


def get_backend(backend_name: str) -> BaseBackend:
    """
    Factory function to get a backend instance by name.
    
    Args:
        backend_name: Name of the backend
    
    Returns:
        Backend instance (not loaded)
        
    Raises:
        ValueError: If backend unknown or deps not installed
    """
    backend_map = {
        "llama_cpp": (LlamaCppBackend, _LLAMA_CPP_AVAILABLE),
        "openai": (OpenAIBackend, _OPENAI_AVAILABLE),
        "ollama": (OllamaBackend, _OLLAMA_AVAILABLE),
        "ollama_native": (OllamaNativeBackend, _OLLAMA_NATIVE_AVAILABLE),
        "transformers": (TransformersBackend, _TRANSFORMERS_AVAILABLE),
        "huggingface": (HuggingFaceBackend, _HUGGINGFACE_AVAILABLE),
        "huggingface_chat": (HuggingFaceChatBackend, _HUGGINGFACE_AVAILABLE),
        "mlx": (MlxBackend, _MLX_AVAILABLE),
    }
    
    if backend_name not in backend_map:
        available = ", ".join(backend_map.keys())
        raise ValueError(
            f"Unknown backend: '{backend_name}'. "
            f"Available backends: {available}"
        )
    
    backend_class, is_available = backend_map[backend_name]
    
    if not is_available or backend_class is None:
        raise ValueError(
            f"Backend '{backend_name}' is not available. "
            f"Required dependencies are not installed. "
            f"Available backends: {', '.join(get_available_backends())}"
        )
    
    return backend_class()


# Stub backend for testing
class StubBackend(BaseBackend):
    """Simple stub backend for testing."""
    
    _name = "stub"
    
    def load(self, config: BackendConfig) -> None:
        self._config = config
        self._model_id = config.model_id
        self._loaded = True
    
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
    ) -> BackendResult:
        return BackendResult(
            text="[STUB RESPONSE]",
            tokens_in=self._estimate_tokens(prompt),
            tokens_out=10,
            model_id=self._model_id,
        )


__all__ = [
    # Base classes
    "BackendError",
    "BackendConfig",
    "BackendResult",
    "ModelBackend",
    "BaseBackend",
    # Backend implementations
    "LlamaCppBackend",
    "OpenAIBackend",
    "OllamaBackend",
    "OllamaNativeBackend",
    "TransformersBackend",
    "HuggingFaceBackend",
    "HuggingFaceChatBackend",
    "MlxBackend",
    "StubBackend",
    # Utilities
    "is_backend_available",
    "get_available_backends",
    "get_backend",
]
