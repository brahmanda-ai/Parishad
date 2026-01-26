"""
Base classes and types for Parishad backends.

This module contains:
- BackendError: Exception for backend failures
- BackendConfig: Configuration dataclass
- BackendResult: Result dataclass
- ModelBackend: Protocol for backend implementations
- BaseBackend: Abstract base class
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class BackendError(Exception):
    """
    Raised when a backend operation fails.
    
    Attributes:
        backend_name: Name of the backend that failed
        model_id: Model identifier (if known)
        original_error: The underlying exception
    """
    
    def __init__(
        self,
        message: str,
        backend_name: str = "",
        model_id: str = "",
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.backend_name = backend_name
        self.model_id = model_id
        self.original_error = original_error


@dataclass
class BackendConfig:
    """
    Configuration for a model backend.
    
    This is a unified config structure that backends can use.
    Backend-specific options go in `extra`.
    """
    
    model_id: str
    """Model identifier (path, HuggingFace ID, or API model name)."""
    
    context_length: int = 4096
    """Maximum context window size in tokens."""
    
    temperature: float = 0.5
    """Default sampling temperature."""
    
    top_p: float = 0.9
    """Default nucleus sampling parameter."""
    
    max_tokens: int = 1024
    """Default maximum tokens to generate."""
    
    stop: list[str] | None = None
    """Default stop sequences."""
    
    timeout: float = 120.0
    """Request timeout in seconds."""
    
    extra: dict[str, Any] = field(default_factory=dict)
    """Backend-specific options (e.g., n_gpu_layers for llama.cpp)."""


@dataclass
class BackendResult:
    """
    Result from a backend generation call.
    
    All backends must return this structure for consistent handling.
    """
    
    text: str
    """Generated text content."""
    
    tokens_in: int
    """Number of input/prompt tokens."""
    
    tokens_out: int
    """Number of output/generated tokens."""
    
    model_id: str = ""
    """Model identifier used for generation."""
    
    finish_reason: str = "stop"
    """Why generation stopped: 'stop', 'length', 'error'."""
    
    latency_ms: float = 0.0
    """Generation latency in milliseconds."""
    
    extra: dict[str, Any] = field(default_factory=dict)
    """Backend-specific metadata."""


@runtime_checkable
class ModelBackend(Protocol):
    """
    Protocol for model backend implementations.
    
    All backends must implement these methods to be usable by ModelRunner.
    """
    
    @property
    def name(self) -> str:
        """Backend name (e.g., 'llama_cpp', 'openai', 'stub')."""
        ...
    
    @property
    def is_loaded(self) -> bool:
        """Whether the backend is ready to generate."""
        ...
    
    @property
    def model_id(self) -> str:
        """Current model identifier."""
        ...
    
    def load(self, config: BackendConfig) -> None:
        """Load/initialize the backend with the given config."""
        ...
    
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
    ) -> BackendResult:
        """Generate text completion."""
        ...
    
    def unload(self) -> None:
        """Unload the model to free resources."""
        ...


class BaseBackend(ABC):
    """
    Abstract base class for backend implementations.
    """
    
    _name: str = "base"
    _model_id: str = ""
    _loaded: bool = False
    _config: BackendConfig | None = None
    
    @property
    def name(self) -> str:
        """Backend name."""
        return self._name
    
    @property
    def is_loaded(self) -> bool:
        """Whether backend is ready."""
        return self._loaded
    
    @property
    def model_id(self) -> str:
        """Current model ID."""
        return self._model_id
    
    @abstractmethod
    def load(self, config: BackendConfig) -> None:
        """Load the backend. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
    ) -> BackendResult:
        """Generate text. Must be implemented by subclasses."""
        pass
    
    def unload(self) -> None:
        """Default unload implementation."""
        self._loaded = False
        self._model_id = ""
        self._config = None
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Cheap token estimation heuristic.
        
        Uses word count * 1.3 as a rough approximation.
        Override in subclasses for more accurate counting.
        """
        if not text:
            return 0
        # Rough approximation: ~1.3 tokens per word for English
        words = len(text.split())
        return int(words * 1.3)
