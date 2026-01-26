"""
Unified model abstraction layer for Parishad.

Provides a consistent interface for different LLM backends:
- llama.cpp (local GGUF models)
- OpenAI API
- HuggingFace Transformers

This module serves as the router that dispatches to the appropriate backend
based on configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple
import logging
import time

from ..roles.base import Slot
from .backends import (
    BackendConfig,
    BackendError,
    BackendResult,
    is_backend_available,
    ModelBackend as BackendProtocol,
    LlamaCppBackend,
    OpenAIBackend,
    TransformersBackend,
    MlxBackend,
    OllamaNativeBackend,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ModelRunnerError(Exception):
    """Base exception for ModelRunner errors."""
    pass


class UnknownSlotError(ModelRunnerError):
    """Raised when an unknown slot is requested."""
    pass


class ModelBackendError(ModelRunnerError):
    """Raised when backend model call fails."""
    pass


class BackendNotAvailableError(ModelRunnerError):
    """Raised when a required backend is not installed."""
    pass


# =============================================================================
# Enums and Configuration
# =============================================================================


class Backend(Enum):
    """Supported LLM backends."""
    LLAMA_CPP = "llama_cpp"
    OPENAI = "openai"
    OLLAMA = "ollama"
    TRANSFORMERS = "transformers"
    MLX = "mlx"
    ANTHROPIC = "anthropic"
    GRPC = "grpc"
    PRP = "prp"
    NATIVE = "native"


@dataclass
class SlotConfig:
    """
    Configuration for a model slot.
    
    Defines how a slot (SMALL/MID/BIG) should be configured,
    including the backend to use and generation parameters.
    """
    model_id: str
    backend: Backend | str = Backend.TRANSFORMERS
    
    # Context and generation settings
    context_length: int = 8192
    default_max_tokens: int = 1024
    default_temperature: float = 0.5
    top_p: float = 0.9
    
    # Stop sequences
    stop: list[str] | None = None
    
    # Timeout for generation
    timeout: float = 120.0
    
    # Backend-specific options (passed to backend as 'extra')
    # For llama_cpp: n_gpu_layers, n_batch, verbose, chat_format
    # For transformers: quantization, device_map, torch_dtype
    # For openai: api_key_env, base_url, organization
    quantization: Optional[str] = None
    device_map: str = "auto"
    model_file: Optional[str] = None
    n_gpu_layers: int = -1
    api_key_env: Optional[str] = None
    
    # Generic extra args (for backend-specific settings like host/port)
    extra: dict[str, Any] = field(default_factory=dict)
    
    # Legacy fields for backward compatibility
    max_context: int = 8192  # Alias for context_length
    top_k: int = 50
    repetition_penalty: float = 1.0
    
    def __post_init__(self):
        """Normalize backend to enum."""
        if isinstance(self.backend, str):
            try:
                self.backend = Backend(self.backend)
            except ValueError:
                # Keep as string for unknown backends
                pass
        
        # Sync max_context with context_length
        if self.max_context != 8192:
            self.context_length = self.max_context
    
    def to_backend_config(self) -> BackendConfig:
        """Convert to BackendConfig for the backends package."""
        extra = {
            "quantization": self.quantization,
            "device_map": self.device_map,
            "n_gpu_layers": self.n_gpu_layers,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }
        
        if self.model_file:
            extra["model_file"] = self.model_file
        if self.api_key_env:
            extra["api_key_env"] = self.api_key_env
            
        # Merge generic extra args
        if self.extra:
            extra.update(self.extra)
        
        return BackendConfig(
            model_id=self.model_file or self.model_id,
            context_length=self.context_length,
            temperature=self.default_temperature,
            top_p=self.top_p,
            max_tokens=self.default_max_tokens,
            stop=self.stop,
            timeout=self.timeout,
            extra=extra,
        )


@dataclass
class ModelConfig:
    """
    Complete model configuration for Parishad.
    
    Can be loaded from YAML or constructed programmatically.
    """
    slots: dict[str, SlotConfig] = field(default_factory=dict)
    
    # Cost tracking weights per slot
    token_weights: dict[str, float] = field(default_factory=lambda: {
        "small": 1.0,
        "mid": 2.5,
        "big": 5.0
    })
    
    # Legacy attributes for backward compatibility
    small: SlotConfig | None = None
    mid: SlotConfig | None = None
    big: SlotConfig | None = None
    default_temperature: float = 0.5
    
    def __post_init__(self):
        """Initialize slots from legacy attributes if provided."""
        if self.small and "small" not in self.slots:
            self.slots["small"] = self.small
        if self.mid and "mid" not in self.slots:
            self.slots["mid"] = self.mid
        if self.big and "big" not in self.slots:
            self.slots["big"] = self.big
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        """Load configuration from YAML file."""
        import yaml
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        slots = {}
        for slot_name, slot_data in data.get("slots", {}).items():
            slots[slot_name] = cls._parse_slot_config(slot_data)
        
        token_weights = data.get("cost", {}).get("token_weights", {
            "small": 1.0, "mid": 2.5, "big": 5.0
        })
        
        return cls(slots=slots, token_weights=token_weights)
    
    @classmethod
    def from_profile(cls, profile_name: str, path: str | Path) -> "ModelConfig":
        """
        Load a specific profile from YAML file.
        
        Args:
            profile_name: Name of the profile to load (e.g., 'stub', 'local_small')
            path: Path to the YAML config file
            
        Returns:
            ModelConfig with the profile's slot configurations
        """
        import yaml
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        profiles = data.get("profiles", {})
        
        if profile_name not in profiles:
            available = list(profiles.keys())
            raise ValueError(
                f"Profile '{profile_name}' not found. Available: {available}"
            )
        
        profile_data = profiles[profile_name]
        
        slots = {}
        slots_data = profile_data.get("slots", profile_data)
        for slot_name, slot_data in slots_data.items():
            slots[slot_name] = cls._parse_slot_config(slot_data)
        
        token_weights = data.get("cost", {}).get("token_weights", {
            "small": 1.0, "mid": 2.5, "big": 5.0
        })
        
        return cls(slots=slots, token_weights=token_weights)
    
    @staticmethod
    def _parse_slot_config(slot_data: dict) -> SlotConfig:
        """Parse a slot configuration dictionary."""
        backend_str = slot_data.get("backend", "transformers")
        try:
            backend = Backend(backend_str)
        except ValueError:
            backend = backend_str  # Keep as string
        
        return SlotConfig(
            model_id=slot_data.get("model_id", ""),
            backend=backend,
            context_length=slot_data.get("context_length", slot_data.get("max_context", 8192)),
            default_max_tokens=slot_data.get("max_tokens", slot_data.get("default_max_tokens", 1024)),
            default_temperature=slot_data.get("temperature", slot_data.get("default_temperature", 0.5)),
            top_p=slot_data.get("top_p", 0.9),
            stop=slot_data.get("stop"),
            timeout=slot_data.get("timeout", 120.0),
            quantization=slot_data.get("quantization"),
            device_map=slot_data.get("device_map", "auto"),
            model_file=slot_data.get("model_file"),
            n_gpu_layers=slot_data.get("n_gpu_layers", -1),
            api_key_env=slot_data.get("api_key_env"),
            max_context=slot_data.get("max_context", 8192),
            top_k=slot_data.get("top_k", 50),
            repetition_penalty=slot_data.get("repetition_penalty", 1.0),
            extra=slot_data.get("extra", {}), # Pass through extra config
        )


# =============================================================================
# Backend Factory
# =============================================================================


def _create_backend(backend_type: Backend | str) -> BackendProtocol:
    """
    Create a backend instance based on type using unified factory.
    
    Args:
        backend_type: Backend enum or string name
        
    Returns:
        Backend instance (not yet loaded/configured)
        
    Raises:
        BackendNotAvailableError: If backend dependencies not installed
    """
    if isinstance(backend_type, str):
        backend_name = backend_type
    else:
        backend_name = backend_type.value
    
    if backend_name == "llama_cpp":
        if not is_backend_available("llama_cpp"):
            raise BackendNotAvailableError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            )
        return LlamaCppBackend()
    
    if backend_name == "openai":
        if not is_backend_available("openai"):
            raise BackendNotAvailableError(
                "openai package is not installed. "
                "Install with: pip install openai"
            )
        return OpenAIBackend()
    
    if backend_name == "transformers":
        if not is_backend_available("transformers"):
            raise BackendNotAvailableError(
                "transformers and torch are not installed. "
                "Install with: pip install transformers torch"
            )
        return TransformersBackend()
    
    if backend_name == "mlx":
        if not is_backend_available("mlx"):
            raise BackendNotAvailableError(
                "mlx-lm is not installed. "
                "Install with: pip install mlx-lm"
            )
        return MlxBackend()
    
    if backend_name == "ollama":
        if not is_backend_available("ollama_native"):
             raise BackendNotAvailableError(
                 "requests package is not installed. "
                 "Install with: pip install requests"
             )
        return OllamaNativeBackend()

    raise ValueError(f"Unknown backend type: {backend_name}")


# =============================================================================
# ModelRunner
# =============================================================================


class ModelRunner:
    """
    Unified interface for running models across different backends.
    
    Manages multiple model slots (SMALL/MID/BIG) and routes generation
    requests to the appropriate backend based on configuration.
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
    ):
        """
        Initialize ModelRunner.
        
        Args:
            config: Model configuration. If None, uses defaults.
        """
        self.config = config or ModelConfig()
        self._backends: dict[str, BackendProtocol] = {}
        self._loaded_slots: set[str] = set()
    
    @classmethod
    def from_profile(
        cls,
        profile_name: str,
        config_path: str | Path,
    ) -> "ModelRunner":
        """
        Create ModelRunner from a named profile in config file.
        
        Args:
            profile_name: Profile to load (e.g., 'local_small')
            config_path: Path to models.yaml configuration
            
        Returns:
            Configured ModelRunner
        """
        config = ModelConfig.from_profile(profile_name, config_path)
        return cls(config=config)
    
    def _get_backend(self, slot: Slot) -> BackendProtocol:
        """
        Get or create backend for a slot.
        
        Args:
            slot: Slot to get backend for
            
        Returns:
            Backend instance for the slot
        """
        slot_name = slot.value
        
        if slot_name in self._backends:
            return self._backends[slot_name]
        
        slot_config = self.config.slots.get(slot_name)
        if not slot_config:
            raise UnknownSlotError(f"No configuration for slot: {slot_name}")
        
        backend = _create_backend(slot_config.backend)
        
        # Get backend type
        backend_type = slot_config.backend
        if isinstance(backend_type, Backend):
            backend_name = backend_type.value
        else:
            backend_name = backend_type
        
        # Create backend using unified factory
        try:
            backend = _create_backend(backend_name)
            self._backends[slot_name] = backend
            return backend
        except BackendNotAvailableError:
            raise
        except Exception as e:
            raise BackendError(
                f"Failed to create backend '{backend_name}' for slot '{slot_name}': {e}"
            ) from e
    
    def load_slot(self, slot: Slot) -> None:
        """
        Load a model slot.
        
        Args:
            slot: Slot to load
        """
        slot_name = slot.value
        
        if slot_name in self._loaded_slots:
            return
        
        slot_config = self.config.slots.get(slot_name)
        
        if not slot_config:
            raise UnknownSlotError(f"No configuration for slot: {slot_name}")
        
        # Get backend (already configured via factory)
        backend = self._get_backend(slot)
        
        # Build backend config
        backend_config = slot_config.to_backend_config()
        
        # DEBUG: Log what we're loading
        logger.info(f"Loading slot {slot_name}: model_id={backend_config.model_id}")
        
        # Load the backend with configuration
        backend.load(backend_config)
        
        # Mark slot as loaded
        self._loaded_slots.add(slot_name)
        
        backend_name = backend.name if hasattr(backend, 'name') else str(type(backend).__name__)
        logger.info(f"Slot {slot_name} loaded with backend {backend_name}")
    
    def unload_slot(self, slot: Slot) -> None:
        """
        Unload a model slot to free memory.
        
        Args:
            slot: Slot to unload
        """
        slot_name = slot.value
        
        if slot_name not in self._loaded_slots:
            return
        
        if slot_name in self._backends:
            self._backends[slot_name].unload()
            del self._backends[slot_name]
        
        self._loaded_slots.discard(slot_name)
        logger.info(f"Slot {slot_name} unloaded")
    
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        slot: Slot,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> Tuple[str, int, str]:
        """
        Generate a response using the specified slot.
        
        Args:
            system_prompt: System prompt for the model
            user_message: User message to respond to
            slot: Which model slot to use (SMALL, MID, BIG)
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)
            stop: Stop sequences (overrides config)
            **kwargs: Additional backend-specific parameters
            
        Returns:
            Tuple of (response_text, tokens_used, model_id)
            
        Raises:
            UnknownSlotError: If slot is not configured
            ModelBackendError: If backend call fails
        """
        # Validate slot
        if not isinstance(slot, Slot):
            raise UnknownSlotError(
                f"Invalid slot type: {type(slot)}. Must be Slot enum."
            )
        
        slot_name = slot.value
        
        # Check if slot exists in config
        if slot_name not in self.config.slots:
            raise UnknownSlotError(
                f"Slot '{slot_name}' not found in configuration. "
                f"Available slots: {list(self.config.slots.keys())}"
            )
        
        # Validate parameters
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        
        if temperature is not None and not (0.0 <= temperature <= 2.0):
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {temperature}"
            )
        
        try:
            # Ensure slot is loaded
            if slot_name not in self._loaded_slots:
                self.load_slot(slot)
            
            backend = self._backends.get(slot_name)
            if not backend:
                raise UnknownSlotError(
                    f"Backend for slot '{slot_name}' not initialized"
                )
            
            # Get defaults from config
            slot_config = self.config.slots.get(slot_name)
            if slot_config:
                max_tokens = max_tokens or slot_config.default_max_tokens
                temperature = (
                    temperature if temperature is not None
                    else slot_config.default_temperature
                )
                stop = stop or slot_config.stop
                top_p = slot_config.top_p
            else:
                max_tokens = max_tokens or 1024
                temperature = temperature if temperature is not None else 0.5
                top_p = 0.9
            
            # Format prompt (combine system and user for backends)
            full_prompt = self._format_prompt(system_prompt, user_message)
            
            # Generate
            start_time = time.perf_counter()
            
            result: BackendResult = backend.generate(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            
            elapsed = time.perf_counter() - start_time
            
            # Calculate total tokens used
            tokens_used = result.tokens_in + result.tokens_out
            
            logger.debug(
                f"Generated {result.tokens_out} tokens with {result.model_id} "
                f"in {elapsed:.2f}s (latency: {result.latency_ms:.1f}ms)"
            )
            
            return result.text, tokens_used, result.model_id
            
        except (UnknownSlotError, ValueError):
            # Re-raise validation errors as-is
            raise
        except BackendError as e:
            # Convert backend errors
            logger.error(f"Backend error for slot {slot_name}: {e}")
            raise ModelBackendError(
                f"Backend error in slot '{slot_name}': {e}"
            ) from e
        except Exception as e:
            # Wrap all other exceptions
            logger.error(f"Model generation failed for slot {slot_name}: {e}")
            raise ModelBackendError(
                f"Backend error in slot '{slot_name}': {type(e).__name__}: {e}"
            ) from e
    
    def _format_prompt(self, system_prompt: str, user_message: str) -> str:
        """
        Format system and user prompts into a single string.
        Using Llama-3 Chat Template for better Llama-3.2-1B control.
        """
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_message}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    
    def get_token_weight(self, slot: Slot) -> float:
        """
        Get the token cost weight for a slot.
        
        Args:
            slot: Slot to get weight for
            
        Returns:
            Token weight multiplier
        """
        return self.config.token_weights.get(slot.value, 1.0)
    
    def unload_all(self) -> None:
        """Unload all loaded model slots."""
        for slot_name in list(self._loaded_slots):
            try:
                self.unload_slot(Slot(slot_name))
            except ValueError:
                # Skip invalid slot names
                pass
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload_all()
        except Exception:
            pass  # Ignore errors during cleanup
