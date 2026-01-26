"""
Profile management for Parishad model configuration.

Provides:
- ProfileManager: Central class for profile switching and management
- Environment detection: Auto-detect available backends and hardware
- Graceful fallbacks: Fall back to simpler profiles when backends unavailable
- Profile validation: Validate profiles before loading

Environment Variables:
- PARISHAD_PROFILE: Override default profile selection
- PARISHAD_CONFIG_PATH: Override default config file path
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .backends import get_available_backends, is_backend_available


logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================


class ProfileMode(Enum):
    """Profile execution modes."""
    LOCAL = "local"        # Local model inference
    API = "api"            # API-based inference
    HYBRID = "hybrid"      # Mix of local and API


class HardwareCapability(Enum):
    """Detected hardware capabilities."""
    CPU_ONLY = "cpu_only"
    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    APPLE_SILICON = "apple_silicon"
    UNKNOWN = "unknown"


# =============================================================================
# Environment Detection
# =============================================================================


@dataclass
class EnvironmentInfo:
    """
    Detected environment information.
    
    Contains information about available backends, hardware, and system state.
    """
    available_backends: list[str] = field(default_factory=list)
    hardware: HardwareCapability = HardwareCapability.UNKNOWN
    gpu_memory_gb: Optional[float] = None
    cpu_cores: int = 1
    system_memory_gb: float = 8.0
    
    # API key availability
    has_openai_key: bool = False
    has_anthropic_key: bool = False
    
    # Python packages
    has_torch: bool = False
    has_transformers: bool = False
    has_llama_cpp: bool = False
    has_tiktoken: bool = False
    
    @property
    def can_run_local(self) -> bool:
        """Check if local inference is possible."""
        return self.has_llama_cpp or self.has_transformers
    
    @property
    def can_run_api(self) -> bool:
        """Check if API inference is possible."""
        return self.has_openai_key or self.has_anthropic_key
    
    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return self.hardware in (
            HardwareCapability.NVIDIA_GPU,
            HardwareCapability.AMD_GPU,
            HardwareCapability.APPLE_SILICON,
        )


def detect_environment() -> EnvironmentInfo:
    """
    Detect the current environment capabilities.
    
    Returns:
        EnvironmentInfo with detected capabilities
    """
    info = EnvironmentInfo()
    
    # Get available backends
    info.available_backends = get_available_backends()
    
    # Check backend-specific packages
    info.has_llama_cpp = is_backend_available("llama_cpp")
    
    try:
        import torch
        info.has_torch = True
        
        # Detect GPU
        if torch.cuda.is_available():
            info.hardware = HardwareCapability.NVIDIA_GPU
            try:
                info.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception:
                pass
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info.hardware = HardwareCapability.APPLE_SILICON
        elif hasattr(torch, 'hip') and torch.hip.is_available():
            info.hardware = HardwareCapability.AMD_GPU
        else:
            info.hardware = HardwareCapability.CPU_ONLY
    except ImportError:
        pass
    
    try:
        import transformers  # noqa: F401
        info.has_transformers = True
    except ImportError:
        pass
    
    try:
        import tiktoken  # noqa: F401
        info.has_tiktoken = True
    except ImportError:
        pass
    
    # Check API keys
    info.has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    info.has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    
    # System resources
    info.cpu_cores = os.cpu_count() or 1
    
    try:
        import psutil
        info.system_memory_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    
    return info


# =============================================================================
# Profile Definitions
# =============================================================================


@dataclass
class ProfileDefinition:
    """
    Definition of a profile with its requirements and fallbacks.
    """
    name: str
    mode: ProfileMode
    description: str
    
    # Required backends
    required_backends: list[str] = field(default_factory=list)
    
    # Fallback chain - try these profiles in order if this one fails
    fallback_chain: list[str] = field(default_factory=list)
    
    # Minimum requirements
    min_memory_gb: float = 0.0
    requires_gpu: bool = False
    requires_api_key: Optional[str] = None
    
    # Priority (higher = preferred)
    priority: int = 0
    
    def check_requirements(self, env: EnvironmentInfo) -> tuple[bool, str]:
        """
        Check if environment meets profile requirements.
        
        Args:
            env: Environment info
            
        Returns:
            Tuple of (is_compatible, reason_if_not)
        """
        # Check backends
        for backend in self.required_backends:
            if backend not in env.available_backends:
                return False, f"Backend '{backend}' not available"
        
        # Check GPU
        if self.requires_gpu and not env.has_gpu:
            return False, "GPU required but not available"
        
        # Check memory
        if self.min_memory_gb > 0 and env.system_memory_gb < self.min_memory_gb:
            return False, f"Requires {self.min_memory_gb}GB RAM, have {env.system_memory_gb:.1f}GB"
        
        # Check API keys
        if self.requires_api_key == "openai" and not env.has_openai_key:
            return False, "OPENAI_API_KEY not set"
        if self.requires_api_key == "anthropic" and not env.has_anthropic_key:
            return False, "ANTHROPIC_API_KEY not set"
        
        return True, ""


# Built-in profile definitions
BUILTIN_PROFILES: dict[str, ProfileDefinition] = {
    "local_gpu": ProfileDefinition(
        name="local_gpu",
        mode=ProfileMode.LOCAL,
        description="Local inference with GPU acceleration (Metal/CUDA)",
        required_backends=["llama_cpp"],
        fallback_chain=["local_small"],
        min_memory_gb=4.0,
        requires_gpu=True,
        priority=25,  # Higher than local_small, will be preferred
    ),
    "local_small": ProfileDefinition(
        name="local_small",
        mode=ProfileMode.LOCAL,
        description="Local inference with small models (1.5B-7B)",
        required_backends=["llama_cpp"],
        fallback_chain=[],
        min_memory_gb=4.0,
        priority=10,
    ),
    "local_medium": ProfileDefinition(
        name="local_medium",
        mode=ProfileMode.LOCAL,
        description="Local inference with medium models (7B-14B)",
        required_backends=["llama_cpp"],
        fallback_chain=["local_small"],
        min_memory_gb=8.0,
        priority=20,
    ),
    "local_large": ProfileDefinition(
        name="local_large",
        mode=ProfileMode.LOCAL,
        description="Local inference with large models (14B+)",
        required_backends=["llama_cpp"],
        fallback_chain=["local_medium", "local_small"],
        min_memory_gb=16.0,
        requires_gpu=True,
        priority=30,
    ),
    "transformers": ProfileDefinition(
        name="transformers",
        mode=ProfileMode.LOCAL,
        description="HuggingFace Transformers models",
        required_backends=["transformers"],
        fallback_chain=["local_small"],
        min_memory_gb=8.0,
        priority=15,
    ),
    "openai": ProfileDefinition(
        name="openai",
        mode=ProfileMode.API,
        description="OpenAI API models (GPT-4, etc.)",
        required_backends=["openai"],
        fallback_chain=[],
        requires_api_key="openai",
        priority=50,
    ),
    "anthropic": ProfileDefinition(
        name="anthropic",
        mode=ProfileMode.API,
        description="Anthropic API models (Claude, etc.)",
        required_backends=[],  # No special backend needed
        fallback_chain=["openai"],
        requires_api_key="anthropic",
        priority=50,
    ),
}


# =============================================================================
# Profile Manager
# =============================================================================


class ProfileManager:
    """
    Central manager for profile switching and management.
    
    Handles:
    - Profile selection based on environment
    - Graceful fallbacks when backends unavailable
    - Environment variable overrides
    - Profile validation
    
    Usage:
        manager = ProfileManager()
        profile = manager.select_profile()  # Auto-select best profile
        runner = manager.create_runner(profile)
    """
    
    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        env_info: Optional[EnvironmentInfo] = None,
    ):
        """
        Initialize ProfileManager.
        
        Args:
            config_path: Path to models.yaml config file
            env_info: Pre-detected environment info (auto-detects if None)
        """
        self.config_path = Path(config_path) if config_path else None
        self.env_info = env_info or detect_environment()
        self._profiles: dict[str, ProfileDefinition] = BUILTIN_PROFILES.copy()
        self._loaded_profiles: dict[str, Any] = {}
        
        # Check for config path override
        if not self.config_path:
            env_path = os.environ.get("PARISHAD_CONFIG_PATH")
            if env_path:
                self.config_path = Path(env_path)
    
    def register_profile(self, profile: ProfileDefinition) -> None:
        """
        Register a custom profile definition.
        
        Args:
            profile: Profile definition to register
        """
        self._profiles[profile.name] = profile
        logger.debug(f"Registered profile: {profile.name}")
    
    def get_profile_definition(self, name: str) -> Optional[ProfileDefinition]:
        """Get profile definition by name."""
        return self._profiles.get(name)
    
    def list_profiles(self) -> list[str]:
        """List all registered profile names."""
        return list(self._profiles.keys())
    
    def list_compatible_profiles(self) -> list[str]:
        """List profiles that are compatible with current environment."""
        compatible = []
        for name, profile in self._profiles.items():
            is_compatible, _ = profile.check_requirements(self.env_info)
            if is_compatible:
                compatible.append(name)
        return compatible
    
    def validate_profile(self, name: str) -> tuple[bool, str]:
        """
        Validate that a profile can be used.
        
        Args:
            name: Profile name to validate
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        profile = self._profiles.get(name)
        if not profile:
            return False, f"Unknown profile: {name}"
        
        return profile.check_requirements(self.env_info)
    
    def select_profile(
        self,
        preferred: Optional[str] = None,
        mode: Optional[ProfileMode] = None,
        allow_fallback: bool = True,
    ) -> str:
        """
        Select the best profile for the current environment.
        
        Args:
            preferred: Preferred profile name (overrides auto-selection)
            mode: Preferred mode (LOCAL, API, etc.)
            allow_fallback: If True, fall back to compatible profile if preferred fails
            
        Returns:
            Selected profile name
            
        Raises:
            RuntimeError: If no compatible profile found
        """
        # Check environment variable override
        env_profile = os.environ.get("PARISHAD_PROFILE")
        if env_profile:
            is_valid, reason = self.validate_profile(env_profile)
            if is_valid:
                logger.info(f"Using profile from PARISHAD_PROFILE: {env_profile}")
                return env_profile
            elif not allow_fallback:
                raise RuntimeError(
                    f"PARISHAD_PROFILE={env_profile} is not compatible: {reason}"
                )
            logger.warning(
                f"PARISHAD_PROFILE={env_profile} not compatible ({reason}), "
                "falling back to auto-selection"
            )
        
        # Try preferred profile
        if preferred:
            is_valid, reason = self.validate_profile(preferred)
            if is_valid:
                return preferred
            elif not allow_fallback:
                raise RuntimeError(f"Profile '{preferred}' not compatible: {reason}")
            
            # Try fallback chain
            profile_def = self._profiles.get(preferred)
            if profile_def:
                for fallback in profile_def.fallback_chain:
                    is_valid, _ = self.validate_profile(fallback)
                    if is_valid:
                        logger.warning(
                            f"Profile '{preferred}' not available ({reason}), "
                            f"falling back to '{fallback}'"
                        )
                        return fallback
        
        # Auto-select best profile
        compatible = []
        for name, profile in self._profiles.items():
            is_valid, _ = profile.check_requirements(self.env_info)
            if is_valid:
                # Filter by mode if specified
                if mode and profile.mode != mode:
                    continue
                compatible.append((name, profile))
        
        if not compatible:
            if mode:
                raise RuntimeError(
                    f"No compatible profiles found for mode '{mode.value}'. "
                    f"Available backends: {self.env_info.available_backends}"
                )
            raise RuntimeError(
                "No compatible profiles found. "
                f"Available backends: {self.env_info.available_backends}"
            )
        
        # Sort by priority (highest first)
        compatible.sort(key=lambda x: x[1].priority, reverse=True)
        
        selected = compatible[0][0]
        logger.info(f"Auto-selected profile: {selected}")
        return selected
    
    def create_runner(
        self,
        profile_name: Optional[str] = None,
        **kwargs,
    ) -> "ModelRunner":  # type: ignore
        """
        Create a ModelRunner with the specified or auto-selected profile.
        
        Args:
            profile_name: Profile to use (auto-selects if None)
            **kwargs: Additional arguments for ModelRunner
            
        Returns:
            Configured ModelRunner
        """
        from .runner import ModelRunner
        
        # Select profile if not specified
        if not profile_name:
            profile_name = self.select_profile()
        
        # Load from config file
        if self.config_path and self.config_path.exists():
            return ModelRunner.from_profile(profile_name, self.config_path, **kwargs)
        
        # No config file
        logger.warning(
            f"Config file not found and no profile selected. "
            f"Set PARISHAD_CONFIG_PATH or provide config_path."
        )
        return ModelRunner(**kwargs)
    
    def get_profile_info(self, name: str) -> dict[str, Any]:
        """
        Get detailed information about a profile.
        
        Args:
            name: Profile name
            
        Returns:
            Dict with profile details
        """
        profile = self._profiles.get(name)
        if not profile:
            return {"error": f"Unknown profile: {name}"}
        
        is_compatible, reason = profile.check_requirements(self.env_info)
        
        return {
            "name": profile.name,
            "mode": profile.mode.value,
            "description": profile.description,
            "compatible": is_compatible,
            "incompatibility_reason": reason if not is_compatible else None,
            "required_backends": profile.required_backends,
            "fallback_chain": profile.fallback_chain,
            "priority": profile.priority,
        }
    
    def get_environment_summary(self) -> dict[str, Any]:
        """Get a summary of the detected environment."""
        return {
            "available_backends": self.env_info.available_backends,
            "hardware": self.env_info.hardware.value,
            "has_gpu": self.env_info.has_gpu,
            "gpu_memory_gb": self.env_info.gpu_memory_gb,
            "cpu_cores": self.env_info.cpu_cores,
            "system_memory_gb": round(self.env_info.system_memory_gb, 1),
            "can_run_local": self.env_info.can_run_local,
            "can_run_api": self.env_info.can_run_api,
            "api_keys": {
                "openai": self.env_info.has_openai_key,
                "anthropic": self.env_info.has_anthropic_key,
            },
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def get_default_profile() -> str:
    """
    Get the default profile for the current environment.
    
    Checks PARISHAD_PROFILE env var first, then auto-selects.
    """
    manager = ProfileManager()
    return manager.select_profile()


def get_profile_manager(config_path: Optional[str | Path] = None) -> ProfileManager:
    """
    Create a ProfileManager with optional config path.
    
    Args:
        config_path: Optional path to models.yaml
        
    Returns:
        Configured ProfileManager
    """
    return ProfileManager(config_path=config_path)


def quick_runner(
    profile: Optional[str] = None,
    config_path: Optional[str | Path] = None,
) -> "ModelRunner":  # type: ignore
    """
    Quickly create a ModelRunner with sensible defaults.
    
    Args:
        profile: Profile name (auto-selects if None)
        config_path: Config file path
        
    Returns:
        Ready-to-use ModelRunner
        
    Example:
        runner = quick_runner("local_small", "config/models.yaml")
        text, tokens, model = runner.generate(
            "You are helpful.",
            "Hello!",
            Slot.SMALL
        )
    """
    manager = ProfileManager(config_path=config_path)
    return manager.create_runner(profile)


__all__ = [
    # Enums
    "ProfileMode",
    "HardwareCapability",
    # Data classes
    "EnvironmentInfo",
    "ProfileDefinition",
    # Manager
    "ProfileManager",
    # Functions
    "detect_environment",
    "get_default_profile",
    "get_profile_manager",
    "quick_runner",
    # Constants
    "BUILTIN_PROFILES",
]
