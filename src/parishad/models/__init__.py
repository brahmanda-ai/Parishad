"""Model abstraction layer for Parishad."""

from .runner import (
    ModelRunner,
    ModelConfig,
    SlotConfig,
    Backend,
    ModelRunnerError,
    UnknownSlotError,
    ModelBackendError,
    BackendNotAvailableError,
    TransformersBackend,
    OpenAIBackend,
)

from .backends import (
    BackendConfig,
    BackendResult,
    BackendError,
    ModelBackend,
    is_backend_available,
    get_available_backends,
)

from .tokenization import (
    estimate_tokens,
    estimate_tokens_simple,
    estimate_tokens_hybrid,
    estimate_prompt_tokens,
    count_tokens_tiktoken,
    is_tiktoken_available,
    get_tokenizer,
    register_tokenizer,
)

from .costs import (
    estimate_cost,
    estimate_query_cost,
    get_model_pricing,
    ModelPricing,
    CostMetrics,
    estimate_flops,
    get_model_size,
)

from .profiles import (
    ProfileManager,
    ProfileMode,
    ProfileDefinition,
    HardwareCapability,
    EnvironmentInfo,
    detect_environment,
    get_default_profile,
    get_profile_manager,
    quick_runner,
    BUILTIN_PROFILES,
)

# Task 7: Performance optimizations
from .optimizations import (
    ResponseCache,
    PersistentCache,
    RequestBatcher,
    ConnectionPool,
    RateLimiter,
    OptimizedRunner,
)

# Task 8: Reliability
from .reliability import (
    RetryStrategy,
    RetryPolicy,
    TimeoutConfig,
    TimeoutManager,
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
    FallbackChain,
    HealthStatus,
    HealthChecker,
    ResilientBackend,
)

# Model Download Manager
from .downloader import (
    ModelManager,
    ModelRegistry,
    ModelInfo,
    ModelSource,
    ModelFormat,
    HuggingFaceDownloader,
    OllamaManager,
    LMStudioManager,
    DownloadProgress,
    DEFAULT_MODEL_DIR,
)


__all__ = [
    # Main classes
    "ModelRunner",
    "ModelConfig",
    "SlotConfig",
    "Backend",
    # Backend protocol
    "ModelBackend",
    "BackendConfig",
    "BackendResult",
    # Exceptions
    "ModelRunnerError",
    "UnknownSlotError",
    "ModelBackendError",
    "BackendNotAvailableError",
    "BackendError",
    # Backend implementations
    "TransformersBackend",
    "OpenAIBackend",
    # Backend utilities
    "is_backend_available",
    "get_available_backends",
    # Tokenization
    "estimate_tokens",
    "estimate_tokens_simple",
    "estimate_tokens_hybrid",
    "estimate_prompt_tokens",
    "count_tokens_tiktoken",
    "is_tiktoken_available",
    "get_tokenizer",
    "register_tokenizer",
    # Cost estimation
    "estimate_cost",
    "estimate_query_cost",
    "get_model_pricing",
    "ModelPricing",
    "CostMetrics",
    "estimate_flops",
    "get_model_size",
    # Profile management
    "ProfileManager",
    "ProfileMode",
    "ProfileDefinition",
    "HardwareCapability",
    "EnvironmentInfo",
    "detect_environment",
    "get_default_profile",
    "get_profile_manager",
    "quick_runner",
    "BUILTIN_PROFILES",
    # Performance optimizations (Task 7)
    "ResponseCache",
    "PersistentCache",
    "RequestBatcher",
    "ConnectionPool",
    "RateLimiter",
    "OptimizedRunner",
    # Reliability (Task 8)
    "RetryStrategy",
    "RetryPolicy",
    "TimeoutConfig",
    "TimeoutManager",
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitOpenError",
    "FallbackChain",
    "HealthStatus",
    "HealthChecker",
    "ResilientBackend",
    # Model Download Manager
    "ModelManager",
    "ModelRegistry",
    "ModelInfo",
    "ModelSource",
    "ModelFormat",
    "HuggingFaceDownloader",
    "OllamaManager",
    "LMStudioManager",
    "DownloadProgress",
    "DEFAULT_MODEL_DIR",
]
