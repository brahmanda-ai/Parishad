"""
Reliability and error handling for Parishad model inference.

Provides:
- RetryPolicy: Configurable retry with exponential backoff
- TimeoutManager: Request timeout handling
- CircuitBreaker: Fail-fast when backend is unhealthy
- FallbackChain: Try multiple backends in sequence
- HealthChecker: Backend health monitoring

These components ensure robust operation even with unreliable backends.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Generic, Optional, TypeVar

from .backends import BackendError, BackendResult, ModelBackend


logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Retry Policy
# =============================================================================


class RetryStrategy(Enum):
    """Retry backoff strategies."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


@dataclass
class RetryPolicy:
    """
    Configurable retry policy with backoff.
    
    Supports fixed, linear, and exponential backoff strategies
    with optional jitter to prevent thundering herd.
    
    Usage:
        policy = RetryPolicy(max_retries=3, strategy=RetryStrategy.EXPONENTIAL)
        
        @policy.wrap
        def make_request():
            return api.call()
        
        # Or manually:
        for attempt in policy.attempts():
            try:
                return make_request()
            except Exception as e:
                if not policy.should_retry(e, attempt):
                    raise
    """
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.1
    
    # Exception types to retry on
    retryable_exceptions: tuple = field(default_factory=lambda: (
        ConnectionError,
        TimeoutError,
        BackendError,
    ))
    
    # Error messages to retry on
    retryable_messages: list[str] = field(default_factory=lambda: [
        "rate limit",
        "overloaded",
        "temporarily unavailable",
        "server error",
        "502",
        "503",
        "504",
    ])
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for attempt number.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)
        
        elif self.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            delay = self.base_delay * (2 ** attempt)
            jitter = delay * self.jitter_factor * random.random()
            delay += jitter
        
        else:
            delay = self.base_delay
        
        return min(delay, self.max_delay)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Check if exception should trigger a retry.
        
        Args:
            exception: The exception that occurred
            attempt: Current attempt number
            
        Returns:
            True if should retry
        """
        if attempt >= self.max_retries:
            return False
        
        # Check exception type
        if isinstance(exception, self.retryable_exceptions):
            return True
        
        # Check error message
        error_msg = str(exception).lower()
        for pattern in self.retryable_messages:
            if pattern.lower() in error_msg:
                return True
        
        return False
    
    def attempts(self):
        """
        Generator yielding attempt numbers.
        
        Usage:
            for attempt in policy.attempts():
                try:
                    return make_request()
                except Exception as e:
                    if not policy.should_retry(e, attempt):
                        raise
                    time.sleep(policy.get_delay(attempt))
        """
        for attempt in range(self.max_retries + 1):
            yield attempt
    
    def wrap(self, func: F) -> F:
        """
        Decorator to apply retry policy to a function.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function with retry logic
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in self.attempts():
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not self.should_retry(e, attempt):
                        raise
                    
                    delay = self.get_delay(attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.max_retries} "
                        f"after {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
            
            raise last_exception  # type: ignore
        
        return wrapper  # type: ignore


# =============================================================================
# Timeout Manager
# =============================================================================


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


@dataclass
class TimeoutConfig:
    """Timeout configuration."""
    connect_timeout: float = 10.0      # Connection timeout
    read_timeout: float = 60.0         # Read/response timeout
    total_timeout: float = 120.0       # Total request timeout
    
    def as_tuple(self) -> tuple[float, float]:
        """Return as (connect, read) tuple for requests library."""
        return (self.connect_timeout, self.read_timeout)


class TimeoutManager:
    """
    Manages request timeouts.
    
    Provides context manager for enforcing timeouts on operations.
    
    Usage:
        manager = TimeoutManager(total_timeout=30.0)
        
        with manager.timeout():
            result = slow_operation()
    """
    
    def __init__(self, config: Optional[TimeoutConfig] = None):
        """Initialize with timeout configuration."""
        self.config = config or TimeoutConfig()
    
    def timeout(self, seconds: Optional[float] = None):
        """
        Context manager for timeout enforcement.
        
        Note: This is a basic implementation. For true timeout enforcement
        in synchronous code, consider using signals or threading.
        """
        timeout_seconds = seconds or self.config.total_timeout
        return _TimeoutContext(timeout_seconds)
    
    def with_timeout(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with timeout.
        
        Uses threading for timeout enforcement.
        """
        result: list[T] = []
        exception: list[Exception] = []
        
        def target():
            try:
                result.append(func(*args, **kwargs))
            except Exception as e:
                exception.append(e)
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self.config.total_timeout)
        
        if thread.is_alive():
            # Thread is still running - timeout occurred
            raise TimeoutError(
                f"Operation timed out after {self.config.total_timeout}s"
            )
        
        if exception:
            raise exception[0]
        
        return result[0]


class _TimeoutContext:
    """Context manager for basic timeout tracking."""
    
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout:
            logger.warning(f"Operation took {elapsed:.1f}s (timeout: {self.timeout}s)")
        return False
    
    @property
    def remaining(self) -> float:
        """Get remaining time."""
        elapsed = time.time() - self.start_time
        return max(0, self.timeout - elapsed)


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5        # Failures before opening
    success_threshold: int = 2        # Successes before closing
    timeout: float = 30.0             # Seconds before half-open
    half_open_max_calls: int = 3      # Max calls in half-open state


class CircuitBreaker:
    """
    Circuit breaker for fail-fast behavior.
    
    When a backend fails repeatedly, the circuit opens and fails fast
    instead of waiting for timeouts. After a cooldown, it tests the
    backend again before fully recovering.
    
    Usage:
        breaker = CircuitBreaker()
        
        @breaker.protect
        def call_backend():
            return backend.generate(prompt)
        
        try:
            result = call_backend()
        except CircuitOpenError:
            # Circuit is open, use fallback
            result = fallback()
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker."""
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.config.timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._success_count = 0
                    logger.info("Circuit breaker entering half-open state")
            
            return self._state
    
    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit breaker closed after recovery")
            else:
                self._failure_count = 0
    
    def record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker re-opened: {exception}")
            
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker opened after {self._failure_count} failures"
                )
    
    def allow_request(self) -> bool:
        """Check if a request is allowed."""
        state = self.state  # This may update state
        
        if state == CircuitState.CLOSED:
            return True
        
        if state == CircuitState.OPEN:
            return False
        
        # HALF_OPEN - allow limited calls
        with self._lock:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
    
    def protect(self, func: F) -> F:
        """
        Decorator to protect a function with circuit breaker.
        
        Args:
            func: Function to protect
            
        Returns:
            Protected function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.allow_request():
                raise CircuitOpenError(
                    f"Circuit breaker is {self.state.value}"
                )
            
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise
        
        return wrapper  # type: ignore
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": self._last_failure_time,
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# Fallback Chain
# =============================================================================


class FallbackChain:
    """
    Chain of backends with automatic fallback.
    
    Tries backends in order until one succeeds. Useful for having
    primary/secondary/tertiary backend configurations.
    
    Usage:
        chain = FallbackChain([
            primary_backend,
            secondary_backend,
            stub_backend,
        ])
        
        result = chain.generate(prompt, config)
    """
    
    def __init__(
        self,
        backends: list[ModelBackend],
        circuit_breaker_enabled: bool = True,
    ):
        """
        Initialize fallback chain.
        
        Args:
            backends: List of backends in priority order
            circuit_breaker_enabled: Use circuit breakers per backend
        """
        self.backends = backends
        
        self._circuit_breakers: dict[int, CircuitBreaker] = {}
        if circuit_breaker_enabled:
            for i in range(len(backends)):
                self._circuit_breakers[i] = CircuitBreaker()
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.5,
        **kwargs,
    ) -> tuple[BackendResult, int]:
        """
        Generate using fallback chain.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (BackendResult, backend_index)
            
        Raises:
            BackendError: If all backends fail
        """
        last_error: Optional[Exception] = None
        
        for i, backend in enumerate(self.backends):
            # Check circuit breaker
            if i in self._circuit_breakers:
                if not self._circuit_breakers[i].allow_request():
                    logger.debug(f"Skipping backend {i}: circuit open")
                    continue
            
            try:
                result = backend.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                
                if i in self._circuit_breakers:
                    self._circuit_breakers[i].record_success()
                
                if i > 0:
                    logger.info(f"Using fallback backend {i}")
                
                return result, i
                
            except Exception as e:
                last_error = e
                logger.warning(f"Backend {i} failed: {e}")
                
                if i in self._circuit_breakers:
                    self._circuit_breakers[i].record_failure(e)
        
        raise BackendError(f"All backends failed. Last error: {last_error}")
    
    def get_stats(self) -> dict:
        """Get chain statistics."""
        return {
            "backends": len(self.backends),
            "circuit_breakers": {
                i: cb.get_stats()
                for i, cb in self._circuit_breakers.items()
            },
        }


# =============================================================================
# Health Checker
# =============================================================================


@dataclass
class HealthStatus:
    """Health status of a backend."""
    healthy: bool
    latency_ms: float
    error: Optional[str] = None
    checked_at: float = field(default_factory=time.time)


class HealthChecker:
    """
    Backend health monitoring.
    
    Periodically checks backend health and tracks metrics.
    
    Usage:
        checker = HealthChecker(backend)
        
        # One-time check
        status = checker.check()
        
        # Start background monitoring
        checker.start_monitoring(interval=30)
    """
    
    def __init__(
        self,
        backend: ModelBackend,
        test_prompt: str = "Hello",
    ):
        """
        Initialize health checker.
        
        Args:
            backend: Backend to monitor
            test_prompt: Prompt for health checks
        """
        self.backend = backend
        self.test_prompt = test_prompt
        
        self._history: list[HealthStatus] = []
        self._max_history = 100
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def check(self) -> HealthStatus:
        """
        Perform a health check.
        
        Returns:
            HealthStatus with results
        """
        start = time.time()
        
        try:
            result = self.backend.generate(
                prompt=self.test_prompt,
                max_tokens=5,
                temperature=0,
            )
            
            latency_ms = (time.time() - start) * 1000
            
            status = HealthStatus(
                healthy=True,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            
            status = HealthStatus(
                healthy=False,
                latency_ms=latency_ms,
                error=str(e),
            )
        
        self._record(status)
        return status
    
    def _record(self, status: HealthStatus) -> None:
        """Record health status."""
        self._history.append(status)
        
        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
    
    def start_monitoring(self, interval: float = 30.0) -> None:
        """Start background health monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                try:
                    self.check()
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._monitoring = False
    
    @property
    def is_healthy(self) -> bool:
        """Check if backend is currently healthy."""
        if not self._history:
            return True  # Assume healthy if no data
        
        # Check last 3 statuses
        recent = self._history[-3:]
        healthy_count = sum(1 for s in recent if s.healthy)
        
        return healthy_count >= 2
    
    @property
    def avg_latency(self) -> float:
        """Average latency over recent history."""
        if not self._history:
            return 0.0
        
        recent = self._history[-10:]
        return sum(s.latency_ms for s in recent) / len(recent)
    
    def get_stats(self) -> dict:
        """Get health statistics."""
        if not self._history:
            return {"status": "unknown", "checks": 0}
        
        recent = self._history[-10:]
        
        return {
            "status": "healthy" if self.is_healthy else "unhealthy",
            "checks": len(self._history),
            "recent_healthy": sum(1 for s in recent if s.healthy),
            "recent_total": len(recent),
            "avg_latency_ms": self.avg_latency,
            "last_check": self._history[-1].checked_at,
            "last_error": next(
                (s.error for s in reversed(self._history) if s.error),
                None
            ),
        }


# =============================================================================
# Resilient Backend Wrapper
# =============================================================================


class ResilientBackend:
    """
    Wrapper that adds all reliability features to a backend.
    
    Combines retry, timeout, circuit breaker, and health checking.
    
    Usage:
        backend = LlamaCppBackend()
        resilient = ResilientBackend(backend)
        
        result = resilient.generate(prompt, max_tokens=100)
    """
    
    def __init__(
        self,
        backend: ModelBackend,
        retry_policy: Optional[RetryPolicy] = None,
        timeout_config: Optional[TimeoutConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        enable_health_check: bool = True,
    ):
        """
        Initialize resilient backend.
        
        Args:
            backend: Base backend to wrap
            retry_policy: Retry configuration
            timeout_config: Timeout configuration
            circuit_config: Circuit breaker configuration
            enable_health_check: Enable health monitoring
        """
        self.backend = backend
        self.retry_policy = retry_policy or RetryPolicy()
        self.timeout_manager = TimeoutManager(timeout_config)
        self.circuit_breaker = CircuitBreaker(circuit_config)
        
        self._health_checker: Optional[HealthChecker] = None
        if enable_health_check:
            self._health_checker = HealthChecker(backend)
    
    @property
    def name(self) -> str:
        """Backend name."""
        return f"resilient({self.backend.name})"
    
    def load(self, config) -> None:
        """Load model."""
        self.backend.load(config)
    
    def unload(self) -> None:
        """Unload model."""
        self.backend.unload()
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.5,
        **kwargs,
    ) -> BackendResult:
        """
        Generate with all reliability features.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            BackendResult from generation
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise CircuitOpenError("Circuit breaker is open")
        
        last_error: Optional[Exception] = None
        
        for attempt in self.retry_policy.attempts():
            try:
                # Apply timeout
                with self.timeout_manager.timeout():
                    result = self.backend.generate(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs,
                    )
                
                self.circuit_breaker.record_success()
                return result
                
            except Exception as e:
                last_error = e
                
                if not self.retry_policy.should_retry(e, attempt):
                    self.circuit_breaker.record_failure(e)
                    raise
                
                delay = self.retry_policy.get_delay(attempt)
                logger.warning(f"Retry {attempt + 1}: {e}, waiting {delay:.1f}s")
                time.sleep(delay)
        
        self.circuit_breaker.record_failure(last_error)  # type: ignore
        raise last_error  # type: ignore
    
    def get_stats(self) -> dict:
        """Get reliability statistics."""
        stats = {
            "circuit_breaker": self.circuit_breaker.get_stats(),
        }
        if self._health_checker:
            stats["health"] = self._health_checker.get_stats()
        return stats


__all__ = [
    # Retry
    "RetryStrategy",
    "RetryPolicy",
    # Timeout
    "TimeoutError",
    "TimeoutConfig",
    "TimeoutManager",
    # Circuit breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitOpenError",
    # Fallback
    "FallbackChain",
    # Health check
    "HealthStatus",
    "HealthChecker",
    # Combined
    "ResilientBackend",
]
