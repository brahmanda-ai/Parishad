"""
Performance optimizations for Parishad model inference.

Provides:
- ResponseCache: LRU cache for model responses
- RequestBatcher: Batch multiple requests for efficiency
- ConnectionPool: Reuse backend connections
- RateLimiter: Token bucket rate limiting

These optimizations are optional and can be enabled via configuration.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, Optional, TypeVar
from contextlib import contextmanager

from .backends import BackendConfig, BackendResult, ModelBackend


logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Response Cache
# =============================================================================


@dataclass
class CacheEntry:
    """Entry in the response cache."""
    key: str
    response: BackendResult
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    @property
    def age_seconds(self) -> float:
        """Age of the entry in seconds."""
        return time.time() - self.created_at


class ResponseCache:
    """
    LRU cache for model responses.
    
    Caches responses based on prompt hash to avoid redundant model calls.
    Thread-safe for concurrent access.
    
    Usage:
        cache = ResponseCache(max_size=1000, ttl_seconds=3600)
        
        key = cache.make_key(prompt, model_id, temperature)
        if cached := cache.get(key):
            return cached
        
        result = model.generate(prompt)
        cache.put(key, result)
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600,
        enabled: bool = True,
    ):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries
            enabled: Whether caching is enabled
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
    
    def make_key(
        self,
        prompt: str,
        model_id: str,
        temperature: float = 0.0,
        max_tokens: int = 0,
        **kwargs,
    ) -> str:
        """
        Create a cache key from request parameters.
        
        Note: Only caches deterministic requests (temperature=0).
        """
        # Only cache deterministic requests
        if temperature > 0.01:
            return ""  # Empty key means don't cache
        
        key_data = json.dumps({
            "prompt": prompt,
            "model_id": model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }, sort_keys=True)
        
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def get(self, key: str) -> Optional[BackendResult]:
        """
        Get cached response.
        
        Args:
            key: Cache key
            
        Returns:
            Cached BackendResult or None
        """
        if not self.enabled or not key:
            return None
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            # Check TTL
            if entry.age_seconds > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Update access stats and move to end (LRU)
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._cache.move_to_end(key)
            
            self._hits += 1
            return entry.response
    
    def put(self, key: str, response: BackendResult) -> None:
        """
        Store response in cache.
        
        Args:
            key: Cache key
            response: Response to cache
        """
        if not self.enabled or not key:
            return
        
        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = CacheEntry(
                key=key,
                response=response,
                created_at=time.time(),
            )
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> int:
        """Clear all cache entries. Returns count cleared."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "enabled": self.enabled,
        }


class PersistentCache(ResponseCache):
    """
    SQLite-backed persistent cache.
    
    Survives process restarts. Uses same interface as ResponseCache.
    """
    
    def __init__(
        self,
        path: str | Path,
        max_size: int = 10000,
        ttl_seconds: float = 86400,  # 24 hours
        enabled: bool = True,
    ):
        super().__init__(max_size=max_size, ttl_seconds=ttl_seconds, enabled=enabled)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    response_json TEXT,
                    created_at REAL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at)")
    
    @contextmanager
    def _get_conn(self):
        """Get database connection."""
        conn = sqlite3.connect(str(self.path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def get(self, key: str) -> Optional[BackendResult]:
        """Get from persistent cache."""
        if not self.enabled or not key:
            return None
        
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT response_json, created_at FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if not row:
                self._misses += 1
                return None
            
            response_json, created_at = row
            
            # Check TTL
            if time.time() - created_at > self.ttl_seconds:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._misses += 1
                return None
            
            # Update access stats
            conn.execute(
                "UPDATE cache SET access_count = access_count + 1, last_accessed = ? WHERE key = ?",
                (time.time(), key)
            )
            
            self._hits += 1
            data = json.loads(response_json)
            return BackendResult(**data)
    
    def put(self, key: str, response: BackendResult) -> None:
        """Store in persistent cache."""
        if not self.enabled or not key:
            return
        
        with self._get_conn() as conn:
            # Evict old entries if needed
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]
            
            if count >= self.max_size:
                # Delete oldest 10%
                delete_count = max(1, self.max_size // 10)
                conn.execute(
                    "DELETE FROM cache WHERE key IN (SELECT key FROM cache ORDER BY last_accessed LIMIT ?)",
                    (delete_count,)
                )
            
            response_json = json.dumps({
                "text": response.text,
                "tokens_in": response.tokens_in,
                "tokens_out": response.tokens_out,
                "model_id": response.model_id,
                "latency_ms": response.latency_ms,
            })
            
            conn.execute(
                """INSERT OR REPLACE INTO cache 
                   (key, response_json, created_at, access_count, last_accessed)
                   VALUES (?, ?, ?, 0, ?)""",
                (key, response_json, time.time(), time.time())
            )
    
    def clear(self) -> int:
        """Clear all entries."""
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]
            conn.execute("DELETE FROM cache")
            return count


# =============================================================================
# Request Batcher
# =============================================================================


@dataclass
class BatchRequest:
    """A request in the batch queue."""
    prompt: str
    config: BackendConfig
    future: "asyncio.Future[BackendResult]"
    submitted_at: float = field(default_factory=time.time)


class RequestBatcher:
    """
    Batch multiple requests for efficient processing.
    
    Collects requests over a time window and processes them together.
    Useful for backends that support batch inference.
    
    Usage:
        batcher = RequestBatcher(backend, batch_size=8, wait_ms=50)
        result = await batcher.submit(prompt, config)
    """
    
    def __init__(
        self,
        backend: ModelBackend,
        batch_size: int = 8,
        wait_ms: float = 50.0,
        enabled: bool = True,
    ):
        """
        Initialize batcher.
        
        Args:
            backend: Backend to use for generation
            batch_size: Maximum batch size
            wait_ms: Maximum wait time before processing
            enabled: Whether batching is enabled
        """
        self.backend = backend
        self.batch_size = batch_size
        self.wait_ms = wait_ms
        self.enabled = enabled
        
        self._queue: list[BatchRequest] = []
        self._lock = threading.Lock()
        self._processing = False
        
        # Statistics
        self._batches_processed = 0
        self._requests_processed = 0
    
    async def submit(
        self,
        prompt: str,
        config: BackendConfig,
    ) -> BackendResult:
        """
        Submit a request for batched processing.
        
        Args:
            prompt: Input prompt
            config: Backend configuration
            
        Returns:
            BackendResult from generation
        """
        if not self.enabled:
            # Direct processing if batching disabled
            return self.backend.generate(
                prompt=prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop,
            )
        
        loop = asyncio.get_event_loop()
        future: asyncio.Future[BackendResult] = loop.create_future()
        
        request = BatchRequest(prompt=prompt, config=config, future=future)
        
        with self._lock:
            self._queue.append(request)
            
            if len(self._queue) >= self.batch_size:
                # Process immediately if batch is full
                self._schedule_processing()
            elif len(self._queue) == 1:
                # Schedule delayed processing
                loop.call_later(self.wait_ms / 1000, self._schedule_processing)
        
        return await future
    
    def _schedule_processing(self) -> None:
        """Schedule batch processing."""
        with self._lock:
            if self._processing or not self._queue:
                return
            
            self._processing = True
            batch = self._queue[:self.batch_size]
            self._queue = self._queue[self.batch_size:]
        
        # Process in thread pool
        try:
            self._process_batch(batch)
        finally:
            with self._lock:
                self._processing = False
    
    def _process_batch(self, batch: list[BatchRequest]) -> None:
        """Process a batch of requests."""
        for request in batch:
            try:
                result = self.backend.generate(
                    prompt=request.prompt,
                    max_tokens=request.config.max_tokens,
                    temperature=request.config.temperature,
                    top_p=request.config.top_p,
                    stop=request.config.stop,
                )
                
                if not request.future.done():
                    request.future.get_loop().call_soon_threadsafe(
                        request.future.set_result, result
                    )
                
            except Exception as e:
                if not request.future.done():
                    request.future.get_loop().call_soon_threadsafe(
                        request.future.set_exception, e
                    )
        
        self._batches_processed += 1
        self._requests_processed += len(batch)
    
    def get_stats(self) -> dict:
        """Get batcher statistics."""
        return {
            "batches_processed": self._batches_processed,
            "requests_processed": self._requests_processed,
            "avg_batch_size": (
                self._requests_processed / self._batches_processed
                if self._batches_processed > 0 else 0
            ),
            "queue_size": len(self._queue),
            "enabled": self.enabled,
        }


# =============================================================================
# Connection Pool
# =============================================================================


class ConnectionPool:
    """
    Pool of reusable backend connections.
    
    Reduces overhead of creating new connections for each request.
    Thread-safe for concurrent access.
    
    Usage:
        pool = ConnectionPool(backend_factory, max_size=4)
        
        with pool.acquire() as backend:
            result = backend.generate(prompt)
    """
    
    def __init__(
        self,
        backend_factory: Callable[[], ModelBackend],
        max_size: int = 4,
        min_size: int = 1,
    ):
        """
        Initialize pool.
        
        Args:
            backend_factory: Factory function to create backends
            max_size: Maximum pool size
            min_size: Minimum backends to keep ready
        """
        self.backend_factory = backend_factory
        self.max_size = max_size
        self.min_size = min_size
        
        self._available: Queue[ModelBackend] = Queue()
        self._in_use: set[int] = set()
        self._lock = threading.Lock()
        self._total_created = 0
        
        # Pre-create minimum backends
        for _ in range(min_size):
            self._create_backend()
    
    def _create_backend(self) -> ModelBackend:
        """Create a new backend instance."""
        backend = self.backend_factory()
        self._available.put(backend)
        self._total_created += 1
        return backend
    
    @contextmanager
    def acquire(self, timeout: float = 30.0):
        """
        Acquire a backend from the pool.
        
        Args:
            timeout: Maximum time to wait for a backend
            
        Yields:
            ModelBackend instance
        """
        backend = None
        
        try:
            # Try to get from available
            try:
                backend = self._available.get(timeout=timeout)
            except Empty:
                # Create new if under limit
                with self._lock:
                    current_size = self._total_created
                    if current_size < self.max_size:
                        backend = self.backend_factory()
                        self._total_created += 1
                    else:
                        raise TimeoutError("No backends available in pool")
            
            with self._lock:
                self._in_use.add(id(backend))
            
            yield backend
            
        finally:
            if backend is not None:
                with self._lock:
                    self._in_use.discard(id(backend))
                self._available.put(backend)
    
    def get_stats(self) -> dict:
        """Get pool statistics."""
        return {
            "total_created": self._total_created,
            "available": self._available.qsize(),
            "in_use": len(self._in_use),
            "max_size": self.max_size,
        }


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Controls request rate to avoid overwhelming backends or hitting API limits.
    
    Usage:
        limiter = RateLimiter(tokens_per_second=10, burst_size=20)
        
        await limiter.acquire()  # Blocks until token available
        result = model.generate(prompt)
    """
    
    def __init__(
        self,
        tokens_per_second: float = 10.0,
        burst_size: int = 20,
    ):
        """
        Initialize rate limiter.
        
        Args:
            tokens_per_second: Token refill rate
            burst_size: Maximum tokens (burst capacity)
        """
        self.tokens_per_second = tokens_per_second
        self.burst_size = burst_size
        
        self._tokens = float(burst_size)
        self._last_refill = time.time()
        self._lock = threading.Lock()
        
        # Statistics
        self._requests = 0
        self._waits = 0
        self._total_wait_time = 0.0
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.burst_size,
            self._tokens + elapsed * self.tokens_per_second
        )
        self._last_refill = now
    
    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, blocking if necessary.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Wait time in seconds
        """
        wait_time = 0.0
        
        with self._lock:
            self._refill()
            
            while self._tokens < tokens:
                # Calculate wait time
                needed = tokens - self._tokens
                wait = needed / self.tokens_per_second
                wait_time += wait
                
                self._lock.release()
                time.sleep(wait)
                self._lock.acquire()
                
                self._refill()
            
            self._tokens -= tokens
            self._requests += 1
            
            if wait_time > 0:
                self._waits += 1
                self._total_wait_time += wait_time
        
        return wait_time
    
    async def acquire_async(self, tokens: int = 1) -> float:
        """Async version of acquire."""
        wait_time = 0.0
        
        with self._lock:
            self._refill()
            
            if self._tokens < tokens:
                needed = tokens - self._tokens
                wait_time = needed / self.tokens_per_second
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            self._waits += 1
            self._total_wait_time += wait_time
        
        with self._lock:
            self._refill()
            self._tokens -= tokens
            self._requests += 1
        
        return wait_time
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "requests": self._requests,
            "waits": self._waits,
            "total_wait_time": self._total_wait_time,
            "avg_wait_time": (
                self._total_wait_time / self._waits if self._waits > 0 else 0
            ),
            "current_tokens": self._tokens,
            "tokens_per_second": self.tokens_per_second,
        }


# =============================================================================
# Optimized Runner Wrapper
# =============================================================================


class OptimizedRunner:
    """
    Wrapper that adds caching, batching, and rate limiting to a ModelRunner.
    
    Usage:
        from parishad.models.runner import ModelRunner
        
        runner = ModelRunner(stub=True)
        optimized = OptimizedRunner(
            runner,
            cache_enabled=True,
            rate_limit=10.0,
        )
        
        text, tokens, model = optimized.generate(
            system_prompt="You are helpful.",
            user_message="Hello!",
            slot=Slot.SMALL,
        )
    """
    
    def __init__(
        self,
        runner: "ModelRunner",  # type: ignore
        cache_enabled: bool = False,
        cache_max_size: int = 1000,
        cache_ttl: float = 3600,
        rate_limit: Optional[float] = None,
        rate_burst: int = 20,
    ):
        """
        Initialize optimized runner.
        
        Args:
            runner: Base ModelRunner to wrap
            cache_enabled: Enable response caching
            cache_max_size: Maximum cache entries
            cache_ttl: Cache TTL in seconds
            rate_limit: Rate limit (requests per second)
            rate_burst: Rate limit burst size
        """
        self.runner = runner
        
        self.cache = ResponseCache(
            max_size=cache_max_size,
            ttl_seconds=cache_ttl,
            enabled=cache_enabled,
        )
        
        self.rate_limiter: Optional[RateLimiter] = None
        if rate_limit is not None:
            self.rate_limiter = RateLimiter(
                tokens_per_second=rate_limit,
                burst_size=rate_burst,
            )
    
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        slot: "Slot",  # type: ignore
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> tuple[str, int, str]:
        """
        Generate with optimizations applied.
        
        Args:
            system_prompt: System prompt
            user_message: User message
            slot: Model slot
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (text, tokens, model_id)
        """
        # Build cache key
        prompt = f"{system_prompt}\n{user_message}"
        cache_key = self.cache.make_key(
            prompt=prompt,
            model_id=slot.value,
            temperature=temperature or 0.0,
            max_tokens=max_tokens or 0,
        )
        
        # Check cache
        if cached := self.cache.get(cache_key):
            logger.debug("Cache hit for request")
            return cached.text, cached.tokens_in + cached.tokens_out, cached.model_id
        
        # Apply rate limiting
        if self.rate_limiter:
            self.rate_limiter.acquire()
        
        # Generate
        text, tokens, model_id = self.runner.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            slot=slot,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        
        # Cache result
        from .backends.base import BackendResult
        result = BackendResult(
            text=text,
            tokens_in=tokens // 2,  # Approximate
            tokens_out=tokens - tokens // 2,
            model_id=model_id,
            latency_ms=0,
        )
        self.cache.put(cache_key, result)
        
        return text, tokens, model_id
    
    def get_stats(self) -> dict:
        """Get optimization statistics."""
        stats = {
            "cache": self.cache.get_stats(),
        }
        if self.rate_limiter:
            stats["rate_limiter"] = self.rate_limiter.get_stats()
        return stats


__all__ = [
    # Cache
    "CacheEntry",
    "ResponseCache",
    "PersistentCache",
    # Batching
    "BatchRequest",
    "RequestBatcher",
    # Connection pool
    "ConnectionPool",
    # Rate limiting
    "RateLimiter",
    # Optimized wrapper
    "OptimizedRunner",
]
