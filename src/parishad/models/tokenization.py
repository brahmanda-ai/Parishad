"""
Tokenization utilities for Parishad.

Provides token estimation for different backends and models.
This is used for:
- Budget enforcement (tracking token usage)
- Cost estimation
- Context length management
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable


# =============================================================================
# Heuristic Token Estimators
# =============================================================================


def estimate_tokens_simple(text: str) -> int:
    """
    Simple heuristic token estimation using word count.
    
    Uses ~1.3 tokens per word as a rough approximation for English text.
    This is fast but not accurate for code or non-English text.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.3)


def estimate_tokens_chars(text: str) -> int:
    """
    Character-based token estimation.
    
    Uses ~4 characters per token as a rough approximation.
    Better for code and mixed content.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_tokens_hybrid(text: str) -> int:
    """
    Hybrid token estimation combining word and character counts.
    
    Uses a weighted combination for better accuracy across
    different content types (prose vs code).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Count words and characters
    words = len(text.split())
    chars = len(text)
    
    # Count code-like patterns (more tokens per character in code)
    code_patterns = len(re.findall(r'[{}()\[\];:,<>=!&|+\-*/]', text))
    
    # Base estimate from words
    word_estimate = int(words * 1.3)
    
    # Character-based estimate
    char_estimate = chars // 4
    
    # If lots of code patterns, weight towards character estimate
    if code_patterns > words * 0.3:
        # Code-heavy: use character estimate
        return max(1, int(char_estimate * 1.1))
    else:
        # Prose-heavy: average of both
        return max(1, (word_estimate + char_estimate) // 2)


# =============================================================================
# Tokenizer Registry
# =============================================================================


# Map of backend/model to tokenizer function
_TOKENIZER_REGISTRY: dict[str, Callable[[str], int]] = {}


def register_tokenizer(
    name: str,
    tokenizer_fn: Callable[[str], int],
) -> None:
    """
    Register a tokenizer function for a backend or model.
    
    Args:
        name: Backend name or model ID
        tokenizer_fn: Function that takes text and returns token count
    """
    _TOKENIZER_REGISTRY[name] = tokenizer_fn


def get_tokenizer(backend: str, model_id: str = "") -> Callable[[str], int]:
    """
    Get the best tokenizer for a backend/model.
    
    Looks up in order:
    1. Exact model_id match
    2. Backend name match
    3. Default hybrid estimator
    
    Args:
        backend: Backend name (e.g., 'openai', 'llama_cpp')
        model_id: Optional model identifier
        
    Returns:
        Tokenizer function
    """
    # Try model-specific tokenizer
    if model_id and model_id in _TOKENIZER_REGISTRY:
        return _TOKENIZER_REGISTRY[model_id]
    
    # Try backend tokenizer
    if backend in _TOKENIZER_REGISTRY:
        return _TOKENIZER_REGISTRY[backend]
    
    # Default
    return estimate_tokens_hybrid


# =============================================================================
# Tiktoken Integration (for OpenAI models)
# =============================================================================


_tiktoken = None


def _get_tiktoken():
    """Lazy import of tiktoken."""
    global _tiktoken
    if _tiktoken is None:
        try:
            import tiktoken
            _tiktoken = tiktoken
        except ImportError:
            return None
    return _tiktoken


@lru_cache(maxsize=8)
def _get_tiktoken_encoding(model: str):
    """Get tiktoken encoding for a model (cached)."""
    tiktoken = _get_tiktoken()
    if tiktoken is None:
        return None
    
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base for unknown models
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens_tiktoken(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens using tiktoken (for OpenAI models).
    
    Falls back to heuristic if tiktoken unavailable.
    
    Args:
        text: Input text
        model: OpenAI model name
        
    Returns:
        Token count
    """
    if not text:
        return 0
    
    encoding = _get_tiktoken_encoding(model)
    if encoding is None:
        return estimate_tokens_hybrid(text)
    
    return len(encoding.encode(text))


def is_tiktoken_available() -> bool:
    """Check if tiktoken is available."""
    return _get_tiktoken() is not None


# =============================================================================
# Register Default Tokenizers
# =============================================================================


# OpenAI models use tiktoken when available
def _openai_tokenizer(text: str) -> int:
    return count_tokens_tiktoken(text, "gpt-4")


register_tokenizer("openai", _openai_tokenizer)

# Other backends use hybrid by default
register_tokenizer("llama_cpp", estimate_tokens_hybrid)
register_tokenizer("transformers", estimate_tokens_hybrid)


# =============================================================================
# Convenience Functions
# =============================================================================


def estimate_tokens(
    text: str,
    backend: str = "",
    model_id: str = "",
) -> int:
    """
    Estimate token count for text.
    
    Uses the best available tokenizer for the backend/model.
    
    Args:
        text: Input text
        backend: Optional backend name
        model_id: Optional model identifier
        
    Returns:
        Estimated token count
    """
    tokenizer = get_tokenizer(backend, model_id)
    return tokenizer(text)


def estimate_prompt_tokens(
    system_prompt: str,
    user_message: str,
    backend: str = "",
    model_id: str = "",
) -> int:
    """
    Estimate tokens for a full prompt (system + user).
    
    Accounts for message formatting overhead.
    
    Args:
        system_prompt: System prompt text
        user_message: User message text
        backend: Optional backend name
        model_id: Optional model identifier
        
    Returns:
        Estimated token count including overhead
    """
    tokenizer = get_tokenizer(backend, model_id)
    
    # Count tokens in each part
    system_tokens = tokenizer(system_prompt)
    user_tokens = tokenizer(user_message)
    
    # Add overhead for message formatting (~4 tokens per message)
    overhead = 8  # system + user messages
    
    return system_tokens + user_tokens + overhead
