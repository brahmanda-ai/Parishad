"""Text processing utilities for Parishad."""

from __future__ import annotations


def truncate_text(
    text: str,
    max_chars: int,
    suffix: str = "... [TRUNCATED]"
) -> str:
    """
    Truncate text to a maximum character count with suffix.
    
    Args:
        text: Text to truncate
        max_chars: Maximum character count
        suffix: Suffix to append when truncating (default: "... [TRUNCATED]")
    
    Returns:
        Original text if under limit, or truncated text with suffix
    
    Examples:
        >>> truncate_text("Hello world", 20)
        'Hello world'
        >>> truncate_text("A" * 100, 10)
        'AAAAAAAAAA... [TRUNCATED]'
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix


def truncate_with_note(
    text: str,
    max_chars: int,
    label: str = "content"
) -> tuple[str, bool]:
    """
    Truncate text and return whether truncation occurred.
    
    Args:
        text: Text to truncate
        max_chars: Maximum character count
        label: Label for the truncation note
    
    Returns:
        Tuple of (truncated_text, was_truncated)
    
    Examples:
        >>> text, truncated = truncate_with_note("short", 100)
        >>> truncated
        False
        >>> text, truncated = truncate_with_note("A" * 200, 50, "worker")
        >>> truncated
        True
    """
    if len(text) <= max_chars:
        return text, False
    
    truncated = text[:max_chars] + f"\n... [{label} truncated: {len(text)} chars total]"
    return truncated, True
