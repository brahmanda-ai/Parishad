"""Logging configuration for Parishad."""

from __future__ import annotations

import logging
import sys
from typing import Optional


# Default truncation limits for memory-efficient logging
DEFAULT_TRUNCATE_LENGTH = 512
MAX_TRUNCATE_LENGTH = 2048


def truncate_for_log(text: str, max_length: int = DEFAULT_TRUNCATE_LENGTH) -> str:
    """
    Truncate a string for logging to avoid large memory allocations.
    
    Args:
        text: The text to truncate
        max_length: Maximum length before truncation (default: 512)
        
    Returns:
        Truncated string with ellipsis if over limit, original otherwise
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... [truncated, {len(text)} total chars]"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Set up logging for Parishad.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs to
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger for parishad
    logger = logging.getLogger("parishad")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a Parishad module.
    
    Args:
        name: Module name (e.g., "orchestrator", "roles.refiner")
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"parishad.{name}")
