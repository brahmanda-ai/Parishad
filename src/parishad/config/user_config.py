"""
User-level configuration management for Parishad.

Handles user preferences stored in ~/.parishad/config.yaml:
- default_profile: Which model profile to use by default
- default_mode: Which execution mode to use by default
- model_dir: Where models are stored
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class UserConfig:
    """User configuration settings."""
    default_profile: str
    default_mode: str
    model_dir: str


def get_user_config_dir() -> Path:
    """
    Return ~/.parishad directory (cross-platform via Path.home()).
    
    Returns:
        Path to user config directory
    """
    return Path.home() / ".parishad"


def get_user_config_path() -> Path:
    """
    Return path to ~/.parishad/config.yaml.
    
    Returns:
        Path to user config file
    """
    return get_user_config_dir() / "config.yaml"


def load_user_config() -> UserConfig:
    """
    Load user config if present, else return sensible defaults.
    
    Defaults:
        default_profile: "local_gpu"
        default_mode: "balanced"
        model_dir: "~/.parishad/models"
    
    Returns:
        UserConfig with settings
    """
    path = get_user_config_path()
    
    if not path.exists():
        # Return default config
        logger.debug("No user config found, using defaults")
        return UserConfig(
            default_profile="local_gpu",
            default_mode="balanced",
            model_dir=str(get_user_config_dir() / "models"),
        )
    
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        
        config = UserConfig(
            default_profile=data.get("default_profile", "local_gpu"),
            default_mode=data.get("default_mode", "balanced"),
            model_dir=data.get("model_dir", str(get_user_config_dir() / "models")),
        )
        
        logger.debug(f"Loaded user config: profile={config.default_profile}, mode={config.default_mode}")
        return config
        
    except Exception as e:
        logger.warning(f"Failed to load user config: {e}, using defaults")
        return UserConfig(
            default_profile="local_gpu",
            default_mode="balanced",
            model_dir=str(get_user_config_dir() / "models"),
        )


def save_user_config(cfg: UserConfig) -> None:
    """
    Save user config to ~/.parishad/config.yaml (create dir if needed).
    
    Args:
        cfg: UserConfig to save
    """
    cfg_dir = get_user_config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    
    data = {
        "default_profile": cfg.default_profile,
        "default_mode": cfg.default_mode,
        "model_dir": cfg.model_dir,
    }
    
    get_user_config_path().write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8"
    )
    
    logger.info(f"Saved user config: profile={cfg.default_profile}, mode={cfg.default_mode}")
