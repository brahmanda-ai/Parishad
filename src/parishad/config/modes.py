"""
Centralized mode configuration for Parishad Sabha.

Maps modes to pipeline configurations and role structures.
This is the single source of truth for the simplified 1/2/3 role system.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ModeConfig:
    """Configuration for a Parishad execution mode/Sabha."""
    mode_key: str  # CLI mode key: "fast", "balanced", "thorough"
    sabha_id: str  # Sabha ID for TUI: "laghu", "madhyam", "maha"
    sabha_name: str  # Display name
    sabha_hindi: str  # Hindi name
    pipeline_config: str  # Pipeline file name: "fast", "core", "extended"
    
    # Role structure
    role_count: int
    role_names: List[str]  # Actual role class names
    
    # Display metadata
    description: str
    ram_gb: int
    speed_label: str
    emoji: str
    model_slots: List[str]


# Single source of truth: Mode definitions with 1/2/3 role structure
MODES: Dict[str, ModeConfig] = {
    "fast": ModeConfig(
        mode_key="fast",
        sabha_id="laghu",
        sabha_name="Laghu Sabha",
        sabha_hindi="लघु सभा",
        pipeline_config="fast",
        role_count=5,
        role_names=["Darbari", "Sainik", "Raja"],  # Representative list
        description="Fastest - optimized council (5 roles)",
        ram_gb=8,
        speed_label="Fast",
        emoji="🚀",
        model_slots=["single"]
    ),
    
    "balanced": ModeConfig(
        mode_key="balanced",
        sabha_id="madhyam",
        sabha_name="Madhyam Sabha",
        sabha_hindi="मध्यम सभा",
        pipeline_config="core",
        role_count=8,
        role_names=["Darbari", "Majumdar", "Sainik", "Prerak"],
        description="Balanced - full core council (8 roles)",
        ram_gb=16,
        speed_label="Medium",
        emoji="⚡",
        model_slots=["heavy", "light"]
    ),
    
    "thorough": ModeConfig(
        mode_key="thorough",
        sabha_id="maha",
        sabha_name="Maha Sabha",
        sabha_hindi="महा सभा",
        pipeline_config="extended",
        role_count=10,
        role_names=["Pantapradhan", "Vidushak", "Sainik", "Raja"],
        description="Thorough - extended council (10 roles)",
        ram_gb=32,
        speed_label="Slow",
        emoji="👑",
        model_slots=["heavy", "mid", "light"]
    ),
}


# Backward compatibility mappings
SABHA_ID_TO_MODE = {
    "laghu": "fast",
    "madhyam": "balanced",
    "maha": "thorough",
}

MODE_TO_SABHA_ID = {
    "fast": "laghu",
    "balanced": "madhyam",
    "thorough": "maha",
    # Also support old names if they exist
    "core": "madhyam",
    "extended": "maha",
}


def get_mode_config(mode_or_sabha: str) -> ModeConfig:
    """
    Get mode configuration by mode key or Sabha ID.
    
    Args:
        mode_or_sabha: Mode key ("fast"/"balanced"/"thorough") or
                      Sabha ID ("laghu"/"madhyam"/"maha") or
                      Old name ("core"/"extended")
    
    Returns:
        ModeConfig for the requested mode
        
    Raises:
        ValueError: If mode/sabha is unknown
    """
    # Direct mode lookup
    if mode_or_sabha in MODES:
        return MODES[mode_or_sabha]
    
    # Sabha ID lookup
    if mode_or_sabha in SABHA_ID_TO_MODE:
        mode_key = SABHA_ID_TO_MODE[mode_or_sabha]
        return MODES[mode_key]
    
    # Old pipeline name lookup
    if mode_or_sabha in MODE_TO_SABHA_ID:
        sabha_id = MODE_TO_SABHA_ID[mode_or_sabha]
        mode_key = SABHA_ID_TO_MODE[sabha_id]
        return MODES[mode_key]
    
    raise ValueError(
        f"Unknown mode/sabha: '{mode_or_sabha}'. "
        f"Valid modes: {list(MODES.keys())}, "
        f"sabha IDs: {list(SABHA_ID_TO_MODE.keys())}"
    )


def get_pipeline_name(mode_or_sabha: str) -> str:
    """Get pipeline config file name for a mode."""
    config = get_mode_config(mode_or_sabha)
    return config.pipeline_config
