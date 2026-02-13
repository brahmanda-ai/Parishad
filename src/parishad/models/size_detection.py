"""
Model size detection and optimization utilities.

Provides utilities to detect model parameter counts and adjust
inference settings based on model size for optimal performance.
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def load_model_catalog() -> dict:
    """
    Load the models.json catalog.
    
    Returns:
        Parsed JSON catalog dict
    """
    try:
        # Find models.json relative to this file
        current_file = Path(__file__)
        catalog_path = current_file.parent.parent / "data" / "models.json"
        
        if not catalog_path.exists():
            logger.warning(f"Model catalog not found at {catalog_path}")
            return {}
        
        with open(catalog_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load model catalog: {e}")
        return {}


def parse_params_from_string(params_str: str) -> float:
    """
    Parse parameter count from strings like "0.5B", "1.5B", "7B", "8x7B".
    
    Args:
        params_str: String representation of parameters (e.g., "0.5B", "7B")
        
    Returns:
        Parameter count in billions
    """
    if not params_str or not isinstance(params_str, str):
        return 0.0
    
    params_lower = params_str.lower().strip()
    
    # Handle MoE models (e.g., "8x7B" = 56B total, but ~7B active)
    moe_match = re.match(r'(\d+)x(\d+)b', params_lower)
    if moe_match:
        experts = int(moe_match.group(1))
        size_per = int(moe_match.group(2))
        # Return active parameters (one expert at a time)
        return float(size_per)
    
    # Handle standard format (e.g., "0.5B", "1.5B", "7B")
    match = re.search(r'(\d+(?:\.\d+)?)\s*b', params_lower)
    if match:
        return float(match.group(1))
    
    # Handle million-scale models (e.g., "500M")
    match_m = re.search(r'(\d+(?:\.\d+)?)\s*m', params_lower)
    if match_m:
        return float(match_m.group(1)) / 1000.0
    
    return 0.0


def get_model_size_category(params_billion: float) -> str:
    """
    Categorize model by size.
    
    Args:
        params_billion: Model size in billions of parameters
        
    Returns:
        Category: "tiny" (<2B), "small" (2-7B), "medium" (7-20B), 
                 "large" (20-70B), "xl" (>70B)
    """
    if params_billion < 2.0:
        return "tiny"
    elif params_billion < 7.0:
        return "small"
    elif params_billion < 20.0:
        return "medium"
    elif params_billion < 70.0:
        return "large"
    else:
        return "xl"


def get_optimal_temperature(params_billion: float, task_type: str = "general") -> float:
    """
    Get optimal temperature based on model size and task type.
    
    Tiny models need much lower temperature for stability.
    Larger models can handle higher creativity without hallucination.
    
    Args:
        params_billion: Model size in billions
        task_type: Task type ("general", "code", "math", "creative")
        
    Returns:
        Optimal temperature value
    """
    category = get_model_size_category(params_billion)
    
    base_temps = {
        "general": {
            "tiny": 0.1,    # Very low for stability
            "small": 0.3,   # Low for consistency
            "medium": 0.5,  # Moderate
            "large": 0.6,   # Moderate-high
            "xl": 0.7       # Higher creativity
        },
        "code": {
            "tiny": 0.05,   # Extremely deterministic
            "small": 0.1,
            "medium": 0.2,
            "large": 0.3,
            "xl": 0.4
        },
        "math": {
            "tiny": 0.05,   # Extremely deterministic
            "small": 0.1,
            "medium": 0.2,
            "large": 0.3,
            "xl": 0.4
        },
        "creative": {
            "tiny": 0.3,    # Still low but allow some variation
            "small": 0.5,
            "medium": 0.7,
            "large": 0.8,
            "xl": 0.9
        }
    }
    
    temps = base_temps.get(task_type, base_temps["general"])
    return temps.get(category, 0.5)


def get_optimal_max_tokens(params_billion: float) -> int:
    """
    Get optimal max_tokens based on model size.
    
    Tiny models struggle with very long outputs.
    
    Args:
        params_billion: Model size in billions
        
    Returns:
        Optimal max_tokens value
    """
    category = get_model_size_category(params_billion)
    
    limits = {
        "tiny": 256,    # Keep outputs short
        "small": 512,   # Moderate length
        "medium": 1024, # Standard
        "large": 2048,  # Long-form
        "xl": 4096      # Very long
    }
    
    return limits.get(category, 1024)


def should_use_direct_mode(params_billion: float) -> bool:
    """
    Determine if a model should use direct Q&A mode instead of orchestration.
    
    DEPRECATED: Now we use intelligent role reduction instead of direct mode.
    Keeping this for backward compatibility but always returns False.
    
    Args:
        params_billion: Model size in billions
        
    Returns:
        Always False (use orchestration with role reduction instead)
    """
    # We no longer use direct mode - instead we reduce roles intelligently
    return False


def get_role_reduction_level(params_billion: float) -> str:
    """
    Determine how many roles to use based on model size.
    
    This preserves the thinking structure while adapting to model capability.
    
    Args:
        params_billion: Model size in billions
        
    Returns:
        "minimal" (2 roles: Sainik → Raja)
        "lite" (3 roles: Sainik → Majumdar → Raja)
        "full" (all roles as designed)
    """
    if params_billion < 1.5:
        return "minimal"  # 2 roles: Sainik → Raja (0.5-1.4B models)
    elif params_billion < 2.5:
        return "lite"     # 3 roles: Sainik → Majumdar → Raja (1.5-2.4B models)
    else:
        return "full"     # Full sabha pipeline (2.5B+ models)


def get_model_params_from_catalog(model_id: str, model_catalog: dict) -> float:
    """
    Extract parameter count from model catalog.
    
    Args:
        model_id: Model identifier or shortcut
        model_catalog: Loaded models.json catalog
        
    Returns:
        Parameter count in billions (0.0 if not found)
    """
    if not model_catalog or "sources" not in model_catalog:
        return 0.0
    
    # Search all sources
    for source_name, source_data in model_catalog["sources"].items():
        models = source_data.get("models", [])
        for model in models:
            # Match by shortcut or name
            if model.get("shortcut") == model_id or model.get("name") == model_id:
                params_str = model.get("params", "")
                return parse_params_from_string(params_str)
    
    # Fallback: try parsing from model_id itself
    return parse_params_from_string(model_id)


def get_inference_optimizations(
    params_billion: float,
    task_type: str = "general"
) -> dict:
    """
    Get complete inference optimization settings for a model size.
    
    Args:
        params_billion: Model size in billions
        task_type: Task type for temperature tuning
        
    Returns:
        Dict with recommended settings
    """
    category = get_model_size_category(params_billion)
    use_direct = should_use_direct_mode(params_billion)
    
    return {
        "params_billion": params_billion,
        "category": category,
        "use_direct_mode": use_direct,
        "temperature": get_optimal_temperature(params_billion, task_type),
        "max_tokens": get_optimal_max_tokens(params_billion),
        "top_p": 0.9 if category != "tiny" else 0.95,  # Slightly higher for stability
        "repeat_penalty": 1.1 if category == "tiny" else 1.0,  # Prevent repetition
        "prompt_style": "simple" if use_direct else "structured",
        "json_output": not use_direct,  # Disable JSON for tiny models
    }


def log_model_optimization_info(model_id: str, params_billion: float):
    """
    Log model optimization information for debugging.
    
    Args:
        model_id: Model identifier
        params_billion: Parameter count
    """
    opts = get_inference_optimizations(params_billion)
    
    logger.info(
        f"Model: {model_id} ({params_billion}B params) - Category: {opts['category']}"
    )
    logger.info(f"Optimizations: temp={opts['temperature']}, max_tokens={opts['max_tokens']}")
    
    if opts['use_direct_mode']:
        logger.warning(
            f"⚠ Tiny model detected ({params_billion}B < 2B) - Using DIRECT MODE "
            "to bypass complex orchestration. Simple Q&A prompts will be used."
        )


# Convenience function for quick checks
def is_tiny_model(model_id: str, model_catalog: Optional[dict] = None) -> Tuple[bool, float]:
    """
    Quick check if a model is tiny (<2B params).
    
    Args:
        model_id: Model identifier
        model_catalog: Optional catalog to search
        
    Returns:
        Tuple of (is_tiny, params_billion)
    """
    if model_catalog:
        params = get_model_params_from_catalog(model_id, model_catalog)
    else:
        params = parse_params_from_string(model_id)
    
    return (params < 2.0 and params > 0.0, params)
