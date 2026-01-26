"""
Configuration loader for pipeline definitions.

Prepares for Phase 2 config-driven pipelines (Core vs Extended).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
import logging

from .exceptions import InvalidPipelineConfigError

logger = logging.getLogger(__name__)


@dataclass
class RoleSpec:
    """Specification for a single role in the pipeline."""
    name: str
    class_name: str
    slot: str
    version: str = "0.1.3"
    budget_tokens: int = 1000
    dependencies: list[str] = field(default_factory=list)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    extra_config: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "class_name": self.class_name,
            "slot": self.slot,
            "version": self.version,
            "budget_tokens": self.budget_tokens,
            "dependencies": self.dependencies,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "extra_config": self.extra_config
        }


def load_pipeline_config(name: str, config_dir: Optional[Path] = None) -> list[RoleSpec]:
    """
    Load pipeline configuration from YAML file.
    
    Args:
        name: Pipeline name ("core" or "extended")
        config_dir: Optional directory containing config files
        
    Returns:
        List of RoleSpec objects defining the pipeline
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    # Resolve config directory
    if config_dir is None:
        # Default to package config directory
        package_dir = Path(__file__).parent.parent
        config_dir = package_dir / "config"
    
    config_path = config_dir / f"pipeline.{name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Pipeline config not found: {config_path}. "
            f"Expected one of: pipeline.core.yaml, pipeline.extended.yaml"
        )
    
    # Load YAML
    logger.debug(f"Loading pipeline config from {config_path}")
    
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    if not data:
        raise ValueError(f"Empty pipeline config: {config_path}")
    
    # Parse roles
    roles_data = data.get("roles", {})
    pipeline_order = data.get("pipeline", [])
    
    if not pipeline_order:
        raise ValueError(f"No pipeline order specified in {config_path}")
    
    # Build RoleSpec list in pipeline order
    role_specs = []
    
    for role_name in pipeline_order:
        role_config = roles_data.get(role_name, {})
        
        if not role_config:
            logger.warning(f"No configuration for role '{role_name}', using defaults")
            role_config = {}
        
        # Extract known fields to avoid duplication in extra_config
        known_fields = {
            "name", "class", "slot", "version", "budget_tokens", 
            "dependencies", "max_tokens", "temperature"
        }
        extra_config = {k: v for k, v in role_config.items() if k not in known_fields}
        
        # Extract role spec
        spec = RoleSpec(
            name=role_name.lower(),  # Always store as lowercase for consistent lookups
            class_name=role_config.get("class", role_name.capitalize()),
            slot=role_config.get("slot", "mid"),
            version=role_config.get("version", "0.1.1"),
            budget_tokens=role_config.get("budget_tokens", 1000),
            dependencies=role_config.get("dependencies", []),
            max_tokens=role_config.get("max_tokens"),
            temperature=role_config.get("temperature"),
            extra_config=extra_config
        )
        
        role_specs.append(spec)
        logger.debug(f"Loaded role spec: {role_name} ({spec.class_name}, slot={spec.slot})")
    
    logger.info(f"Loaded pipeline '{name}' with {len(role_specs)} roles: {pipeline_order}")
    
    # Validate the loaded configuration
    validation_result = validate_pipeline_config(role_specs)
    if not validation_result["valid"]:
        raise InvalidPipelineConfigError(validation_result["errors"])
    
    return role_specs


def validate_pipeline_config(role_specs: list[RoleSpec]) -> dict[str, any]:
    """
    Validate a loaded pipeline configuration.
    
    Args:
        role_specs: List of role specifications
        
    Returns:
        Validation result dict with 'valid' (bool) and 'errors' (list) keys
    """
    errors = []
    
    # Check for empty pipeline
    if not role_specs:
        errors.append("Pipeline is empty")
        return {"valid": False, "errors": errors}
    
    # Check for duplicate role names
    role_names = [spec.name for spec in role_specs]
    duplicates = [name for name in role_names if role_names.count(name) > 1]
    if duplicates:
        errors.append(f"Duplicate role names: {set(duplicates)}")
    
    # Check for valid slots
    valid_slots = {"small", "mid", "big"}
    for spec in role_specs:
        if spec.slot not in valid_slots:
            errors.append(f"Invalid slot '{spec.slot}' for role '{spec.name}'")
    
    # Check for circular dependencies
    for spec in role_specs:
        for dep in spec.dependencies:
            if dep not in role_names:
                errors.append(f"Role '{spec.name}' depends on unknown role '{dep}'")
            if dep == spec.name:
                errors.append(f"Role '{spec.name}' has circular self-dependency")
    
    # Check budget sanity
    for spec in role_specs:
        if spec.budget_tokens < 0:
            errors.append(f"Negative budget for role '{spec.name}': {spec.budget_tokens}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def get_available_pipelines(config_dir: Optional[Path] = None) -> list[str]:
    """
    List all available pipeline configurations.
    
    Args:
        config_dir: Optional directory containing config files
        
    Returns:
        List of pipeline names (without .yaml extension)
    """
    if config_dir is None:
        package_dir = Path(__file__).parent.parent
        config_dir = package_dir / "config"
    
    if not config_dir.exists():
        return []
    
    # Find all pipeline.*.yaml files
    pipeline_files = config_dir.glob("pipeline.*.yaml")
    
    # Extract names
    names = []
    for path in pipeline_files:
        # Extract name between "pipeline." and ".yaml"
        name = path.stem.replace("pipeline.", "")
        if name != "pipeline":  # Exclude "pipeline.yaml" itself
            names.append(name)
    
    return sorted(names)
