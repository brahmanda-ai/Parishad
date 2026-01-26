"""
Custom exceptions for orchestrator module.
"""


class InvalidPipelineConfigError(ValueError):
    """Raised when pipeline configuration is invalid."""
    
    def __init__(self, errors: list[str]):
        self.errors = errors
        error_list = "\n  - ".join(errors)
        super().__init__(
            f"Invalid pipeline configuration:\n  - {error_list}"
        )
