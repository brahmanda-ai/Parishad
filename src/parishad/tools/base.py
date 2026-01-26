"""
Base classes and interfaces for Parishad tools.
Tools enable the agent to interact with the external world (Perception and Action).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standardized output from a tool execution."""
    success: bool
    data: Any  # The actual result (text, json, file path, etc.)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for all tools."""

    name: str = "base_tool"
    description: str = "Base tool description"
    
    def __init__(self, **kwargs):
        """Initialize the tool."""
        pass

    @property
    def schema(self) -> Dict[str, Any]:
        """Return the JSON schema for this tool's input."""
        # By default, can infer from the `run` method type hints if using Pydantic V2
        # For now, subclasses should define this explicitly or we use a decorator helper
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass

    def __call__(self, **kwargs) -> ToolResult:
        """Syntactic sugar for running the tool."""
        try:
            return self.run(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )
