"""
Unified Perception Tool (The Eyes & Ears).
Uses MarkItDown to convert Documents, Images, and Audio to text.
Lightweight, no heavy model downloads.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseTool, ToolResult


class PerceptionTool(BaseTool):
    """
    Unified tool for perceiving the world (Docs, Images, Audio).
    Powered by Microsoft MarkItDown.
    """
    
    name = "perception"
    description = "Convert files (PDF, Docx, Images, Audio) into text. Use for reading docs or seeing images."
    
    def __init__(self):
        super().__init__()
        self._markitdown = None

    def _get_markitdown(self):
        """Lazy import."""
        if self._markitdown is None:
            try:
                from markitdown import MarkItDown
                # Initialize without LLM client by default to keep it local/free.
                # Image description will be basic (OCR/Metadata) unless LLM is provided later.
                self._markitdown = MarkItDown()
            except ImportError:
                raise ImportError(
                    "markitdown dependency is missing. "
                    "Install with: pip install parishad[perception]"
                )
        return self._markitdown

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file (pdf, docx, jpg, mp3, etc.)"
                    }
                },
                "required": ["file_path"]
            }
        }

    def run(self, file_path: str) -> ToolResult:
        """Convert file to text."""
        try:
            path = Path(file_path)
            if not path.exists():
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"File not found: {file_path}"
                )
            
            md = self._get_markitdown()
            
            # Convert
            result = md.convert(str(path))
            
            if result and hasattr(result, "text_content"):
                return ToolResult(
                    success=True,
                    data=result.text_content,
                    metadata={
                        "source": str(path),
                        "format": path.suffix,
                        "title": result.title if hasattr(result, "title") else None
                    }
                )
            else:
                 return ToolResult(
                    success=False,
                    data=None,
                    error="Conversion returned empty result."
                )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Perception failed: {str(e)}"
            )
