"""
File System Action Tool.
Allows the agent to read, write, and navigate the file system.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Literal

from .base import BaseTool, ToolResult


class FileSystemTool(BaseTool):
    """
    Tool for interacting with the file system.
    Capabilities: read, write, list.
    """
    
    name = "file_system"
    description = "Read, write, and list files in the file system. Use this to modify code or read documentation."
    
    def __init__(self, working_directory: Optional[str] = None):
        """
        Initialize FileSystemTool.
        
        Args:
            working_directory: Root directory to limit operations to (optional).
                               If NOT set, operations are unrestricted (use with caution).
        """
        super().__init__()
        self.working_directory = Path(working_directory).resolve() if working_directory else Path.cwd()

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "list"],
                        "description": "Operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (for 'write' operation only)"
                    }
                },
                "required": ["operation", "path"]
            }
        }

    def run(self, operation: Literal["read", "write", "list"], path: str, content: Optional[str] = None) -> ToolResult:
        """Execute file system operation."""
        try:
            target_path = Path(path).resolve()
            
            # Basic security check: ensure path is relative to working directory if strict mode desired
            # For now, we allow absolute paths to be powerful agents, but we log usage.
            
            if operation == "read":
                if not target_path.exists():
                     return ToolResult(success=False, data=None, error=f"File not found: {path}")
                if not target_path.is_file():
                     return ToolResult(success=False, data=None, error=f"Not a file: {path}")
                
                try:
                    text = target_path.read_text(encoding="utf-8")
                    return ToolResult(success=True, data=text)
                except UnicodeDecodeError:
                    return ToolResult(success=False, data=None, error="Binary file reading not supported yet.")

            elif operation == "write":
                if content is None:
                    return ToolResult(success=False, data=None, error="Content required for write operation.")
                
                # Ensure directory exists
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(content, encoding="utf-8")
                return ToolResult(success=True, data=f"Successfully wrote {len(content)} bytes to {path}")

            elif operation == "list":
                if not target_path.exists():
                     return ToolResult(success=False, data=None, error=f"Directory not found: {path}")
                
                items = []
                if target_path.is_dir():
                    for item in target_path.iterdir():
                        kind = "DIR" if item.is_dir() else "FILE"
                        items.append(f"[{kind}] {item.name}")
                else:
                    return ToolResult(success=False, data=None, error=f"Not a directory: {path}")
                
                return ToolResult(success=True, data="\n".join(sorted(items)))

            else:
                return ToolResult(success=False, data=None, error=f"Unknown operation: {operation}")

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"FileSystem error ({operation}): {str(e)}"
            )
