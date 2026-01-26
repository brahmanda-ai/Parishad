"""
Shell Execution Tool.
Allows the agent to execute system commands.
"""

import subprocess
import shlex
import os
from typing import Any, Dict, Optional

from .base import BaseTool, ToolResult


class ShellTool(BaseTool):
    """
    Tool for executing shell commands.
    Capabilities: run_command.
    """
    
    name = "shell"
    description = "Execute shell commands in the terminal."
    
    def __init__(self, timeout: int = 60, safe_mode: bool = False):
        """
        Initialize ShellTool.
        
        Args:
            timeout: Maximum execution time in seconds.
            safe_mode: If True, disallows potentially destructive commands (mock implementation).
        """
        super().__init__()
        self.timeout = timeout
        self.safe_mode = safe_mode

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Directory to execute command in (optional)"
                    }
                },
                "required": ["command"]
            }
        }

    def run(self, command: str, cwd: Optional[str] = None) -> ToolResult:
        """Run a shell command."""
        try:
            if self.safe_mode:
                # Basic blacklist (very naive, purely illustrative)
                forbidden = ["rm -rf /", ":(){ :|:& };:"]
                if any(f in command for f in forbidden):
                    return ToolResult(success=False, data=None, error="Command blocked by safe mode.")

            # Resolve cwd
            working_dir = os.path.abspath(cwd) if cwd else os.getcwd()

            # Execute
            # Using shell=True is dangerous but necessary for complex commands (pipes, etc.) often used by agents
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            
            if result.returncode == 0:
                return ToolResult(success=True, data=output.strip())
            else:
                 return ToolResult(
                    success=False, 
                    data=output.strip(), 
                    error=f"Command failed with exit code {result.returncode}"
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                data=None,
                error=f"Command timed out after {self.timeout}s"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Shell execution failed: {str(e)}"
            )
