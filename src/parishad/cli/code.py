"""
Parishad CLI - Unified TUI with setup wizard and chat interface.

Features:
- Setup wizard on first run (Sabha selection, model browser)
- Interactive chat with agentic coding assistant
- Advanced input (@mentions, /commands, ? help)
"""

from __future__ import annotations

import json
import os
import re
import sys
import subprocess
import socket
import shutil
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from dataclasses import dataclass, field

from textual.app import App, ComposeResult
from textual.message import Message
from textual.containers import Container, Vertical, Horizontal, Grid, ScrollableContainer
from textual.widgets import (
    Button, Footer, Header, Input, Label, ListItem, 
    ListView, Select, Static, TabbedContent, TabPane,
    ProgressBar, RichLog
)
from textual.suggester import Suggester
from textual.binding import Binding
from textual.screen import Screen
from textual import on
from textual.message import Message
from rich.text import Text
from rich.panel import Panel


# =============================================================================
# Configuration - Robust path resolution with fallbacks
# =============================================================================

def _get_config_dir() -> Path:
    """
    Get config directory - always uses ~/.parishad for consistency.
    This is the single source of truth for Parishad configuration.
    """
    return Path.home() / ".parishad"

# Define config constants
CONFIG_DIR = _get_config_dir()
CONFIG_FILE = CONFIG_DIR / "config.json"

def load_parishad_config() -> Optional[ParishadConfig]:
    """
    Load Parishad configuration from disk.
    
    Returns:
        ParishadConfig if valid config exists, None otherwise
    """
    try:
        if not CONFIG_FILE.exists():
            return None
        
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # New structure: session data is under 'session' key
        # Old structure: session data is at root level
        session_data = data.get("session", data)
        
        return ParishadConfig.from_dict(session_data, full_config=data)
        
    except (json.JSONDecodeError, KeyError, Exception) as e:
        # Invalid config, treat as no config
        return None


@dataclass
class ParishadConfig:
    """Central configuration for Parishad TUI."""
    sabha: Optional[str] = None          # "laghu" | "madhyam" | "maha"
    backend: Optional[str] = None        # "llama_cpp" | "mlx" | "transformers" | "ollama"
    model: Optional[str] = None          # model id/name
    cwd: str = ""       # working directory (optional)
    setup_complete: bool = False
    
    # Multi-model assignment mapping (slot -> model_id)
    model_map: Dict[str, str] = field(default_factory=dict)
    
    # Store other fields to preserve them (e.g., system, models, permissions)
    extra_fields: Dict = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, session_data: Dict, full_config: Dict = None) -> "ParishadConfig":
        """Create config from dictionary."""
        # Store full config for preservation
        extra = full_config if full_config else {}
        
        return cls(
            sabha=session_data.get("sabha"),
            backend=session_data.get("backend"),
            model=session_data.get("model"),
            cwd=session_data.get("cwd", ""),
            setup_complete=session_data.get("setup_complete", False),
            model_map=session_data.get("model_map", {}),
            extra_fields=extra
        )
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for JSON serialization."""
        # Start with preserved fields
        result = dict(self.extra_fields) if self.extra_fields else {}
        
        # Update top-level flags
        result["setup_complete"] = True
        
        # Update session data
        result["session"] = {
            "sabha": self.sabha,
            "backend": self.backend,
            "model": self.model,
            "cwd": self.cwd,
            "model_map": self.model_map
        }
        
        return result

    def get_mode(self) -> str:
        """Get mode name from sabha using modes.py mapping."""
        from ..config.modes import SABHA_ID_TO_MODE
        return SABHA_ID_TO_MODE.get(self.sabha, "fast")
    
    def get_pipeline_config(self) -> str:
        """Get pipeline config name for engine initialization."""
        from ..config.modes import get_pipeline_name
        return get_pipeline_name(self.sabha)


def save_parishad_config(config: ParishadConfig) -> bool:
    """
    Save Parishad configuration to disk atomically.
    
    Uses atomic write pattern: write to .tmp file, then rename.
    This prevents corruption if process is interrupted.
    
    Args:
        config: Configuration to save
        
    Returns:
        True if save successful, False otherwise
    """
    try:
        # DEBUG LOGGING
        db_path = Path.home() / "parishad_debug.log"
        with open(db_path, "a") as f:
            f.write(f"DEBUG: Attempting to save config to {CONFIG_FILE}\n")

        # Ensure config directory exists
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Read existing file to get latest state of other fields (models, system, etc.)
        # This prevents overwriting updates from other components (like ModelManager)
        current_data = {}
        if CONFIG_FILE.exists():
            try:
                 with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
            except Exception:
                 # If read fails, fall back to what we have in config.extra_fields
                 current_data = dict(config.extra_fields) if config.extra_fields else {}
        else:
            current_data = dict(config.extra_fields) if config.extra_fields else {}

        # Update session data managed by this config object
        # We explicitly update only what we own
        current_data["session"] = {
             "sabha": config.sabha,
             "backend": config.backend,
             "model": config.model,
             "cwd": config.cwd,
             "model_map": config.model_map
        }
        current_data["setup_complete"] = True

        # Write to temporary file first
        tmp_file = CONFIG_FILE.with_suffix(".json.tmp")
        
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=2)
        
        # Atomic rename (overwrites existing file)
        tmp_file.replace(CONFIG_FILE)
        
        with open(db_path, "a") as f:
            f.write(f"DEBUG: Config saved successfully to {CONFIG_FILE}\n")
        
        return True
        
    except Exception as e:
        # Log the error
        db_path = Path.home() / "parishad_debug.log"
        with open(db_path, "a") as f:
            f.write(f"DEBUG: save_parishad_config FAILED: {e}\n")
            import traceback
            f.write(traceback.format_exc())
        
        # Clean up temp file if it exists
        try:
            tmp_file = CONFIG_FILE.with_suffix(".json.tmp")
            if tmp_file.exists():
                tmp_file.unlink()
        except:
            pass
        
        return False


# =============================================================================
# Input Parsing Layer - Task 2
# =============================================================================

@dataclass
class LoadedFile:
    """Represents a loaded file with its content."""
    path: str
    exists: bool
    content: Optional[str] = None
    error: Optional[str] = None
    size_bytes: int = 0


@dataclass
class ParsedInput:
    """
    Structured representation of user input after parsing.
    
    Attributes:
        raw: Original input string
        is_command: True if input starts with /
        command_name: Command name without / (e.g., "help", "exit")
        command_args: List of command arguments
        tools: List of file references (e.g., [{"type": "file", "path": "foo.py"}])
        flags: Dict of boolean flags (e.g., {"idk": True, "safe": False})
        user_query: Natural language part with @ and # tokens removed
    """
    raw: str
    is_command: bool = False
    command_name: Optional[str] = None
    command_args: List[str] = field(default_factory=list)
    tools: List[Dict[str, str]] = field(default_factory=list)
    flags: Dict[str, bool] = field(default_factory=dict)
    user_query: str = ""


def parse_input(raw: str) -> ParsedInput:
    """
    Parse user input into structured format.
    
    Handles:
    - Slash commands: /help, /exit, /clear, /config, etc.
    - File references: @path/to/file.py
    - Flags: #idk, #safe, #noguess
    
    Args:
        raw: Raw input string from user
        
    Returns:
        ParsedInput with parsed components
    """
    raw_stripped = raw.strip()
    
    # Empty input
    if not raw_stripped:
        return ParsedInput(raw=raw, user_query="")
    
    # Command detection (starts with /)
    if raw_stripped.startswith("/"):
        parts = raw_stripped.split(maxsplit=1)
        cmd_name = parts[0][1:].lower()  # Remove / and lowercase
        cmd_args = parts[1].split() if len(parts) > 1 else []
        
        return ParsedInput(
            raw=raw,
            is_command=True,
            command_name=cmd_name,
            command_args=cmd_args
        )
    
    # Not a command - parse tools and flags
    tools = []
    flags = {}
    
    # Pattern for @file references
    # Matches: @filename.ext, @path/to/file.ext, @"path with spaces.txt"
    file_pattern = r'@(?:"([^"]+)"|([^\s]+))'
    
    for match in re.finditer(file_pattern, raw_stripped):
        # Group 1 is quoted path, group 2 is unquoted path
        file_path = match.group(1) if match.group(1) else match.group(2)
        tools.append({
            "type": "file",
            "path": file_path
        })
    
    # Remove @file references from query
    query_without_files = re.sub(file_pattern, '', raw_stripped)
    
    # Pattern for flags: #idk, #safe, #noguess
    flag_pattern = r'#(idk|safe|noguess|careful)\b'
    
    for match in re.finditer(flag_pattern, query_without_files, re.IGNORECASE):
        flag_name = match.group(1).lower()
        flags[flag_name] = True
    
    # Remove flags from query
    user_query = re.sub(flag_pattern, '', query_without_files, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    user_query = ' '.join(user_query.split())
    
    return ParsedInput(
        raw=raw,
        is_command=False,
        tools=tools,
        flags=flags,
        user_query=user_query
    )


def load_file(file_path: str, base_dir: Path, max_size_kb: int = 1024) -> LoadedFile:
    """
    Load a file with error handling and size limits.
    
    Args:
        file_path: Path to file (relative or absolute)
        base_dir: Base directory for resolving relative paths
        max_size_kb: Maximum file size in KB (default 1MB)
        
    Returns:
        LoadedFile with content or error information
    """
    try:
        # Resolve path
        path_obj = Path(file_path)
        if not path_obj.is_absolute():
            path_obj = base_dir / path_obj
        
        path_obj = path_obj.resolve()
        
        # Check existence
        if not path_obj.exists():
            return LoadedFile(
                path=file_path,
                exists=False,
                error=f"File not found: {file_path}"
            )
        
        if path_obj.is_dir():
             # Handle directories by listing content (simulating 'ls' or 'tree')
             try:
                 # Simple listing for now. Could be enhanced with a Tool run if full `ls -R` needed.
                 # Let's do a shallow listing with file types.
                 items = []
                 for item in sorted(path_obj.iterdir()):
                     prefix = "[DIR]" if item.is_dir() else "[FILE]"
                     size = f"{item.stat().st_size}b" if item.is_file() else ""
                     items.append(f"{prefix} {item.name} {size}")
                 
                 dir_content = f"Directory Listing for {file_path}:\n" + "\n".join(items)
                 
                 return LoadedFile(
                    path=file_path,
                    exists=True,
                    content=dir_content,
                    size_bytes=len(dir_content.encode('utf-8'))
                )
             except Exception as e:
                 return LoadedFile(
                    path=file_path,
                    exists=True,
                    error=f"Error listing directory: {e}"
                )

        if not path_obj.is_file():
            return LoadedFile(
                path=file_path,
                exists=False,
                error=f"Not a file or directory: {file_path}"
            )
            
        # Check for binary/image extensions to prevent crash
        suffix = path_obj.suffix.lower()
        binary_exts = {
            ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".ico",  # Images
            ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar",  # Archives
            ".exe", ".bin", ".dll", ".so", ".dylib",  # Binaries
            ".pyc", ".pkl", ".db", ".sqlite"  # Data
        }
        
        # Calculate max bytes early
        max_bytes = max_size_kb * 1024
        
        if suffix in binary_exts:
             try:
                 from ..tools.perception import PerceptionTool
                 # Attempt conversion
                 # Attempt conversion (Use default local config for CLI view)
                 pt = PerceptionTool()
                 result = pt.run(str(path_obj))
                 
                 if result.success:
                     content_preview = result.data
                     # Truncate if too huge
                     if len(content_preview) > max_bytes:
                         content_preview = content_preview[:max_bytes] + "... [Truncated]"
                         
                     return LoadedFile(
                        path=file_path,
                        exists=True,
                        content=f"[Content processed by PerceptionTool]\n{content_preview}",
                        size_bytes=path_obj.stat().st_size
                    )
                 else:
                     # PerceptionTool ran but failed
                     return LoadedFile(
                        path=file_path,
                        exists=True,
                        content="",
                        size_bytes=path_obj.stat().st_size,
                        error=f"Perception failed: {result.error}"
                     )
             except ImportError as e:
                 return LoadedFile(
                    path=file_path,
                    exists=True,
                    content="",
                    size_bytes=path_obj.stat().st_size,
                    error=f"Preview unavailable: PerceptionTool import failed ({e})"
                )
             except Exception as e:
                 return LoadedFile(
                    path=file_path,
                    exists=True,
                    content="",
                    size_bytes=path_obj.stat().st_size,
                    error=f"Preview unavailable: {str(e)}"
                )

             # Classify for better user message
             if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".ico"}:
                 file_type = "Image"
             elif suffix == ".pdf":
                 file_type = "PDF"
             elif suffix in {".zip", ".tar", ".gz", ".7z", ".rar"}:
                 file_type = "Archive"
             else:
                 file_type = "Binary"

             return LoadedFile(
                path=file_path,
                exists=True,
                content=f"[{file_type} file detected: {file_path}. Content not viewable in TUI.]",
                size_bytes=path_obj.stat().st_size,
                error=f"{file_type} file skipped (content not extractable)"
            )
        
        # Check size
        size_bytes = path_obj.stat().st_size
        max_bytes = max_size_kb * 1024
        
        if size_bytes > max_bytes:
            # Read truncated
            with open(path_obj, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(max_bytes)
            
            return LoadedFile(
                path=file_path,
                exists=True,
                content=content,
                size_bytes=size_bytes,
                error=f"File truncated (size: {size_bytes // 1024}KB, limit: {max_size_kb}KB)"
            )
        
        # Read full file
        with open(path_obj, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        return LoadedFile(
            path=file_path,
            exists=True,
            content=content,
            size_bytes=size_bytes
        )
        
    except PermissionError:
        return LoadedFile(
            path=file_path,
            exists=True,
            error=f"Permission denied: {file_path}"
        )
    except Exception as e:
        return LoadedFile(
            path=file_path,
            exists=False,
            error=f"Error reading {file_path}: {type(e).__name__}: {e}"
        )


def build_augmented_prompt(user_query: str, loaded_files: List[LoadedFile], flags: Dict[str, bool]) -> str:
    """
    Build the final prompt with file contents and flag guidance.
    
    Args:
        user_query: User's natural language query
        loaded_files: List of loaded files with their contents
        flags: Dict of flags like {"idk": True}
        
    Returns:
        Augmented prompt string
    """
    parts = []
    
    # Add flag guidance at the beginning if present
    if flags.get("idk") or flags.get("careful"):
        parts.append(
            "Important: The user prefers you to admit when you don't know rather than guess. "
            "If you are uncertain or lack sufficient information, explicitly say 'I don't know' "
            "or 'I'm not sure' instead of making assumptions.\n"
        )
    
    if flags.get("safe") or flags.get("noguess"):
        parts.append(
            "Important: The user wants safe, conservative responses. "
            "Avoid speculation and only state what you're confident about.\n"
        )
    
    # Add file contents
    valid_files = [f for f in loaded_files if f.exists and f.content]
    if valid_files:
        parts.append("\nYou are being provided with file contents from the current project:\n")
        
        for file in valid_files:
            parts.append(f"\n<<FILE: {file.path}>>")
            parts.append(file.content)
            parts.append("</FILE>\n")
            
            if file.error:  # Truncation warning
                parts.append(f"[Note: {file.error}]\n")
    
    # Add user query
    if user_query:
        if valid_files or flags:
            parts.append(f"\nUser request:\n{user_query}")
        else:
            parts.append(user_query)
    
    return "".join(parts)


# ASCII logo - Devanagari परिषद् with left-to-right saffron gradient (vibrant)
LOGO = """[#e65e1c]   ██████[/][#ff671f]        [/][#ff7a3d]        [/]
[#e65e1c]  ██ ╔═[/][#ff671f]═██     [/][#ff7a3d]        [/]
[#e65e1c]███████[/][#ff671f]██████████[/][#ff7a3d]███████[/][#ff8c5a]████████[/][#ff9e78]████████[/][#ffb095]██████████═╗[/]
[#e65e1c]  ╚═██ ╔═[/][#ff671f]═██ ╔═██[/][#ff7a3d] ╔═▀▀▀▀█[/][#ff8c5a]█ ╔═███╔[/][#ff9e78]══██ ╔═[/][#ffb095]══════██ ╔═╝[/]
[#e65e1c]  ██ ║ [/][#ff671f] ██ ║ ██[/][#ff7a3d] ║     █[/][#ff8c5a]█ ║ ██ █[/][#ff9e78]█ ██ ║  [/][#ffb095] ██████ ║[/]
[#e65e1c]  ██ ║ [/][#ff671f] ██ ║ ██[/][#ff7a3d] ║   ██▀[/][#ff8c5a]▀╔╝ ██ ║[/][#ff9e78] ███ ║ █[/][#ffb095]█ ╔═════╝[/]
[#e65e1c]    ███[/][#ff671f]███ ║ ██[/][#ff7a3d] ║ ██   [/][#ff8c5a]╔╝    ██[/][#ff9e78]████ ║ █[/][#ffb095]█ ║  ██═╗[/]
[#e65e1c]       [/][#ff671f] ██ ║ ██[/][#ff7a3d] ║   ██ [/][#ff8c5a]╚═╗     [/][#ff9e78]  ██ ║  [/][#ffb095] ██████ ║[/]
[#e65e1c]       [/][#ff671f] ██ ║ ██[/][#ff7a3d] ║     █[/][#ff8c5a]█ ║     [/][#ff9e78]  ██ ║  [/][#ffb095]     ██ ║[/]
[#e65e1c]       [/][#ff671f] ╚══╝ ╚═[/][#ff7a3d]═╝     ╚[/][#ff8c5a]══╝     [/][#ff9e78]  ╚══╝  [/][#ffb095]     ╚══╝[/]"""


# =============================================================================
# Sabha (Council) Configurations
# =============================================================================

@dataclass
class SabhaConfig:
    """Sabha configuration."""
    id: str
    name: str
    hindi_name: str
    description: str
    roles: int
    ram_gb: int
    speed: str
    emoji: str
    model_slots: list  # ["heavy", "mid", "light"] etc.


SABHAS = [
    SabhaConfig(
        id="maha",
        name="Maha Sabha",
        hindi_name="महा सभा",
        description="Thorough: 3 roles (Analysis → Planning → Execution)",
        roles=3,
        ram_gb=32,
        speed="Slow",
        emoji="👑",
        model_slots=["heavy", "mid", "light"]
    ),
    SabhaConfig(
        id="madhyam",
        name="Madhyam Sabha", 
        hindi_name="मध्यम सभा",
        description="Balanced: 2 roles (Planning → Execution)",
        roles=2,
        ram_gb=16,
        speed="Medium",
        emoji="⚡",
        model_slots=["heavy", "light"]
    ),
    SabhaConfig(
        id="laghu",
        name="Laghu Sabha",
        hindi_name="लघु सभा",
        description="Fast: 1 role (Direct Execution)",
        roles=1,
        ram_gb=8,
        speed="Fast",
        emoji="🚀",
        model_slots=["single"]
    ),
]




# =============================================================================
# Model Catalog
# =============================================================================

MODELS_JSON_PATH = Path(__file__).parent.parent / "data" / "models.json"

@dataclass
class ModelInfo:
    """Model information."""
    name: str
    shortcut: str
    size_gb: float
    description: str
    source: str  # gguf, mlx, safetensors (model format)
    quantization: str = "Q4_K_M"
    distributor: str = ""
    params: str = ""
    tags: list = None


def load_model_catalog() -> dict:
    """Load model catalog from JSON file."""
    if MODELS_JSON_PATH.exists():
        try:
            with open(MODELS_JSON_PATH, encoding='utf-8') as f:
                data = json.load(f)
            
            catalog = {}
            for source_key, source_data in data.get("sources", {}).items():
                models = []
                for m in source_data.get("models", []):
                    models.append(ModelInfo(
                        name=m.get("name", ""),
                        shortcut=m.get("shortcut", ""),
                        size_gb=m.get("size_gb", 0),
                        description=m.get("description", ""),
                        source=source_key,
                        quantization=m.get("quantization", "Q4_K_M"),
                        distributor=m.get("distributor", ""),
                        params=m.get("params", ""),
                        tags=m.get("tags", []),
                    ))
                catalog[source_key] = models
            return catalog
        except Exception as e:
            print(f"Error loading models.json: {e}")
    
    # Fallback to minimal catalog
    return {
        "gguf": [
            ModelInfo("Llama 3.2 3B", "bartowski/Llama-3.2-3B-Instruct-GGUF", 2.0, "Efficient and fast", "gguf", "Q4_K_M", "Meta", "3B"),
            ModelInfo("Qwen 2.5 7B", "Qwen/Qwen2.5-7B-Instruct-GGUF", 4.5, "Excellent reasoning", "gguf", "Q4_K_M", "Alibaba", "7B"),
        ],
        "mlx": [
            ModelInfo("Llama 3.2 3B MLX", "mlx-community/Llama-3.2-3B-Instruct-4bit", 2.0, "Mac optimized", "mlx", "4-bit", "Meta", "3B"),
        ],
        "safetensors": [
            ModelInfo("Llama 3.2 3B", "meta-llama/Llama-3.2-3B-Instruct", 6.0, "Full precision", "safetensors", "FP16", "Meta", "3B"),
        ],
    }


# Load catalog on import
MODEL_CATALOG = load_model_catalog()


# =============================================================================
# Model Manager Integration (matches CLI system)
# =============================================================================

def map_source_to_backend(source: str) -> str:
    """
    Map model format to runtime backend.
    
    Args:
        source: Model format ("gguf" / "mlx" / "safetensors" / "ollama")
        
    Returns:
        Backend name for ModelConfig
    """
    mapping = {
        "gguf": "llama_cpp",         # GGUF → llama.cpp
        "mlx": "mlx",                # MLX → mlx backend (Apple Silicon)
        "safetensors": "transformers", # Safetensors → transformers
        "ollama": "ollama",          # Ollama → ollama API (legacy)
        "native": "native",          # Native → MLX distributed
    }
    return mapping.get(source.lower(), "llama_cpp")


def get_available_models_with_status() -> Dict[str, List[Dict]]:
    """
    Get models grouped by format, with download status.
    Uses ModelManager to check what's actually downloaded.
    
    Returns:
        {
            "gguf": [{"id": "qwen2.5:1.5b", "name": "...", "downloaded": True, ...}, ...],
            "mlx": [...],
            "safetensors": [...]
        }
    """
    from parishad.models.downloader import ModelManager
    
    try:
        manager = ModelManager()
        downloaded_models = {m.name: m for m in manager.list_models()}
    except Exception as e:
        print(f"Warning: Could not access ModelManager: {e}")
        downloaded_models = {}
    
    # Combine downloaded models + popular models from catalog
    result = {}
    
    for source, models in MODEL_CATALOG.items():
        result[source] = []
        for model in models:
            model_id = model.shortcut
            is_downloaded = model_id in downloaded_models
            
            model_dict = {
                "id": model_id,
                "name": model.name,
                "downloaded": is_downloaded,
                "size": f"{model.size_gb:.1f} GB" if model.size_gb > 0 else "Unknown",
                "quantization": model.quantization,
                "distributor": model.distributor,
                "params": model.params,
                "tags": model.tags,
                "description": model.description,
            }
            
            if is_downloaded:
                dl_model = downloaded_models[model_id]
                model_dict["path"] = str(dl_model.path)
                model_dict["size"] = dl_model.size_human
            
            result[source].append(model_dict)
    
    return result


def ensure_model_available(
    model_id: str, 
    source: str, 
    progress_callback=None,
    cancel_event=None
) -> Optional[Path]:
    """
    Ensure model is downloaded and return path with progress tracking.
    
    Args:
        model_id: Model identifier (e.g., "qwen2.5:1.5b")
        source: Format to download ("gguf" / "mlx" / "safetensors")
        progress_callback: Optional callback for progress updates
        cancel_event: Optional threading.Event to signal cancellation
        
    Returns:
        Path to model file, or None if download fails/cancelled
    """
    from parishad.models.downloader import ModelManager
    
    try:
        manager = ModelManager()
        
        # Check if already present
        path = manager.get_model_path(model_id)
        if path and path.exists():
            return path
        
        # Need to download - wrap progress callback to check for cancellation
        if progress_callback:
            def wrapped_callback(progress):
                # Check if cancelled
                if cancel_event and cancel_event.is_set():
                    raise KeyboardInterrupt("Download cancelled by user")
                progress_callback(progress)
            
            model_info = manager.download(
                model_spec=model_id,
                source=source,
                progress_callback=wrapped_callback
            )
        else:
            model_info = manager.download(
                model_spec=model_id,
                source=source
            )
        
        return model_info.path
        
    except KeyboardInterrupt:
        print("\nDownload cancelled by user")
        return None
    except Exception as e:
        print(f"Error ensuring model availability: {e}")
        return None


# =============================================================================
# Backend and Model Availability Detection
# =============================================================================

def detect_available_backends() -> Dict[str, Tuple[bool, str]]:
    """
    Detect which backends are available on this system.
    
    Returns:
        Dict mapping backend_id -> (available: bool, status_message: str)
    """
    results = {}
    
    # llama.cpp (for GGUF models)
    try:
        import llama_cpp
        results["llama_cpp"] = (True, "llama-cpp-python installed")
    except ImportError:
        results["llama_cpp"] = (False, "llama-cpp-python not installed")
    
    # MLX (for Apple Silicon)
    try:
        import mlx_lm
        results["mlx"] = (True, "mlx-lm installed")
    except ImportError:
        results["mlx"] = (False, "mlx-lm not installed (Mac only)")
    
    # Transformers (for Safetensors)
    try:
        import transformers
        import torch
        results["transformers"] = (True, "Transformers installed")
    except ImportError:
        results["transformers"] = (False, "transformers/torch not installed")
    
    # Ollama (legacy support)
    try:
        if shutil.which("ollama"):
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                results["ollama"] = (True, "Ollama running")
            else:
                results["ollama"] = (False, "Ollama installed but not running")
        else:
            results["ollama"] = (False, "Ollama not installed")
    except Exception as e:
        results["ollama"] = (False, f"Ollama check failed: {e}")
    
    # Native MLX backend (distributed)
    try:
        # Check if native server is reachable
        host = os.environ.get("NATIVE_MLX_HOST", "10.0.0.2")
        port = int(os.environ.get("NATIVE_MLX_PORT", "29500"))
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            results["native"] = (True, f"MLX server at {host}:{port}")
        else:
            results["native"] = (False, f"MLX server unreachable at {host}:{port}")
    except Exception as e:
        results["native"] = (False, f"Native check failed: {e}")
    
    return results


def is_model_available(model_id: str, backend: str) -> bool:
    """
    Check if a specific model is available locally for the given backend.
    
    Args:
        model_id: Model identifier (e.g., "llama3.2:3b", "meta-llama/Llama-3.2-3B")
        backend: Backend name ("llama_cpp", "mlx", "transformers", "ollama", "native", etc.)
    
    Returns:
        True if model is available, False otherwise
    """
    if backend == "ollama":
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse output and check if model_id exists
                # Ollama list format: NAME    ID    SIZE    MODIFIED
                for line in result.stdout.splitlines()[1:]:  # Skip header
                    if line.strip():
                        model_name = line.split()[0]
                        if model_name == model_id or model_name.startswith(model_id):
                            return True
            return False
        except Exception:
            return False
    
    elif backend in ("llama_cpp", "gguf"):
        # Check if GGUF model exists via ModelManager
        try:
            from parishad.models.downloader import ModelManager
            manager = ModelManager()
            model_path = manager.get_model_path(model_id)
            return model_path is not None and model_path.exists()
        except Exception:
            return False
    
    elif backend in ("mlx",):
        # Check if MLX model exists via ModelManager
        try:
            from parishad.models.downloader import ModelManager
            manager = ModelManager()
            model_path = manager.get_model_path(model_id)
            return model_path is not None and model_path.exists()
        except Exception:
            return False
    
    elif backend in ("transformers", "safetensors"):
        # Check if Safetensors/Transformers model exists via ModelManager
        try:
            from parishad.models.downloader import ModelManager
            manager = ModelManager()
            model_path = manager.get_model_path(model_id)
            if model_path and model_path.exists():
                return True
            
            # Also check HF cache
            hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
            cache_dir = Path(hf_home) / "hub"
            
            if not cache_dir.exists():
                return False
            
            # Convert model_id to cache directory format
            # e.g., "meta-llama/Llama-3.2-3B" -> "models--meta-llama--Llama-3.2-3B"
            cache_model_dir = "models--" + model_id.replace("/", "--")
            model_path = cache_dir / cache_model_dir
            
            return model_path.exists() and model_path.is_dir()
        except Exception:
            return False
    
    elif backend == "native":
        # For native backend, check if server is reachable
        try:
            host = os.environ.get("NATIVE_MLX_HOST", "10.0.0.2")
            port = int(os.environ.get("NATIVE_MLX_PORT", "29500"))
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            
            return result == 0
        except Exception:
            return False
    
    # Unknown backend or not implemented
    return False


def get_available_models_for_backend(backend: str) -> List[ModelInfo]:
    """
    Get list of actually available models for a backend.
    
    Args:
        backend: Backend name
    
    Returns:
        List of ModelInfo with available models
    """
    models = []
    
    if backend == "ollama":
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines()[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            model_name = parts[0]
                            size = parts[2] if len(parts) > 2 else "?"
                            
                            # Try to parse size
                            size_gb = 0.0
                            if "GB" in size:
                                try:
                                    size_gb = float(size.replace("GB", ""))
                                except:
                                    pass
                            
                            models.append(ModelInfo(
                                name=model_name.split(":")[0].title(),
                                shortcut=model_name,
                                size_gb=size_gb,
                                description=f"Local Ollama model ({size})",
                                tags="ollama,local",
                                available=True
                            ))
        except Exception:
            pass
    
    elif backend in ("huggingface", "transformers"):
        try:
            hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
            cache_dir = Path(hf_home) / "hub"
            
            if cache_dir.exists():
                for model_dir in cache_dir.iterdir():
                    if model_dir.is_dir() and model_dir.name.startswith("models--"):
                        # Extract model ID from directory name
                        model_id = model_dir.name.replace("models--", "").replace("--", "/")
                        
                        # Estimate size
                        size_gb = 0.0
                        try:
                            total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                            size_gb = total_size / (1024 ** 3)
                        except:
                            pass
                        
                        models.append(ModelInfo(
                            name=model_id.split("/")[-1],
                            shortcut=model_id,
                            size_gb=size_gb,
                            description="Downloaded from HuggingFace",
                            tags="huggingface,local",
                            available=True
                        ))
        except Exception:
            pass
    
    elif backend == "native":
        # Native backend - show the configured remote model
        host = os.environ.get("NATIVE_MLX_HOST", "10.0.0.2")
        port = os.environ.get("NATIVE_MLX_PORT", "29500")
        
        models.append(ModelInfo(
            name="Llama 3.2 1B (Remote)",
            shortcut="mlx-community/Llama-3.2-1B-Instruct-4bit",
            size_gb=0.0,  # Remote model
            description=f"MLX cluster at {host}:{port}",
            tags="native,remote,mlx",
            available=is_model_available("mlx-community/Llama-3.2-1B-Instruct-4bit", "native")
        ))
    
    return models


CSS = """
Screen {
    background: $surface;
}

/* Setup Screen Styles */
#setup-container {
    width: 100%;
    height: 1fr;
    overflow-y: auto;
}

#setup-content {
    width: 100%;
    height: auto;
    padding: 1 2;
}

#setup-header {
    height: auto;
    content-align: center middle;
    padding-bottom: 1;
}

#setup-logo {
    width: 100%;
    height: auto;
    content-align: center middle;
    text-align: center;
}


#step-title {
    text-align: center;
    width: 100%;
    padding: 1;
}

#sabha-step-title {
    text-align: center;
    padding: 1;
}

#model-step-title {
    text-align: center;
    padding: 1;
}

.sabha-grid {
    width: 100%;
    height: auto;
    layout: horizontal;
    padding: 1;
}

/* Maha Sabha - Gold */
.sabha-card-maha {
    width: 1fr;
    height: auto;
    min-height: 10;
    padding: 1 2;
    margin: 0 1;
    background: #1a1500 50%;
}
.sabha-card-maha.selected {
    background: #3a2e00;
    text-style: bold;
}

/* Madhyam Sabha - Blue */
.sabha-card-madhyam {
    width: 1fr;
    height: auto;
    min-height: 10;
    padding: 1 2;
    margin: 0 1;
    background: #0a1520 50%;
}
.sabha-card-madhyam.selected {
    background: #103050;
    text-style: bold;
}

/* Laghu Sabha - Green */
.sabha-card-laghu {
    width: 1fr;
    height: auto;
    min-height: 10;
    padding: 1 2;
    margin: 0 1;
    background: #0a200a 50%;
}
.sabha-card-laghu.selected {
    background: #104010;
    text-style: bold;
}

.sabha-card-title-english {
    text-align: center;
    padding-bottom: 0;
}

.sabha-card-title-hindi {
    text-align: center;
    padding-top: 0;
}

.sabha-card-desc {
    color: $text-muted;
}

.sabha-card-stats {
    color: $text-muted;
}

.model-summary {
    text-align: center;
    padding: 0 1 1 1;
    height: auto;
    width: 1fr;
}

.model-summary-bar {
    height: auto;
    layout: horizontal;
    align: center middle;
    padding: 0 1;
}

.btn-reset {
    width: auto;
    min-width: 10;
    height: 3;
}



/* Model Browser */
#model-browser {
    width: 100%;
    height: 1fr;
    border: round $primary;
    margin: 1 0;
}

.model-tabs {
    height: 3;
    layout: horizontal;
    background: $panel;
    margin-bottom: 1;
}

/* GGUF Tab - Blue */
#tab-gguf {
    width: 1fr;
    border: none;
    background: #1a1a2e;
    color: #4a9eff;
}

#tab-gguf:hover {
    background: #252545;
}

#tab-gguf.active {
    background: #4a9eff;
    color: #ffffff;
}

/* MLX Tab - Orange/Red */
#tab-mlx {
    width: 1fr;
    border: none;
    background: #2a1a1a;
    color: #ff6b35;
}

#tab-mlx:hover {
    background: #3a2525;
}

#tab-mlx.active {
    background: #ff6b35;
    color: #ffffff;
}

/* Safetensors Tab - Green */
#tab-safetensors {
    width: 1fr;
    border: none;
    background: #1a2e1e;
    color: #00cc88;
}

#tab-safetensors:hover {
    background: #254530;
}

#tab-safetensors.active {
    background: #00cc88;
    color: #ffffff;
}


#model-search {
    margin: 1 0;
    border: round $primary;
}

#model-search:focus {
    border: round $accent;
}

#model-browser-container {
    width: 100%;
    height: 25;
    min-height: 15;
    border: round $primary;
    margin: 1 0;
}

.model-list {
    height: 100%;
    width: 100%;
}

.model-item {
    height: 3;
    padding: 0 1;
    border: round $panel;
    margin: 0;
}

.model-item:hover {
    border: round $accent;
    background: $boost;
}

.model-item.selected {
    border: double $success;
    background: $boost;
}


/* Action Buttons */
#action-bar {
    height: 5;
    layout: horizontal;
    padding: 1 2;
    margin-top: 1;
    background: $panel;
    border-top: solid $primary;
}

#btn-skip {
    width: 1fr;
    margin: 0 1;
    height: 3;
}

#btn-continue {
    width: 1fr;
    margin: 0 1;
    height: 3;
}

/* Chat Screen Styles */
#welcome {
    height: auto;
    padding: 1;
}

#logo {
    height: auto;
    max-height: 12;
    content-align: center middle;
    text-align: center;
    padding-top: 1;
}

#tips {
    height: auto;
    max-height: 4;
    color: $text-muted;
    padding: 0 1;
    content-align: center middle;
}

#role-progress {
    height: 1;
    margin: 0 1;
}

#chat-area {
    height: 1fr;
    min-height: 10;
    padding: 0 1;
    overflow-y: auto;
    scrollbar-size-vertical: 2;
}

#input-box {
    height: 3;
    border: round $primary;
    padding: 0 1;
    margin: 0 1 1 1;
    layout: horizontal;
}

#input-box:focus-within {
    border: round $accent;
}

#prompt-prefix {
    width: auto;
    color: $accent;
}

#prompt-input {
    width: 1fr;
    border: none;
    background: transparent;
}

#status {
    height: 1;
    padding: 0 1;
}

#prompt-input:focus {
    border: none;
}

#status {
    height: 1;
    padding: 0 1;
    color: $text-muted;
}

#role-progress {
    height: 1;
    text-align: center;
    padding: 0 1;
}
"""


# =============================================================================
# Setup Screen
# =============================================================================

class SabhaCard(Static):
    """A clickable Sabha selection card."""
    
    class Selected(Message):
        """Message when Sabha is selected."""
        def __init__(self, sabha: SabhaConfig) -> None:
            self.sabha = sabha
            super().__init__()
    
    def __init__(self, sabha: SabhaConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sabha = sabha
        self.is_selected = False
    
    def compose(self) -> ComposeResult:
        yield Static(
            f"{self.sabha.emoji} [bold]{self.sabha.name}[/bold]",
            classes="sabha-card-title-english"
        )

        yield Static(self.sabha.description, classes="sabha-card-desc")
        yield Static(
            f"[dim]Roles:[/dim] {self.sabha.roles}  "
            f"[dim]RAM:[/dim] {self.sabha.ram_gb}GB  "
            f"[dim]Speed:[/dim] {self.sabha.speed}",
            classes="sabha-card-stats"
        )
    
    def on_click(self) -> None:
        self.post_message(self.Selected(self.sabha))
    
    def select(self) -> None:
        self.is_selected = True
        self.add_class("selected")
    
    def deselect(self) -> None:
        self.is_selected = False
        self.remove_class("selected")


class ModelCard(Static):
    """A clickable model selection card."""
    
    class Selected(Message):
        """Message when model is selected."""
        def __init__(self, model: ModelInfo) -> None:
            self.model = model
            super().__init__()
    
    def __init__(self, model: ModelInfo, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.is_selected = False
        self.is_installed = self._check_installed()
    
    def _check_installed(self) -> bool:
        """Check if this model is already downloaded."""
        try:
            from parishad.models.downloader import ModelManager
            manager = ModelManager()
            path = manager.get_model_path(self.model.shortcut)
            return path is not None and path.exists()
        except Exception:
            return False
    
    def compose(self) -> ComposeResult:
        m = self.model
        # Show green tick if installed
        installed_icon = "[green]✓[/green] " if self.is_installed else ""
        yield Static(
            f"{installed_icon}[bold]{m.name}[/bold] [cyan]{m.params}[/cyan] "
            f"[dim]({m.size_gb:.1f}GB)[/dim] [yellow]{m.distributor}[/yellow]"
        )

    
    def on_click(self) -> None:
        self.post_message(self.Selected(self.model))
    
    def select(self) -> None:
        self.is_selected = True
        self.add_class("selected")
    
    def deselect(self) -> None:
        self.is_selected = False
        self.remove_class("selected")



class SetupScreen(Screen):
    """Setup wizard screen for first-time configuration."""
    
    BINDINGS = [
        Binding("escape", "skip", "Skip Setup"),
        Binding("enter", "confirm", "Confirm"),
    ]
    
    def __init__(self, initial_config: Optional[ParishadConfig] = None) -> None:
        super().__init__()
        self.initial_config = initial_config  # For re-setup scenarios
        self.selected_sabha: Optional[SabhaConfig] = None
        self.selected_models: Dict[str, ModelInfo] = {}  # Map slot_name -> model
        self.current_slot_idx: int = 0
        self.current_source = "gguf"  # Default to GGUF format
        self.step = 1  # 1 = Sabha, 2 = Model
        self.is_downloading = False # Lock to prevent concurrent setup
        
        # Pre-populate from initial_config if provided
        if initial_config:
            # Find sabha config
            for sabha in SABHAS:
                if sabha.id == initial_config.sabha:
                    self.selected_sabha = sabha
                    break
            self.current_source = initial_config.backend
    
    def compose(self) -> ComposeResult:
        # Everything in one scrollable container
        with ScrollableContainer(id="setup-container"):
            with Vertical(id="setup-content"):
                # Logo and welcome message (scrolls with content)
                yield Static(LOGO, id="setup-logo", markup=True)
                yield Static(
                    "[bold]Welcome to Parishad![/bold]\n"
                    "[dim]Let's set up your AI council.[/dim]",
                    id="step-title"
                )
                
                # Step 1: Sabha Selection
                yield Static("[bold]Step 1:[/bold] Choose your Sabha (Council)", id="sabha-step-title")
                yield Horizontal(
                    *[SabhaCard(sabha, classes=f"sabha-card-{sabha.id}") for sabha in SABHAS],
                    classes="sabha-grid"
                )
        
                
                # Step 2: Model Selection
                yield Static("\n[bold]Step 2:[/bold] Select models (Waiting for Sabha...)", id="model-step-title")
                
                # Selected models summary with reset button
                yield Horizontal(
                    Static("", id="model-summary", classes="model-summary"),
                    Button("🔄 Reset", id="btn-reset-models", variant="default", classes="btn-reset"),
                    classes="model-summary-bar"
                )
                
                # Model browser with format tabs
                with Container(id="model-browser-container"):
                    yield Horizontal(
                        Button("🔷 GGUF", id="tab-gguf", classes="model-tab active"),
                        Button("🍎 MLX", id="tab-mlx", classes="model-tab"),
                        Button("🛡️ Safetensors", id="tab-safetensors", classes="model-tab"),
                        classes="model-tabs"
                    )
                    
                    # Search bar
                    yield Input(placeholder="🔍 Search models...", id="model-search")
        
                    yield ScrollableContainer(
                        *[ModelCard(m, classes="model-item") for m in MODEL_CATALOG.get("gguf", [])],
                        id="model-list",
                        classes="model-list"
                    )
        
                # Progress bar (initially hidden via CSS)
                yield ProgressBar(total=100, show_eta=True, id="download-progress")
                
                # Action buttons - inside scrollable area
                yield Horizontal(
                    Button("Skip (use defaults)", id="btn-skip", variant="default"),
                    Button("Continue →", id="btn-continue", variant="primary"),
                    id="action-bar"
                )
    
    def on_mount(self) -> None:
        # Pre-select recommended Sabha (Laghu)
        for card in self.query(SabhaCard):
            if card.sabha.id == "laghu":
                card.select()
                self.selected_sabha = card.sabha
                break
        
        # Pre-select first model
        model_cards = list(self.query(ModelCard))
        if model_cards:
            model_cards[0].select()
            # Don't set model yet, wait for user click
    
    @on(SabhaCard.Selected)
    def handle_sabha_selected(self, event: SabhaCard.Selected) -> None:
        for card in self.query(SabhaCard):
            card.deselect()
        event.sabha  # The sabha that was selected
        # Find the card that sent the message and select it
        for card in self.query(SabhaCard):
            if card.sabha.id == event.sabha.id:
                card.select()
                self.selected_sabha = event.sabha
                self.current_slot_idx = 0
                self.selected_models = {}
                self._update_model_step_title()
                # Enable/disable continue button
                self.query_one("#btn-continue", Button).disabled = True
                break
    
    @on(ModelCard.Selected)
    def handle_model_selected(self, event: ModelCard.Selected) -> None:
        if not self.selected_sabha:
            return

        # Record selection for current slot
        slots = self.selected_sabha.model_slots
        if self.current_slot_idx < len(slots):
            current_slot = slots[self.current_slot_idx]
            self.selected_models[current_slot] = event.model
            
            # Move to next slot
            self.current_slot_idx += 1
            
            # Update summary display
            self._update_model_summary()
            
            # Check if done
            if self.current_slot_idx >= len(slots):
                self.query_one("#btn-continue", Button).disabled = False
                self.query_one("#model-step-title", Static).update(
                    f"\n[bold]Step 2:[/bold] All models selected! ([green]✓ Ready[/green])"
                )
            else:
                self._update_model_step_title()
    
    def _update_model_step_title(self) -> None:
        if not self.selected_sabha:
            return
            
        slots = self.selected_sabha.model_slots
        if self.current_slot_idx < len(slots):
            current_slot = slots[self.current_slot_idx]
            self.query_one("#model-step-title", Static).update(
                f"\n[bold]Step 2:[/bold] Select [cyan]{current_slot.upper()}[/cyan] model "
                f"({self.current_slot_idx + 1}/{len(slots)})"
            )
            
            # Reset card selection visually for next pick
            for card in self.query(ModelCard):
                card.deselect()
    
    def _update_model_summary(self) -> None:
        """Update the selected models summary text."""
        if not self.selected_models:
            self.query_one("#model-summary", Static).update("")
            return
            
        summary = []
        for slot, model in self.selected_models.items():
            summary.append(f"[dim]{slot.title()}:[/dim] [cyan]{model.name}[/cyan]")
        
        self.query_one("#model-summary", Static).update("  ".join(summary))
    
    @on(Button.Pressed, "#btn-reset-models")
    def reset_model_selection(self) -> None:
        """Reset all model selections and start over."""
        if not self.selected_sabha:
            return
        
        self.current_slot_idx = 0
        self.selected_models = {}
        self._update_model_step_title()
        self._update_model_summary()
        self.query_one("#btn-continue", Button).disabled = True
    
    @on(Button.Pressed, "#tab-gguf")
    def show_gguf(self) -> None:
        self._switch_tab("gguf")
    
    @on(Button.Pressed, "#tab-mlx")
    def show_mlx(self) -> None:
        self._switch_tab("mlx")
    
    @on(Button.Pressed, "#tab-safetensors")
    def show_safetensors(self) -> None:
        self._switch_tab("safetensors")
    
    @on(Input.Changed, "#model-search")
    def on_search_changed(self, event: Input.Changed) -> None:
        """Filter models based on search query."""
        self._update_model_list(event.value)
    
    def _switch_tab(self, source: str) -> None:
        self.current_source = source
        
        # Update tab styling
        for btn in self.query(".model-tab"):
            btn.remove_class("active")
        self.query_one(f"#tab-{source}").add_class("active")
        
        # Clear search and update list
        search_input = self.query_one("#model-search", Input)
        search_input.value = ""
        self._update_model_list("")
    
    def _update_model_list(self, search_query: str = "") -> None:
        """Update model list with optional search filter."""
        model_list = self.query_one("#model-list")
        model_list.remove_children()
        
        models = MODEL_CATALOG.get(self.current_source, [])
        
        # Filter by search query
        if search_query:
            query = search_query.lower()
            models = [
                m for m in models
                if query in m.name.lower()
                or query in m.description.lower()
                or query in m.distributor.lower()
                or query in m.params.lower()
                or (m.tags and any(query in tag for tag in m.tags))
            ]
        
        # Add filtered models
        for model in models:
            model_list.mount(ModelCard(model, classes="model-item"))
        
        # Select first model if any
        model_cards = list(self.query(ModelCard))
        if model_cards:
            model_cards[0].select()
            self.selected_model = model_cards[0].model

    
    @on(Button.Pressed, "#btn-skip")
    def action_skip(self) -> None:
        """Handle skip/abort - preserve initial config or use defaults."""
        if self.initial_config:
            # Re-setup scenario - abort and keep existing config
            self.dismiss(self.initial_config)
        else:
            # First-run scenario - create default config
            default_config = ParishadConfig(
                sabha="laghu",
                backend="ollama",  # Default to Ollama (matches CLI)
                model="qwen2.5:1.5b",  # Small Ollama model
                cwd=str(Path.cwd())
            )
            save_parishad_config(default_config)
            self.dismiss(default_config)
    
    @on(Button.Pressed, "#btn-continue")
    def action_confirm(self) -> None:
        """Handle confirm - download models then save config and return."""
        # Strict concurrency check
        if self.is_downloading:
            self.notify("Setup already in progress. Please wait...", severity="warning")
            return

        if self.selected_sabha and len(self.selected_models) >= len(self.selected_sabha.model_slots):
            # Create ParishadConfig from selections
            # Store format (gguf/mlx/safetensors) for backend mapping
            primary_model = list(self.selected_models.values())[0].shortcut if self.selected_models else "qwen2.5:1.5b"
            
            new_config = ParishadConfig(
                sabha=self.selected_sabha.id,
                backend=self.current_source,  # Format: gguf/mlx/safetensors
                model=primary_model,  # Model ID for ModelManager
                cwd=str(Path.cwd())
            )
            
            # Download models before saving config
            self.run_worker(self._async_download_models(new_config), exclusive=True)
    
    async def _async_download_models(self, config: ParishadConfig) -> None:
        """Download selected models asynchronously using CLI's ModelManager, then save config."""
        from parishad.models.downloader import ModelManager
        import asyncio
        
        manager = ModelManager()
        loop = asyncio.get_event_loop()
        
        # Collect all unique model shortcuts that need checking
        models_to_download = []
        for slot, model_info in self.selected_models.items():
            # Check if model already exists using get_model_path (correct API)
            model_path = manager.get_model_path(model_info.shortcut)
            if model_path is None or not model_path.exists():
                # Model not found or file missing, need to download
                models_to_download.append(model_info)
        
        # UI Feedback
        btn = self.query_one("#btn-continue", Button)
        pbar = self.query_one("#download-progress", ProgressBar)
        
        original_label = str(btn.label)
        btn.disabled = True
        btn.label = "⏳ Setting up..."
        
        # Show progress bar
        pbar.display = True
        pbar.update(total=100, progress=0)
        
        self.is_downloading = True # Set lock
        
        # Debug: Log what we're doing
        db_path = Path.home() / "parishad_debug.log"
        with open(db_path, "a") as f:
            f.write(f"\n=== SETUP CONTINUE CLICKED ===\n")
            f.write(f"Selected models: {list(self.selected_models.keys())}\n")
            f.write(f"Models to download: {[m.name for m in models_to_download]}\n")
        
        try:
            download_errors = []
            
            # If no models need downloading, just save config and exit
            if not models_to_download:
                with open(db_path, "a") as f:
                    f.write(f"All models installed - saving config and exiting\n")
                self.notify("✓ All models already available!", severity="information", timeout=3)
                if save_parishad_config(config):
                    with open(db_path, "a") as f:
                        f.write(f"Config saved successfully, dismissing\n")
                    self.dismiss(config)
                else:
                    self.notify("Failed to save configuration", severity="error", timeout=5)
                return # Exit early if nothing to download
            
            # Download each missing model
            for i, model_info in enumerate(models_to_download):
                self.notify(f"Downloading {model_info.name}...\nPlease wait (this may take a while)", timeout=10)
                
                # Reset progress for new file
                pbar.update(total=100, progress=0)
                
                # This is EXACTLY what CLI does in main.py:download_model
                def _do_download():
                    """Execute download in thread pool (production-safe)."""
                    # DEBUG LOGGING
                    db_path = Path.home() / "parishad_debug.log"
                    with open(db_path, "a") as f:
                        f.write(f"DEBUG: Starting download for {model_info.name} from {model_info.source}\n")
                    
                    def _progress(p):
                        """Track download progress and update TUI safely."""
                        if p.total_bytes > 0:
                            # Calculate percentage
                            percent = (p.downloaded_bytes / p.total_bytes) * 100
                            # Update TUI from thread
                            self.app.call_from_thread(pbar.update, progress=percent)
                        
                    try:
                        res = manager.download(
                            model_spec=model_info.shortcut,
                            source="huggingface" if model_info.source == "huggingface" else model_info.source, 
                            progress_callback=_progress
                        )
                        with open(db_path, "a") as f:
                            f.write(f"DEBUG: Download success: {res}\n")
                    except Exception as e:
                        with open(db_path, "a") as f:
                            f.write(f"DEBUG: Download FAILED: {e}\n")
                        import traceback
                        with open(db_path, "a") as f:
                            traceback.print_exc(file=f)
                        return False # Explicit failure return

                    return True # Explicit success return
                
                # Run in thread pool to avoid blocking TUI
                success = await loop.run_in_executor(None, _do_download)
                
                if not success:
                    download_errors.append(f"{model_info.name} failed (check ~/parishad_debug.log)")
                    
            # Check results
            if download_errors:
                error_msg = "\n".join(download_errors)
                self.notify(f"Download errors occurred:\n{error_msg}", severity="error", timeout=10)
            else:
                pbar.update(total=100, progress=100) # Show full completion
                self.notify("Setup complete! Saving configuration...", timeout=5)
                # Success - save config and proceed
                if save_parishad_config(config):
                    self.dismiss(config)
                else:
                    self.notify("Error saving configuration! Check permissions.", severity="error")

        except Exception as e:
            self.notify(f"Critical Error: {str(e)}", severity="error")
            with open(Path.home() / "parishad_debug.log", "a") as f:
                f.write(f"DEBUG: Critical Outer Exception: {e}\n")
        
        finally:
            # Always reset UI state and lock
            btn.disabled = False
            btn.label = original_label
            pbar.display = False # Hide progress bar
            self.is_downloading = False
    
    def _save_config(self, use_defaults: bool = False) -> None:
        """Deprecated: Config is now saved via ParishadConfig.
        
        This method is kept for backward compatibility but should not be used.
        Use save_parishad_config() instead.
        """
        pass


# =============================================================================
# Sabha Progress Display
# =============================================================================

# Role metadata for display
ROLE_INFO = {
    "darbari": {"emoji": "🏛️", "name": "Darbari", "desc": "Analyzing query"},
    "majumdar": {"emoji": "📋", "name": "Majumdar", "desc": "Creating plan"},
    "sainik": {"emoji": "⚔️", "name": "Sainik", "desc": "Implementing"},
    "prerak": {"emoji": "🔍", "name": "Prerak", "desc": "Checking"},
    "raja": {"emoji": "👑", "name": "Raja", "desc": "Deciding"},
}

CORE_ROLES = ["darbari", "majumdar", "sainik", "prerak", "raja"]


class RoleProgressBar(Static):
    """Display Sabha role execution progress."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_role = None
        self.completed_roles = []
    
    def set_active(self, role_name: str) -> None:
        """Set the currently active role."""
        self.active_role = role_name
        self._update_display()
    
    def mark_complete(self, role_name: str) -> None:
        """Mark a role as complete."""
        if role_name not in self.completed_roles:
            self.completed_roles.append(role_name)
        self._update_display()
    
    def reset(self) -> None:
        """Reset progress."""
        self.active_role = None
        self.completed_roles = []
        self._update_display()
    
    def _update_display(self) -> None:
        """Update the progress display."""
        parts = []
        for role in CORE_ROLES:
            info = ROLE_INFO[role]
            if role in self.completed_roles:
                parts.append(f"[green]{info['emoji']}[/green]")
            elif role == self.active_role:
                parts.append(f"[yellow]{info['emoji']} {info['name']}...[/yellow]")
            else:
                parts.append(f"[dim]{info['emoji']}[/dim]")
        
        self.update(" ".join(parts))




class CommandSuggester(Suggester):
    """Suggests slash commands and arguments for Parishad CLI."""
    
    def __init__(self):
        super().__init__(use_cache=False)
        self.cached_models = None
    
    async def get_suggestion(self, value: str) -> Optional[str]:
        """Get best suggestion (first candidate)."""
        candidates = await self.get_candidates(value)
        return candidates[0] if candidates else None

    async def get_candidates(self, value: str) -> List[str]:
        """Get all matching candidates for the current input."""
        candidates = []
        # Early return only if not a command AND not a file reference
        if not value.startswith("/") and "@" not in value:
            return candidates
            
        value_lower = value.lower()
        
        # 1. Command arguments
        if value_lower.startswith("/sabha "):
            current_arg = value_lower[7:]
            options = ["laghu", "madhyam", "maha"]
            return [f"/sabha {opt}" for opt in options if opt.startswith(current_arg)]
            
        if value_lower.startswith("/model "):
            if self.cached_models is None:
                # Setup basic list if manager fails
                self.cached_models = []
                try:
                    from parishad.models.downloader import ModelManager
                    # Get model names
                    self.cached_models = [m.name for m in ModelManager().list_models()]
                except Exception:
                     pass
            
            if self.cached_models:
               current_arg = value[7:]
               return [f"/model {m}" for m in self.cached_models if m.startswith(current_arg)]
            return []

        if value_lower.startswith("/assign "):
            # Logic for assign args
            # 1. Load models if needed
            if self.cached_models is None:
                self.cached_models = []
                try:
                    from parishad.models.downloader import ModelManager
                    self.cached_models = [m.name for m in ModelManager().list_models()]
                except Exception:
                     pass

            parts = value.split(" ")
            current_token = parts[-1]
            prefix_tokens = " ".join(parts[:-1]) 
            
            suggestions = []
            slots = ["big=", "mid=", "small=", "planner=", "judge="]
            
            if "=" in current_token:
                # Suggesting value for specific slot
                # e.g. big=lla -> big=llama3
                key, val_prefix = current_token.split("=", 1)
                
                # Filter models matches
                model_matches = [m for m in self.cached_models if m.startswith(val_prefix)]
                for m in model_matches:
                    suggestions.append(f"{prefix_tokens} {key}={m}")
            else:
                # Suggesting slot OR model (Smart Mode)
                # 1. Slots
                for s in slots:
                    if s.startswith(current_token):
                         suggestions.append(f"{prefix_tokens} {s}")
                
                # 2. Models (Smart Mode)
                model_matches = [m for m in self.cached_models if m.startswith(current_token)]
                for m in model_matches:
                    suggestions.append(f"{prefix_tokens} {m}")
                    
            return suggestions

        # 2. @-file Autocomplete (Phase 13)
        # Check if the *last token* starts with @
        last_token = value.split(" ")[-1]
        if last_token.startswith("@"):
             prefix = value[: -len(last_token)] # Everything before the token
             partial_path = last_token[1:] # Strip @
             
             try:
                 # Resolve directory and search pattern
                 if "/" in partial_path:
                     dir_part, file_part = partial_path.rsplit("/", 1)
                     search_dir = Path.cwd() / dir_part
                     glob_pattern = f"{file_part}*"
                     display_dir = f"{dir_part}/"
                 else:
                     search_dir = Path.cwd()
                     glob_pattern = f"{partial_path}*"
                     display_dir = ""
                
                 if search_dir.exists() and search_dir.is_dir():
                     matches = []
                     # List files and dirs
                     for item in search_dir.glob(glob_pattern):
                         # Skip hidden files unless explicitly typed "."
                         if item.name.startswith(".") and not partial_path.startswith("."):
                             continue
                             
                         # Append / to directories
                         suffix = "/" if item.is_dir() else ""
                         candidate = f"@{display_dir}{item.name}{suffix}"
                         matches.append(candidate)
                     
                     # Sort: Directories first, then files
                     matches.sort(key=lambda x: (not x.endswith("/"), x))
                     
                     # Limit to 10 suggestions to avoid clutter
                     return [f"{prefix}{m}" for m in matches[:15]]
                     
             except Exception:
                 pass
             return []

        # 3. Top-level commands
        commands = [
            "/help", "/exit", "/clear", "/config", "/setup", 
            "/model", "/sabha", "/redownload", "/assign", "/scan"
        ]
        return [cmd for cmd in commands if cmd.startswith(value_lower)]


class ShellInput(Input):
    """Input widget with shell-like suggestion cycling (Up/Down) and Tab completion."""
    
    BINDINGS = [
        Binding("up", "cycle_suggestion(-1)", "Previous", show=False),
        Binding("down", "cycle_suggestion(1)", "Next", show=False),
        Binding("tab", "accept_suggestion", "Accept", show=False),
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cycle_index = -1
        self.current_candidates = []
        self.original_prefix = ""
        self.is_cycling = False
        
    def watch_value(self, value: str) -> None:
        """Reset cycling when user types manually."""
        if not self.is_cycling:
            self.original_prefix = value
            self.cycle_index = -1
            self.current_candidates = []

    async def action_cycle_suggestion(self, delta: int) -> None:
        """Cycle through suggestions by updating the value directly."""
        prefix = self.original_prefix
        if not prefix:
            return

        # Refresh candidates if needed
        if not self.current_candidates:
            if hasattr(self.suggester, "get_candidates"):
                 self.current_candidates = await self.suggester.get_candidates(prefix)
            else:
                 self.current_candidates = []
            
        if not self.current_candidates:
            return

        # Advance index
        if self.cycle_index == -1:
            self.cycle_index = 0 if delta > 0 else len(self.current_candidates) - 1
        else:
            self.cycle_index = (self.cycle_index + delta) % len(self.current_candidates)
            
        # Update value directly (Zsh style)
        self.is_cycling = True
        self.value = self.current_candidates[self.cycle_index]
        self.cursor_position = len(self.value)
        self.is_cycling = False
        
    async def action_accept_suggestion(self) -> None:
        """Accept current suggestion (Tab)."""
        # If we are already cycling, the value is set, just move cursor
        if self.cycle_index != -1:
            self.cursor_position = len(self.value)
            return

        # Otherwise, try to fetch suggestion manually since we can't access self.suggestion
        if self.suggester and self.value:
            sug = await self.suggester.get_suggestion(self.value)
            if sug:
                self.value = sug
                self.cursor_position = len(self.value)



class ParishadApp(App):
    """Parishad Code TUI Application."""
    
    # Custom message for opening setup screen from worker thread
    class OpenSetup(Message):
        """Message to open setup screen from worker thread."""
        pass
    
    # Custom messages for non-blocking thread worker communication
    class LogMessage(Message):
        """Non-blocking log message from worker thread."""
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()
    
    class SabhaResultReady(Message):
        """Non-blocking message when Sabha result is ready."""
        def __init__(self, trace) -> None:
            self.trace = trace
            super().__init__()
    
    class SabhaError(Message):
        """Non-blocking message when Sabha encounters an error."""
        def __init__(self, error: Exception, traceback_str: str) -> None:
            self.error = error
            self.traceback_str = traceback_str
            super().__init__()
    
    class CouncilReady(Message):
        """Non-blocking message when council initialization completes."""
        def __init__(self, success: bool, profile: str = "", error_msg: str = "") -> None:
            self.success = success
            self.profile = profile
            self.error_msg = error_msg
            super().__init__()
    
    class WorkerComplete(Message):
        """Non-blocking message when any worker completes."""
        def __init__(self, worker_type: str) -> None:
            self.worker_type = worker_type
            super().__init__()
    
    CSS = CSS
    SCREENS = {"setup": SetupScreen}
    BINDINGS = [
        Binding("ctrl+c", "quit", "Exit", show=False),
        Binding("ctrl+l", "clear", "Clear", show=False),
    ]
    
    def __init__(self, model: str = None, sabha: str = None, backend: str = None, cwd: str = "."):
        super().__init__()
        self.cwd = Path(cwd).resolve()
        self.council = None
        self.ctrl_c_pressed = False
        self.download_cancel_event = None  # Track download cancellation
        self.download_progress_line = None  # Track last progress line for updates
        self._initializing = False  # Prevent concurrent initialization
        self._processing_query = False  # Prevent concurrent query processing
        
        # CRITICAL FOR WINDOWS: Thread-safe result queue for native threading
        # This bypasses Textual's worker system which causes freezes on Windows
        self._result_queue = queue.Queue()
        self._worker_thread = None
        self._subprocess = None  # For subprocess-based inference
        
        # Load config from disk
        self.config = load_parishad_config()
        
        # Apply overrides from CLI if provided
        if self.config:
            self.model = model or self.config.model
            self.backend = backend or self.config.backend
            self.sabha = sabha or self.config.sabha
        else:
            # No config file - use CLI params or defaults
            self.model = model or "llama3.2:3b"
            self.backend = backend or "ollama"
            self.sabha = sabha or "laghu"
    
    def _load_config(self) -> dict:
        """Deprecated: Use load_parishad_config() instead.
        
        This method is kept for backward compatibility.
        """
        config = load_parishad_config()
        if config:
            return config.to_dict()
        return {}
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        cwd_str = str(self.cwd)
        if len(cwd_str) > 40:
            cwd_str = "~" + cwd_str[-39:]
        
        # Header section (fixed height)
        yield Static(LOGO, id="logo", markup=True)
        yield Static(
            f"\n[dim]Tips: Ask questions, edit files, run commands. /help for more.[/dim]\n"
            f"[dim]{self.model} · {cwd_str}[/dim]",
            id="tips",
            markup=True
        )
        yield RoleProgressBar(id="role-progress")
        
        # Chat area (takes remaining space)
        yield RichLog(id="chat-area", markup=True, wrap=True, auto_scroll=True, highlight=True)
        
        # Input box (fixed at bottom)
        yield Container(
            Static("> ", id="prompt-prefix"),
            ShellInput(placeholder="Type your message...", id="prompt-input", suggester=CommandSuggester()),
            id="input-box"
        )
        yield Static("[dim]? for help · Ctrl+C to exit[/dim]", id="status")
    
    def on_mount(self) -> None:
        """Focus input on mount and handle startup flow."""
        # Check if we need setup
        # Requires setup if:
        # 1. No config exists OR
        # 2. setup_complete is False OR
        # 3. Config exists but Sabha/Model not selected
        needs_setup = False
        if not self.config:
            needs_setup = True
        elif not self.config.setup_complete:
            needs_setup = True
        elif not self.config.sabha or not self.config.model:
            needs_setup = True
            
        if needs_setup:
            # Show setup screen (Sabha selection + Model browser)
            self.push_screen(SetupScreen(self.config), callback=self._on_setup_complete)
        else:
            # Config complete - go straight to chat
            self._initialize_chat()
    
    def on_parishad_app_open_setup(self, message: OpenSetup) -> None:
        """Handle OpenSetup message - open setup screen."""
        self.push_screen(SetupScreen(), callback=self._on_setup_complete)
    
    def _on_setup_complete(self, config: Optional[ParishadConfig]) -> None:
        """Callback when setup is completed or aborted."""
        if config:
            # Setup completed with new config
            self.config = config
            self.model = config.model
            self.backend = config.backend
            self.sabha = config.sabha
            self.cwd = Path(config.cwd) if config.cwd else Path.cwd()
            
            # Initialize chat with new config
            self._initialize_chat()
        else:
            # Setup aborted with no previous config - should not happen
            # (action_skip now always returns a config)
            self.exit()
    
    def _initialize_chat(self) -> None:
        """Initialize chat interface after config is ready."""
        self.query_one("#prompt-input", Input).focus()
        
        # Prevent concurrent initialization
        if self._initializing:
            return
        
        # Run model loading in a thread worker to avoid freezing UI on Windows
        # CRITICAL: Using thread=True ensures blocking model loading doesn't freeze the TUI
        self.run_worker(self._initialize_council_thread_worker, thread=True, exclusive=True)
    
    async def _async_initialize_council(self) -> None:
        """Async worker to initialize Sabha council without blocking UI."""
        if self._initializing:
            self.log_message("[yellow]Already initializing...[/yellow]\n")
            return
        
        self._initializing = True
        
        try:
            from parishad.orchestrator.engine import Parishad
            from parishad.config.user_config import load_user_config
            import asyncio
            
            self.log_message("[cyan]🔄 Initializing Sabha council...[/cyan]\n")
            
            # Load user config for profile (same as CLI run does)
            user_cfg = load_user_config()
            profile = user_cfg.default_profile
            mode = user_cfg.default_mode
            
            self.log_message(f"[dim]  • Profile: {profile}[/dim]\n")
            self.log_message(f"[dim]  • Mode: {mode}[/dim]\n")
            
            # Get pipeline config from Sabha selection
            if self.config:
                config_name = self.config.get_pipeline_config()
                self.log_message(f"[dim]  • Pipeline: {config_name}[/dim]\n")
            else:
                config_name = "core"  # Default fallback
                self.log_message(f"[dim]  • Pipeline: {config_name} (default)[/dim]\n")
            
            self.log_message(f"[yellow]⏳ Loading models (this may take 30-60 seconds)...[/yellow]\n")
            
            # Initialize Parishad exactly like CLI run does
            # CRITICAL: Pass model_config_path=None so it uses profiles + models.yaml
            loop = asyncio.get_event_loop()
            
            self.log_message(f"[dim]  • Creating Parishad engine...[/dim]\n")
            
            # Add timeout to prevent indefinite freezing when backend is unavailable
            try:
                # Build user_forced_config from model_map
                user_forced_config = {}
                if self.config.model_map:
                    # Initialize manager to resolve paths
                    from parishad.models.downloader import ModelManager
                    model_manager = ModelManager()
                    
                    msg_backend = self.config.backend or "ollama"
                    
                    for slot, model_id in self.config.model_map.items():
                        # Default to current config backend
                        current_backend = msg_backend
                        model_file = None
                        
                        # Check if it's a known model to resolve backend/path
                        model_info = model_manager.registry.get(model_id)
                        if model_info:
                             # Handle Enum comparison correctly
                             source = model_info.source.value if hasattr(model_info.source, "value") else str(model_info.source)
                             
                             if source == "local":
                                  current_backend = "llama_cpp"
                                  model_file = str(model_info.path)
                             elif source == "ollama":
                                  current_backend = "ollama"
                             elif source == "mlx":
                                  current_backend = "mlx"
                        else:
                            # Fallback heuristics if not in registry
                            if model_id.startswith("local:"):
                                current_backend = "llama_cpp"
                            elif model_id.startswith("ollama:") or ":" in model_id:
                                current_backend = "ollama"

                        user_forced_config[slot] = {
                            "model_id": model_id,
                            "backend_type": current_backend
                        }
                        if model_file:
                             user_forced_config[slot]["model_file"] = model_file

                self.council = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: Parishad(
                            config=config_name,
                            model_config_path=None,  # Let engine use profiles + models.yaml
                            profile=profile,
                            pipeline_config_path=None,
                            trace_dir=None,
                            mock=False,
                            stub=False,
                            mode=mode,
                            user_forced_config=user_forced_config or None,
                            no_retry=False,
                        )
                    ),
                    timeout=120.0  # 2 minute timeout for model loading
                )
            except asyncio.TimeoutError:
                self.log_message(
                    "[red]✗ Model loading timed out (120 seconds)[/red]\n"
                    "[yellow]⚠ The backend may not be running or model download is stalled.[/yellow]\n"
                    "[dim]Hints:[/dim]\n"
                    "[dim]  • Check if Ollama is running: ollama serve[/dim]\n"
                    "[dim]  • Verify model is downloaded: parishad models list[/dim]\n"
                    "[dim]  • Try /setup to reconfigure[/dim]\n"
                )
                self.council = None
                return
            except Exception as e:
                # Production-grade error handling with specific VRAM/GPU messages
                error_msg = str(e)
                self.log_message(f"[red]✗ Error loading Sabha council:[/red]\n")
                
                # Check for specific error types and provide helpful messages
                if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                    self.log_message(
                        "[yellow]⚠️  GPU Memory Issue Detected[/yellow]\n"
                        "[dim]The selected model is too large for your GPU.[/dim]\n"
                        "[dim]\n"
                        "[bold]Solutions:[/bold]\n"
                        "[dim]  1. Try a smaller model (0.5B, 1B, or 3B parameters)[/dim]\n"
                        "[dim]  2. Use more aggressive quantization (Q2, Q3, Q4)[/dim]\n"
                        "[dim]  3. Close other GPU-intensive applications[/dim]\n"
                        "[dim]  4. Check VRAM usage: nvidia-smi[/dim]\n"
                        "[dim]\n"
                        "[dim]Use /setup to select a different model.[/dim]\n"
                    )
                elif "not found" in error_msg.lower() or "no such file" in error_msg.lower():
                    self.log_message(
                        "[yellow]⚠️  Model Not Found[/yellow]\n"
                        "[dim]The model could not be located on this system.[/dim]\n"
                        "[dim]\n"
                        "[bold]Solutions:[/bold]\n"
                        "[dim]  1. Run /setup to download the model[/dim]\n"
                        "[dim]  2. Check: parishad models list[/dim]\n"
                        "[dim]  3. Download manually: parishad download <model>[/dim]\n"
                    )
                elif "no gpu" in error_msg.lower() or "cuda not available" in error_msg.lower():
                    self.log_message(
                        "[yellow]⚠️  No GPU Detected[/yellow]\n"
                        "[dim]This project requires a CUDA-capable GPU.[/dim]\n"
                        "[dim]\n"
                        "[bold]Requirements:[/bold]\n"
                        "[dim]  • NVIDIA GPU with CUDA support[/dim]\n"
                        "[dim]  • Updated GPU drivers[/dim]\n"
                        "[dim]  • PyTorch with CUDA installed[/dim]\n"
                        "[dim]\n"
                        "[dim]Check: nvidia-smi[/dim]\n"
                    )
                else:
                    self.log_message(f"[red]{type(e).__name__}: {error_msg}[/red]\n")
                
                self.council = None
                return
            
            if self.council:
                self.log_message(
                    f"[green]✅ Sabha council ready![/green]\n"
                    f"[dim]Models loaded from profile '{profile}'[/dim]\n"
                    f"[dim]You can now start asking questions.[/dim]\n"
                )
            else:
                self.log_message("[red]✗ Council initialization returned None[/red]\n")
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log_message(
                f"[red]✗ Error loading Sabha council:[/red]\n"
                f"[red]{type(e).__name__}: {e}[/red]\n"
                f"[dim]{tb}[/dim]\n"
            )
            self.council = None
        finally:
            self._initializing = False
    
    def _initialize_council_thread_worker(self) -> None:
        """
        Initialize Sabha council in a dedicated thread worker.
        
        CRITICAL FOR WINDOWS: This method runs in a real OS thread (not an asyncio executor)
        which prevents the TUI from freezing during blocking model loading.
        
        UI updates are sent via non-blocking post_message to prevent deadlock.
        """
        if self._initializing:
            self.post_message(self.LogMessage("[yellow]Already initializing...[/yellow]\n"))
            return
        
        self._initializing = True
        
        try:
            from parishad.orchestrator.engine import Parishad
            from parishad.config.user_config import load_user_config
            
            self.post_message(self.LogMessage("[cyan]🔄 Initializing Sabha council...[/cyan]\n"))
            
            # Load user config for profile (same as CLI run does)
            user_cfg = load_user_config()
            profile = user_cfg.default_profile
            mode = user_cfg.default_mode
            
            self.post_message(self.LogMessage(f"[dim]  • Profile: {profile}[/dim]\n"))
            self.post_message(self.LogMessage(f"[dim]  • Mode: {mode}[/dim]\n"))
            
            # Get pipeline config from Sabha selection
            if self.config:
                config_name = self.config.get_pipeline_config()
                self.post_message(self.LogMessage(f"[dim]  • Pipeline: {config_name}[/dim]\n"))
            else:
                config_name = "core"  # Default fallback
                self.post_message(self.LogMessage(f"[dim]  • Pipeline: {config_name} (default)[/dim]\n"))
            
            self.post_message(self.LogMessage(f"[yellow]⏳ Loading models (this may take 30-60 seconds)...[/yellow]\n"))
            self.post_message(self.LogMessage(f"[dim]  • Creating Parishad engine...[/dim]\n"))
            
            # Build user_forced_config from model_map
            user_forced_config = {}
            if self.config and self.config.model_map:
                # Initialize manager to resolve paths
                from parishad.models.downloader import ModelManager
                model_manager = ModelManager()
                
                msg_backend = self.config.backend or "ollama"
                
                for slot, model_id in self.config.model_map.items():
                    # Default to current config backend
                    current_backend = msg_backend
                    model_file = None
                    
                    # Check if it's a known model to resolve backend/path
                    model_info = model_manager.registry.get(model_id)
                    if model_info:
                        # Handle Enum comparison correctly
                        source = model_info.source.value if hasattr(model_info.source, "value") else str(model_info.source)
                        
                        if source == "local":
                            current_backend = "llama_cpp"
                            model_file = str(model_info.path)
                        elif source == "ollama":
                            current_backend = "ollama"
                        elif source == "mlx":
                            current_backend = "mlx"
                    else:
                        # Fallback heuristics if not in registry
                        if model_id.startswith("local:"):
                            current_backend = "llama_cpp"
                        elif model_id.startswith("ollama:") or ":" in model_id:
                            current_backend = "ollama"

                    user_forced_config[slot] = {
                        "model_id": model_id,
                        "backend_type": current_backend
                    }
                    if model_file:
                        user_forced_config[slot]["model_file"] = model_file

            # Create the Parishad engine (blocking call in this thread)
            self.council = Parishad(
                config=config_name,
                model_config_path=None,  # Let engine use profiles + models.yaml
                profile=profile,
                pipeline_config_path=None,
                trace_dir=None,
                mock=False,
                stub=False,
                mode=mode,
                user_forced_config=user_forced_config or None,
                no_retry=False,
            )
            
            if self.council:
                self.post_message(self.CouncilReady(success=True, profile=profile))
            else:
                self.post_message(self.CouncilReady(success=False, error_msg="Council initialization returned None"))
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.post_message(self.CouncilReady(
                success=False, 
                error_msg=f"{type(e).__name__}: {e}\n{tb}"
            ))
            self.council = None
        finally:
            self._initializing = False
    
    # DEPRECATED: TUI now uses same engine setup as CLI 'parishad run'
    # This method is no longer called
    # def _create_model_config_from_tui(self):
    #     """Create ModelConfig from TUI ParishadConfig."""
    #     ...
    
    def _check_backend_availability(self) -> None:
        """Check if the configured backend is available and show help if not."""
        from parishad.models.backends import is_backend_available, get_available_backends
        
        if not self.config:
            return
        
        # Map TUI backend to runner backend
        backend_map = {
            "ollama": "ollama",
            "huggingface": "transformers",
            "lmstudio": "openai",
            "openai": "openai",
            "local": "llama_cpp",
            "llama_cpp": "llama_cpp",
            "transformers": "transformers",
            "mlx": "mlx",
        }
        
        backend_name = backend_map.get(self.config.backend.lower(), self.config.backend)
        
        if not is_backend_available(backend_name):
            available = get_available_backends()
            self.log_message(
                f"\n[yellow]⚠ Backend '{self.config.backend}' is not available![/yellow]\n\n"
                f"[bold]Selected backend:[/bold] {self.config.backend}\n"
                f"[bold]Model:[/bold] {self.config.model}\n\n"
                f"[bold]Issue:[/bold] Required dependencies not installed.\n\n"
                f"[bold]Available backends:[/bold] {', '.join(available)}\n\n"
                f"[bold]To fix:[/bold]\n"
            )
            
            # Show installation instructions based on backend
            if backend_name == "transformers":
                self.log_message(
                    "  [cyan]pip install transformers torch[/cyan]\n"
                    "  (For GPU: pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118)\n"
                )
            elif backend_name == "ollama":
                self.log_message(
                    "  1. Install Ollama: [cyan]https://ollama.ai[/cyan]\n"
                    "  2. Pull model: [cyan]ollama pull " + self.config.model + "[/cyan]\n"
                )
            elif backend_name == "llama_cpp":
                self.log_message(
                    "  [cyan]pip install llama-cpp-python[/cyan]\n"
                    "  (For GPU: CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install llama-cpp-python)\n"
                )
            elif backend_name == "openai":
                self.log_message(
                    "  [cyan]pip install openai[/cyan]\n"
                    "  Set API key: [cyan]export OPENAI_API_KEY=your_key[/cyan]\n"
                )
            elif backend_name == "mlx":
                self.log_message(
                    "  [cyan]pip install mlx-lm[/cyan]\n"
                    "  (Only works on Apple Silicon M1/M2/M3/M4)\n"
                )
            
            self.log_message(
                "\n[dim]Or run [cyan]/setup[/cyan] to choose a different backend.[/dim]\n"
            )
            return
        
        # Backend is available - show success message
        self.log_message(
            f"[dim]Sabha council initialized with {self.config.backend} backend[/dim]"
        )
    
    def _check_model_availability(self) -> None:
        """DEPRECATED: Use _check_backend_availability instead."""
        import subprocess
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            installed_models = result.stdout.lower()
            
            # Check if our model is in the list
            model_name = self.model.split(":")[0].lower()  # e.g., "llama3.2" from "llama3.2:3b"
            if model_name not in installed_models:
                self.log_message(
                    f"\n[yellow]⚠ Model '{self.model}' not installed![/yellow]\n"
                    f"[dim]Run this command in another terminal to install it:[/dim]\n"
                    f"[bold cyan]  ollama pull {self.model}[/bold cyan]\n"
                    f"[dim]Then restart parishad.[/dim]\n"
                )
        except FileNotFoundError:
            self.log_message(
                "\n[yellow]⚠ Ollama not found![/yellow]\n"
                "[dim]Install Ollama from:[/dim] [bold cyan]https://ollama.ai[/bold cyan]\n"
            )
        except Exception:
            pass  # Silent fail for other errors

    
    def log_message(self, message: str) -> None:
        """Add message to chat log."""
        chat = self.query_one("#chat-area", RichLog)
        chat.write(message)
        chat.scroll_end()
    
    def _sanitize_answer_text(self, text: str) -> str:
        """
        Extract clean text from any format: plain text, JSON, malformed JSON, etc.
        
        Uses aggressive parsing to handle all model quirks.
        """
        import json
        import re
        
        if not text or not text.strip():
            return text
        
        original = text
        text = text.strip()
        
        # Strategy 1: If it's plain text (no JSON structure), return as-is
        if not any(marker in text.lower() for marker in ['{', '[', 'json', '"final_answer"', '"answer"']):
            return text
        
        # Strategy 2: Try to extract from JSON structure
        # Normalize malformed JSON first
        normalized = text
        
        # Remove 'json' prefix: json { -> {
        normalized = re.sub(r'^json\s*', '', normalized, flags=re.IGNORECASE)
        
        # Fix parentheses: (...) -> {...}
        if normalized.startswith('(') and ')' in normalized:
            normalized = normalized.replace('(', '{', 1)
            normalized = normalized[::-1].replace(')', '}', 1)[::-1]  # Replace last )
        
        # Fix key names with spaces: "final answer" -> "final_answer"
        normalized = re.sub(r'"([a-z]+)\s+([a-z]+)"(\s*):', r'"\1_\2"\3:', normalized, flags=re.IGNORECASE)
        
        # Fix period separators: ". " -> ", "
        normalized = re.sub(r'"\s*\.\s*"', '", "', normalized)
        
        # Try to parse normalized JSON
        try:
            data = json.loads(normalized)
            if isinstance(data, dict):
                # Priority extraction
                for key in ["final_answer", "answer", "response", "content", "text", "result"]:
                    if key in data and isinstance(data[key], str) and len(data[key]) > 10:
                        return data[key]
        except:
            pass
        
        # Strategy 3: Regex extraction from JSON-like text
        patterns = [
            r'"final_answer"\s*:\s*"([^"]+)"',
            r'"answer"\s*:\s*"([^"]+)"',
            r'"response"\s*:\s*"([^"]+)"',
            r'"content"\s*:\s*"([^"]+)"',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1)
        
        # Strategy 4: Find longest quoted string (likely the answer)
        quoted = re.findall(r'"([^"]{50,})"', text)
        if quoted:
            return max(quoted, key=len)
        
        # Strategy 5: Extract sentence-like text
        sentences = re.findall(r'[A-Z][^.!?]{20,}[.!?]', text)
        if sentences:
            return ' '.join(sentences[:3])
        
        # Last resort: return original
        return original
    
    @on(Input.Submitted)
    def handle_input(self, event: Input.Submitted) -> None:
        """Handle user input submission with parsing layer."""
        raw_input = event.value.strip()
        input_widget = event.input
        input_widget.value = ""
        
        if not raw_input:
            return
        
        # Parse input
        parsed = parse_input(raw_input)
        
        # Show user message (with original input for transparency)
        self.log_message(f"\n[bold]> {parsed.raw}[/bold]")
        
        # Handle commands
        if parsed.is_command:
            self.handle_command(parsed)
            return
        
        # Load referenced files
        loaded_files = []
        if parsed.tools:
            self.log_message("[dim]📎 Loading files...[/dim]")
            for tool in parsed.tools:
                if tool["type"] == "file":
                    loaded = load_file(tool["path"], Path(self.cwd))
                    loaded_files.append(loaded)
                    
                    if loaded.exists and loaded.content:
                        size_kb = loaded.size_bytes // 1024
                        self.log_message(f"  [green]✓[/green] {loaded.path} ({size_kb}KB)")
                        if loaded.error:  # Truncation
                            self.log_message(f"    [yellow]⚠[/yellow] {loaded.error}")
                    else:
                        self.log_message(f"  [red]✗[/red] {loaded.error}")
        
        # Show active flags
        if parsed.flags:
            flag_names = ", ".join(f"#{k}" for k in parsed.flags.keys())
            self.log_message(f"[dim]🏴 Active flags: {flag_names}[/dim]")
        
        # Build augmented prompt
        final_prompt = build_augmented_prompt(
            parsed.user_query,
            loaded_files,
            parsed.flags
        )
        
        if not final_prompt.strip():
            self.log_message("[yellow]⚠ Empty query after parsing[/yellow]")
            return
        
        # Process with Sabha council
        progress = self.query_one("#role-progress", RoleProgressBar)
        progress.reset()
        if sys.platform == "win32":
             self.log_message("[dim]  -- Sabha deliberating...[/dim]")
        else:
             self.log_message("[dim]  ⎿ सभा विचार-विमर्श...[/dim]")  # Sabha deliberating in Hindi
        
        if not self.council:
            self.log_message("\n[red]✗ Sabha council not loaded![/red]")
            self.log_message("[yellow]⚠ The model failed to initialize. Check the error messages above.[/yellow]")
            self.log_message("[dim]Hint: Try running /setup to reconfigure, or check models.yaml profile.[/dim]\n")
            return
        
        # Prevent concurrent query processing
        if self._processing_query:
            self.log_message("[yellow]⚠ Already processing a query, please wait...[/yellow]")
            return
        
        # CRITICAL FOR WINDOWS: The GIL blocks Textual's event loop during llama-cpp inference
        # even when using threads. We use subprocess.Popen to spawn a SEPARATE Python process.
        # This is the only way to keep the TUI responsive on Windows with llama-cpp-python.
        # See: docs/TUI_FREEZE_WINDOWS.md for full technical explanation.
        self._processing_query = True
        
        # Check VRAM compatibility before starting (like LM Studio)
        vram_warning_shown = False
        try:
            import json
            config_json_path = Path.home() / ".parishad" / "config.json"
            if config_json_path.exists():
                with open(config_json_path) as f:
                    user_config_data = json.load(f)
                
                session = user_config_data.get("session", {})
                model_key = session.get("model")
                
                if model_key:
                    # Simple file size based check as fallback
                    available_models = user_config_data.get("models", {})
                    if model_key in available_models:
                        size_bytes = available_models[model_key].get("size_bytes", 0)
                        size_gb = size_bytes / 1e9
                        
                        # Get VRAM - smart check with CPU/GPU offloading intelligence
                        try:
                            import torch
                            if torch.cuda.is_available():
                                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                                
                                # Estimate: file size * 1.8 for loaded model + KV cache
                                estimated_usage = size_gb * 1.8
                                usage_percent = (estimated_usage / vram_gb) * 100
                                
                                # Smart warnings based on CPU/GPU offloading behavior
                                if usage_percent <= 100:
                                    # Model fits entirely in GPU - optimal performance
                                    if size_gb >= 1.5:
                                        self.log_message(f"\n[green]✓ Model fits in GPU: {size_gb:.1f}GB model | {vram_gb:.1f}GB VRAM[/green]")
                                        self.log_message(f"[dim]Expected: Fast inference (GPU-only)[/dim]\n")
                                        vram_warning_shown = True
                                        
                                elif usage_percent <= 150:
                                    # Slight CPU offload - still good performance
                                    gpu_portion = int((vram_gb / estimated_usage) * 100)
                                    cpu_portion = 100 - gpu_portion
                                    self.log_message(f"\n[cyan]ℹ CPU+GPU Offload: {size_gb:.1f}GB model | {vram_gb:.1f}GB VRAM ({usage_percent:.0f}%)[/cyan]")
                                    self.log_message(f"[cyan]Split: ~{gpu_portion}% GPU, ~{cpu_portion}% CPU RAM[/cyan]")
                                    self.log_message(f"[dim]Expected: Good performance with minimal slowdown (2-5 min/query)[/dim]\n")
                                    vram_warning_shown = True
                                    
                                elif usage_percent <= 250:
                                    # Moderate CPU offload - noticeable slowdown
                                    gpu_portion = int((vram_gb / estimated_usage) * 100)
                                    cpu_portion = 100 - gpu_portion
                                    self.log_message(f"\n[yellow]⚠ Significant CPU Offload: {size_gb:.1f}GB model | {vram_gb:.1f}GB VRAM ({usage_percent:.0f}%)[/yellow]")
                                    self.log_message(f"[yellow]Split: ~{gpu_portion}% GPU, ~{cpu_portion}% CPU RAM[/yellow]")
                                    self.log_message(f"[yellow]Expected: Moderate slowdown (10-20 min/query)[/yellow]\n")
                                    vram_warning_shown = True
                                    
                                else:
                                    # Heavy CPU offload - significant slowdown
                                    gpu_portion = int((vram_gb / estimated_usage) * 100)
                                    cpu_portion = 100 - gpu_portion
                                    self.log_message(f"\n[red]⚠ Heavy CPU Offload: {size_gb:.1f}GB model | {vram_gb:.1f}GB VRAM ({usage_percent:.0f}%)[/red]")
                                    self.log_message(f"[red]Split: ~{gpu_portion}% GPU, ~{cpu_portion}% CPU RAM[/red]")
                                    self.log_message(f"[red]Expected: Very slow inference (30+ min/query)[/red]")
                                    self.log_message(f"[dim]Smaller models (0.5-3B) recommended for better experience[/dim]\n")
                                    vram_warning_shown = True
                        except Exception as vram_ex:
                            # Silently skip VRAM check if torch unavailable (CPU-only or torch not installed)
                            pass
        except Exception as e:
            # Log error but don't block execution
            self.log_message(f"[dim]⚠ VRAM check failed: {e}[/dim]")
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"VRAM check error: {e}", exc_info=True)
        
        # Save query to temp file for subprocess to read
        query_file = Path.home() / ".parishad" / "temp_query.txt"
        result_file = Path.home() / ".parishad" / "temp_result.json"
        status_file = Path.home() / ".parishad" / "temp_status.txt"
        
        # Clean up old files
        for f in [result_file, status_file]:
            if f.exists():
                f.unlink()
        
        query_file.write_text(final_prompt, encoding="utf-8")
        
        # Get the Python executable path
        python_exe = sys.executable
        
        # Build inline script that runs inference and saves result
        inline_script = f'''
import sys
import json
from pathlib import Path

query_file = Path(r"{query_file}")
result_file = Path(r"{result_file}")
status_file = Path(r"{status_file}")

try:
    status_file.write_text("starting", encoding="utf-8")
    
    query = query_file.read_text(encoding="utf-8")
    
    status_file.write_text("loading", encoding="utf-8")
    
    # Import and run inference
    from parishad.orchestrator.engine import Parishad
    from parishad.config.user_config import load_user_config
    import sys
    import os
    
    # Add parent dir to path to import load_parishad_config
    sys.path.insert(0, os.path.dirname(__file__))
    
    user_cfg = load_user_config()
    
    # Load config to get pipeline name from Sabha selection
    # Create debug log file for config loading
    debug_log = Path.home() / ".parishad" / "sabha_config_debug.log"
    
    config_name = "core"  # Default fallback
    try:
        # Import the function from the same file
        config_file = Path.home() / ".parishad" / "config.json"
        debug_log.write_text(f"Reading config from: {{config_file}}\\n", encoding="utf-8")
        
        if config_file.exists():
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                session = data.get("session", {{}})
                sabha = session.get("sabha", "madhyam")
                
                debug_msg = debug_log.read_text(encoding="utf-8")
                debug_msg += f"Found sabha: {{sabha}}\\n"
                debug_log.write_text(debug_msg, encoding="utf-8")
                
                # Map sabha to pipeline config
                from parishad.config.modes import get_pipeline_name
                config_name = get_pipeline_name(sabha)
                
                debug_msg = debug_log.read_text(encoding="utf-8")
                debug_msg += f"Mapped to config: {{config_name}}\\n"
                debug_log.write_text(debug_msg, encoding="utf-8")
        else:
            debug_msg = debug_log.read_text(encoding="utf-8")
            debug_msg += f"Config file not found, using default: {{config_name}}\\n"
            debug_log.write_text(debug_msg, encoding="utf-8")
    except Exception as e:
        import traceback
        debug_msg = debug_log.read_text(encoding="utf-8") if debug_log.exists() else ""
        debug_msg += f"ERROR loading config: {{str(e)}}\\n{{traceback.format_exc()}}\\n"
        debug_msg += f"Using default config: {{config_name}}\\n"
        debug_log.write_text(debug_msg, encoding="utf-8")
    
    # Log final config being used
    debug_msg = debug_log.read_text(encoding="utf-8")
    debug_msg += f"\\nFINAL CONFIG TO USE: {{config_name}}\\n"
    debug_log.write_text(debug_msg, encoding="utf-8")
    
    council = Parishad(
        config=config_name,
        profile=user_cfg.default_profile,
        mode=user_cfg.default_mode,
    )
    
    # Confirm Parishad initialized with correct config
    debug_msg = debug_log.read_text(encoding="utf-8")
    debug_msg += f"Parishad initialized with config: {{council.config_name}}\\n"
    debug_log.write_text(debug_msg, encoding="utf-8")
    
    status_file.write_text("running", encoding="utf-8")
    
    trace = council.run(query)
    
    status_file.write_text("complete", encoding="utf-8")
    
    # Save result as JSON
    result = {{
        "success": True,
        "roles": len(trace.roles),
        "tokens": trace.total_tokens,
        "final_answer": trace.final_answer.final_answer if trace.final_answer else None,
        "error": trace.error,
    }}
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    
except Exception as e:
    import traceback
    result = {{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    status_file.write_text("error", encoding="utf-8")
'''
        
        # Write script to temp file
        script_file = Path.home() / ".parishad" / "temp_inference_script.py"
        script_file.write_text(inline_script, encoding="utf-8")
        
        # Launch subprocess - runs in a COMPLETELY SEPARATE PROCESS (no GIL sharing!)
        # Use CREATE_NO_WINDOW on Windows to avoid console popup
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
        
        self._subprocess = subprocess.Popen(
            [python_exe, str(script_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            cwd=str(self.cwd),
        )
        
        # Poll for result file
        poll_count = 0
        max_polls = 2400  # 20 minutes max (2400 * 0.5s = 1200s) - needed for 3B models on 4GB VRAM
        
        def poll_subprocess_result():
            nonlocal poll_count
            poll_count += 1
            
            # Check if subprocess is still alive
            if self._subprocess.poll() is not None and not result_file.exists():
                # Subprocess died without creating result file
                exit_code = self._subprocess.poll()
                self.log_message(f"\n[red]✗ Subprocess crashed (exit code: {exit_code})[/red]")
                self._processing_query = False
                return
            
            # Check timeout
            if poll_count > max_polls:
                self.log_message("\n[red]✗ Query timed out after 20 minutes[/red]")
                try:
                    self._subprocess.terminate()
                except:
                    pass
                self._processing_query = False
                return
            
            # Check if result file exists (inference complete)
            if result_file.exists():
                try:
                    result_text = result_file.read_text(encoding="utf-8")
                    result = json.loads(result_text)
                    
                    # Write to output.json in current workspace for user reference
                    output_file = self.cwd / "output.json"
                    try:
                        if result.get("success") and result.get("final_answer"):
                            # Sanitize answer to extract clean text from JSON if needed
                            clean_answer = self._sanitize_answer_text(result['final_answer'])
                            output_file.write_text(clean_answer, encoding="utf-8")
                    except Exception as e:
                        # Also log if this fails
                        import traceback
                        self.log_message(f"[dim]⚠ Could not write output.json: {e}[/dim]")
                    
                    if result.get("success"):
                        # Display the result
                        self.log_message(f"\n[dim]━━━ Sabha Activity ({result.get('roles')} roles, {result.get('tokens')} tokens) ━━━[/dim]")
                        
                        if result.get("final_answer"):
                            answer = self._sanitize_answer_text(result['final_answer'])
                            self.log_message(f"\n[bold]👑 Raja's Answer:[/bold]\n{answer}\n")
                        elif result.get("error"):
                            self.log_message(f"\n[red]Error: {result['error']}[/red]")
                        else:
                            self.log_message("\n[green]Query completed successfully![/green]")
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        traceback_msg = result.get('traceback', '')[:500]
                        self.log_message(f"\n[red]Error: {error_msg}[/red]\n[dim]{traceback_msg}...[/dim]")
                    
                    # Cleanup temp files
                    for f in [result_file, status_file, script_file]:
                        try:
                            if f.exists():
                                f.unlink()
                        except Exception as e:
                            pass
                    
                    self._processing_query = False
                    try:
                        self.query_one("#prompt-input", Input).focus()
                    except:
                        pass
                    
                except json.JSONDecodeError as e:
                    self.log_message(f"\n[red]✗ Failed to parse result: {e}[/red]")
                    self._processing_query = False
                except Exception as e:
                    self.log_message(f"\n[red]✗ Error reading result: {e}[/red]")
                    self._processing_query = False
            else:
                # Keep polling until result is ready
                if self._processing_query:
                    self.set_timer(0.5, poll_subprocess_result)
        
        # Start polling for result
        self.set_timer(0.5, poll_subprocess_result)
    
    async def _async_run_sabha(self, query: str, progress: RoleProgressBar) -> None:
        """Execute Sabha council asynchronously to prevent UI freezing."""
        if self._processing_query:
            return
        
        self._processing_query = True
        
        import asyncio
        
        try:
            # Run the Sabha council pipeline with augmented prompt in thread pool
            # Add timeout to prevent indefinite freeze if model doesn't respond
            loop = asyncio.get_event_loop()
            try:
                trace = await asyncio.wait_for(
                    loop.run_in_executor(None, self.council.run, query),
                    timeout=300.0  # 5 minute timeout for query execution
                )
            except asyncio.TimeoutError:
                self.log_message(
                    "\n[red]✗ Query execution timed out (5 minutes)[/red]\n"
                    "[yellow]⚠ The model may be stuck or the backend is unresponsive.[/yellow]\n"
                    "[dim]Hints:[/dim]\n"
                    "[dim]  • Check if your model backend is still running[/dim]\n"
                    "[dim]  • Try a simpler query[/dim]\n"
                    "[dim]  • Restart with: /setup[/dim]\n"
                )
                return
            
            # Update progress bar based on trace
            for role_output in trace.roles:
                role_name = role_output.role.lower()
                progress.mark_complete(role_name)
            
            # Display role activity summary (collapsible style)
            self.log_message(f"\n[dim]━━━ Sabha Activity ({len(trace.roles)} roles, {trace.total_tokens} tokens) ━━━[/dim]")
            
            for role_output in trace.roles:
                role_name = role_output.role.lower()
                info = ROLE_INFO.get(role_name, {"emoji": "❓", "name": role_name.title()})
                status_icon = "[green]✓[/green]" if role_output.status == "success" else "[red]✗[/red]"
                
                # Brief summary of what the role did
                summary = ""
                if role_name == "darbari" and role_output.core_output:
                    task_type = role_output.core_output.get("task_type", "unknown")
                    summary = f"→ Task: {task_type}"
                elif role_name == "majumdar" and role_output.core_output:
                    steps = role_output.core_output.get("steps", [])
                    summary = f"→ {len(steps)} step plan"
                elif role_name == "prerak" and role_output.core_output:
                    flags = role_output.core_output.get("flags", [])
                    if not flags:
                        summary = "→ No issues"
                    else:
                        summary = f"→ {len(flags)} issue(s)"
                elif role_name == "raja" and role_output.core_output:
                    conf = role_output.core_output.get("confidence", 0)
                    summary = f"→ Confidence: {int(conf*100)}%"
                
                # Show model used
                model_str = ""
                if role_output.metadata and role_output.metadata.model_id:
                     mid = role_output.metadata.model_id
                     # Strip path
                     if "/" in mid:
                         mid = mid.split("/")[-1]
                     # Strip extension (optional but cleaner)
                     if mid.endswith(".gguf"):
                         mid = mid[:-5]
                     model_str = f"[dim]({mid})[/dim]"

                if role_output.status == "error":
                     err_msg = role_output.error or "Unknown error"
                     # Show full error
                     summary = f"[red]{err_msg}[/red]"

                self.log_message(f"  {info['emoji']} {info['name']} {model_str}: {status_icon} {summary}")
            
            self.log_message(f"[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]\n")
            
            # Check for silent file generation (common source of confusion)
            for role_output in trace.roles:
                if role_output.core_output and role_output.core_output.get("target_file"):
                    fpath = role_output.core_output.get("target_file")
                    self.log_message(f"\n[bold blue]📁 File Generated:[/bold blue] {fpath}")

            # Display the final answer from Raja
            if trace.final_answer:
                answer = trace.final_answer.final_answer
                # Sanitize answer to remove any JSON formatting that may have leaked through
                answer = self._sanitize_answer_text(answer)
                self.log_message(f"\n[bold]👑 Raja's Answer:[/bold]\n{answer}\n")
            elif trace.error:
                self.log_message(f"\n[red]Error: {trace.error}[/red]")
            else:
                 # Check if we generated a file but no text answer
                file_gen = any(r.core_output and r.core_output.get("target_file") for r in trace.roles)
                if not file_gen:
                    self.log_message("\n[yellow]No answer generated[/yellow]")
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log_message(f"\n[red]Error ({type(e).__name__}): {e}[/red]\n[dim]{tb[:500]}...[/dim]")
        finally:
            # Always reset processing flag and re-enable input
            self._processing_query = False
            try:
                self.query_one("#user-input").focus()
            except:
                pass
    
    def _native_sabha_worker(self, query: str) -> None:
        """
        Native Python thread worker for Sabha execution.
        
        CRITICAL FOR WINDOWS: This uses a regular Python thread and a thread-safe
        queue instead of Textual's worker system which causes freezes on Windows.
        """
        debug_log(">>> WORKER THREAD STARTED <<<")
        debug_log(f"Worker thread ID: {threading.current_thread().ident}")
        debug_log(f"Query to process: {query[:100]}...")
        
        try:
            # Run the blocking inference in this native thread
            debug_log("Calling self.council.run()... (this will block)")
            debug_log("=== INFERENCE START ===")
            
            trace = self.council.run(query)
            
            debug_log("=== INFERENCE COMPLETE ===")
            debug_log(f"Trace received: {trace is not None}")
            if trace:
                debug_log(f"Trace roles: {len(trace.roles)}, tokens: {trace.total_tokens}")
            
            # Put result in queue (thread-safe, non-blocking)
            debug_log("Putting result in queue...")
            self._result_queue.put(("success", trace))
            debug_log("Result queued successfully!")
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            debug_log(f"!!! WORKER ERROR: {type(e).__name__}: {e}")
            debug_log(f"Traceback: {tb[:500]}")
            # Put error in queue
            self._result_queue.put(("error", (e, tb)))
            debug_log("Error queued.")
        
        debug_log(">>> WORKER THREAD EXITING <<<")
    
    def _poll_result_queue(self) -> None:
        """
        Timer callback to poll the result queue for Sabha results.
        
        This is called by a Textual timer and runs on the main event loop thread,
        so it's safe to update the UI directly.
        """
        debug_log("POLL: Timer fired, checking queue...")
        
        try:
            # Non-blocking check for results
            result_type, result_data = self._result_queue.get_nowait()
            
            debug_log(f"POLL: Got result from queue! Type: {result_type}")
            
            # Process the result on the main thread (safe for UI updates)
            if result_type == "success":
                debug_log("POLL: Processing success result, calling _display_sabha_result_direct...")
                self._display_sabha_result_direct(result_data)
                debug_log("POLL: Display complete!")
            else:
                debug_log("POLL: Processing error result...")
                error, tb = result_data
                self.log_message(f"\n[red]Error ({type(error).__name__}): {error}[/red]\n[dim]{tb[:500]}...[/dim]")
            
            # Clean up
            debug_log("POLL: Cleaning up, setting _processing_query = False")
            self._processing_query = False
            try:
                self.query_one("#prompt-input", Input).focus()
                debug_log("POLL: Input refocused!")
            except Exception:
                debug_log("POLL: Could not refocus input")
            
            debug_log("=== QUERY EXECUTION COMPLETE ===")
            
        except queue.Empty:
            # No result yet, keep polling
            if self._processing_query:
                # Don't log every tick to avoid log spam, just every 10th
                self.set_timer(0.1, self._poll_result_queue)
    
    def _display_sabha_result_direct(self, trace) -> None:
        """Display Sabha result directly (called from main thread via poll timer)."""
        # Update progress bar based on trace
        try:
            progress = self.query_one("#role-progress", RoleProgressBar)
            for role_output in trace.roles:
                role_name = role_output.role.lower()
                progress.mark_complete(role_name)
        except Exception:
            pass  # Progress bar update is non-critical
        
        # Display role activity summary (collapsible style)
        self.log_message(f"\n[dim]━━━ Sabha Activity ({len(trace.roles)} roles, {trace.total_tokens} tokens) ━━━[/dim]")
        
        for role_output in trace.roles:
            role_name = role_output.role.lower()
            info = ROLE_INFO.get(role_name, {"emoji": "❓", "name": role_name.title()})
            status_icon = "[green]✓[/green]" if role_output.status == "success" else "[red]✗[/red]"
            
            # Brief summary of what the role did
            summary = ""
            if role_name == "darbari" and role_output.core_output:
                task_type = role_output.core_output.get("task_type", "unknown")
                summary = f"→ Task: {task_type}"
            elif role_name == "majumdar" and role_output.core_output:
                steps = role_output.core_output.get("steps", [])
                summary = f"→ {len(steps)} step plan"
            elif role_name == "prerak" and role_output.core_output:
                flags = role_output.core_output.get("flags", [])
                if not flags:
                    summary = "→ No issues"
                else:
                    summary = f"→ {len(flags)} issue(s)"
            elif role_name == "raja" and role_output.core_output:
                conf = role_output.core_output.get("confidence", 0)
                summary = f"→ Confidence: {int(conf*100)}%"
            
            # Show model used
            model_str = ""
            if role_output.metadata and role_output.metadata.model_id:
                mid = role_output.metadata.model_id
                if "/" in mid:
                    mid = mid.split("/")[-1]
                if mid.endswith(".gguf"):
                    mid = mid[:-5]
                model_str = f"[dim]({mid})[/dim]"

            if role_output.status == "error":
                err_msg = role_output.error or "Unknown error"
                summary = f"[red]{err_msg}[/red]"

            self.log_message(f"  {info['emoji']} {info['name']} {model_str}: {status_icon} {summary}")
        
        self.log_message(f"[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]\n")
        
        # Check for file generation
        for role_output in trace.roles:
            if role_output.core_output and role_output.core_output.get("target_file"):
                fpath = role_output.core_output.get("target_file")
                self.log_message(f"\n[bold blue]📁 File Generated:[/bold blue] {fpath}")

        # Display the final answer from Raja
        if trace.final_answer:
            answer = trace.final_answer.final_answer
            # Sanitize answer to remove any JSON formatting that may have leaked through
            answer = self._sanitize_answer_text(answer)
            self.log_message(f"\n[bold]👑 Raja's Answer:[/bold]\n{answer}\n")
        elif trace.error:
            self.log_message(f"\n[red]Error: {trace.error}[/red]")
        else:
            file_gen = any(r.core_output and r.core_output.get("target_file") for r in trace.roles)
            if not file_gen:
                self.log_message("\n[yellow]No answer generated[/yellow]")
    
    def _run_sabha_thread_worker(self, query: str) -> None:
        """
        Execute Sabha council in a dedicated thread worker.
        
        CRITICAL FOR WINDOWS: This method runs in a real OS thread (not an asyncio executor)
        which prevents the TUI from freezing during blocking llama-cpp inference.
        
        UI updates are sent via non-blocking post_message to prevent deadlock.
        """
        if self._processing_query:
            return
        
        self._processing_query = True
        
        try:
            # Run the blocking inference in this thread
            # This won't freeze the UI because it's a real thread worker
            trace = self.council.run(query)
            
            # Send non-blocking message with result (won't deadlock!)
            self.post_message(self.SabhaResultReady(trace))
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            # Send non-blocking error message
            self.post_message(self.SabhaError(e, tb))
        finally:
            self._processing_query = False
            # Send non-blocking completion message
            self.post_message(self.WorkerComplete("sabha"))
    
    def on_parishad_app_log_message(self, message: LogMessage) -> None:
        """Handle non-blocking log messages from worker threads."""
        self.log_message(message.text)
    
    def on_parishad_app_sabha_result_ready(self, message: SabhaResultReady) -> None:
        """Handle Sabha result from worker thread (non-blocking)."""
        trace = message.trace
        
        # Update progress bar based on trace
        try:
            progress = self.query_one("#role-progress", RoleProgressBar)
            for role_output in trace.roles:
                role_name = role_output.role.lower()
                progress.mark_complete(role_name)
        except Exception:
            pass  # Progress bar update is non-critical
        
        # Display role activity summary (collapsible style)
        self.log_message(f"\n[dim]━━━ Sabha Activity ({len(trace.roles)} roles, {trace.total_tokens} tokens) ━━━[/dim]")
        
        for role_output in trace.roles:
            role_name = role_output.role.lower()
            info = ROLE_INFO.get(role_name, {"emoji": "❓", "name": role_name.title()})
            status_icon = "[green]✓[/green]" if role_output.status == "success" else "[red]✗[/red]"
            
            # Brief summary of what the role did
            summary = ""
            if role_name == "darbari" and role_output.core_output:
                task_type = role_output.core_output.get("task_type", "unknown")
                summary = f"→ Task: {task_type}"
            elif role_name == "majumdar" and role_output.core_output:
                steps = role_output.core_output.get("steps", [])
                summary = f"→ {len(steps)} step plan"
            elif role_name == "prerak" and role_output.core_output:
                flags = role_output.core_output.get("flags", [])
                if not flags:
                    summary = "→ No issues"
                else:
                    summary = f"→ {len(flags)} issue(s)"
            elif role_name == "raja" and role_output.core_output:
                conf = role_output.core_output.get("confidence", 0)
                summary = f"→ Confidence: {int(conf*100)}%"
            
            # Show model used
            model_str = ""
            if role_output.metadata and role_output.metadata.model_id:
                mid = role_output.metadata.model_id
                # Strip path
                if "/" in mid:
                    mid = mid.split("/")[-1]
                # Strip extension (optional but cleaner)
                if mid.endswith(".gguf"):
                    mid = mid[:-5]
                model_str = f"[dim]({mid})[/dim]"

            if role_output.status == "error":
                err_msg = role_output.error or "Unknown error"
                # Show full error
                summary = f"[red]{err_msg}[/red]"

            self.log_message(f"  {info['emoji']} {info['name']} {model_str}: {status_icon} {summary}")
        
        self.log_message(f"[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]\n")
        
        # Check for silent file generation (common source of confusion)
        for role_output in trace.roles:
            if role_output.core_output and role_output.core_output.get("target_file"):
                fpath = role_output.core_output.get("target_file")
                self.log_message(f"\n[bold blue]📁 File Generated:[/bold blue] {fpath}")

        # Display the final answer from Raja
        if trace.final_answer:
            answer = trace.final_answer.final_answer
            # Sanitize answer to remove any JSON formatting that may have leaked through
            answer = self._sanitize_answer_text(answer)
            self.log_message(f"\n[bold]👑 Raja's Answer:[/bold]\n{answer}\n")
        elif trace.error:
            self.log_message(f"\n[red]Error: {trace.error}[/red]")
        else:
            # Check if we generated a file but no text answer
            file_gen = any(r.core_output and r.core_output.get("target_file") for r in trace.roles)
            if not file_gen:
                self.log_message("\n[yellow]No answer generated[/yellow]")
    
    def on_parishad_app_sabha_error(self, message: SabhaError) -> None:
        """Handle Sabha error from worker thread (non-blocking)."""
        self.log_message(f"\n[red]Error ({type(message.error).__name__}): {message.error}[/red]\n[dim]{message.traceback_str[:500]}...[/dim]")
    
    def on_parishad_app_worker_complete(self, message: WorkerComplete) -> None:
        """Handle worker completion - refocus input."""
        try:
            self.query_one("#prompt-input", Input).focus()
        except Exception:
            pass
    
    def on_parishad_app_council_ready(self, message: CouncilReady) -> None:
        """Handle council initialization completion (non-blocking)."""
        if message.success:
            self.log_message(
                f"[green]✅ Sabha council ready![/green]\n"
                f"[dim]Models loaded from profile '{message.profile}'[/dim]\n"
                f"[dim]You can now start asking questions.[/dim]\n"
            )
        else:
            self.log_message(
                f"[red]✗ Error loading Sabha council:[/red]\n"
                f"[dim]{message.error_msg}[/dim]\n"
            )
    
    # Keep these for backward compatibility but they are no longer used for thread workers
    def _display_sabha_result(self, trace) -> None:
        """Display Sabha result on main thread (DEPRECATED - use message handlers now)."""
        # Delegate to message handler
        self.on_parishad_app_sabha_result_ready(self.SabhaResultReady(trace))
    
    def _display_sabha_error(self, error: Exception, tb: str) -> None:
        """Display Sabha error on main thread (DEPRECATED - use message handlers now)."""
        self.on_parishad_app_sabha_error(self.SabhaError(error, tb))
    
    def _refocus_input(self) -> None:
        """Refocus input widget after query completion (DEPRECATED - use message handlers now)."""
        try:
            self.query_one("#prompt-input", Input).focus()
        except Exception:
            pass
    
    def handle_command(self, parsed: ParsedInput) -> None:
        """Handle slash commands with ParsedInput."""
        cmd = parsed.command_name
        args = parsed.command_args
        
        if cmd in ("exit", "quit", "q"):
            self._cmd_exit()
        elif cmd == "help" or cmd == "?":
            self._cmd_help()
        elif cmd == "clear":
            self._cmd_clear()
        elif cmd == "config":
            self._cmd_config()
        elif cmd in ("setup", "settings"):
            self._cmd_setup()
        elif cmd == "model":
            self._cmd_model(args)
        elif cmd == "sabha":
            self._cmd_sabha(args)
        elif cmd == "assign":
             self._cmd_assign(args)
        elif cmd == "redownload":
            self._cmd_redownload()
        elif cmd == "scan":
            self._cmd_scan()
        elif cmd == "save":
            self._cmd_save()
        else:
            self.log_message(
                f"[yellow]Unknown command: /{cmd}[/yellow]\n"
                f"[dim]Type /help for available commands[/dim]"
            )
    
    def _cmd_exit(self) -> None:
        """Exit the TUI."""
        self.exit()
    
    def _cmd_help(self) -> None:
        """Show TUI help."""
        self.log_message(
            "\n[bold cyan]═══ Parishad TUI Help ═══[/bold cyan]\n\n"
            "[bold]Commands:[/bold]\n"
            "  [cyan]/help[/cyan]      Show this help\n"
            "  [cyan]/exit[/cyan]      Exit the Parishad TUI\n"
            "  [cyan]/clear[/cyan]     Clear the chat area\n"
            "  [cyan]/config[/cyan]    Show current Sabha/mode/backend\n"
            "  [cyan]/setup[/cyan]     Re-configure Sabha/backend/model\n"
            "  [cyan]/model[/cyan]     Show or change the current model\n"
            "  [cyan]/model[/cyan]     Show or change the current model\n"
            "  [cyan]/redownload[/cyan] Re-download current model (if deleted/corrupted)\n"
            "  [cyan]/scan[/cyan]       Scan disk for new GGUF models\n\n"
            "[bold]Tools:[/bold]\n"
            "  [cyan]@file.py[/cyan]   Reference a file (contents will be included)\n"
            "  [cyan]@\"path with spaces.txt\"[/cyan]  Reference files with spaces (use quotes)\n\n"
            "[bold]Flags:[/bold]\n"
            "  [cyan]#idk[/cyan]       I don't know - prefer abstaining to guessing\n"
            "  [cyan]#safe[/cyan]      Safe mode - conservative, no speculation\n"
            "  [cyan]#noguess[/cyan]   Similar to #safe - avoid assumptions\n\n"
            "[bold]Examples:[/bold]\n"
            "  [dim]> @main.py explain this code[/dim]\n"
            "  [dim]> what is quantum entanglement #idk[/dim]\n"
            "  [dim]> @config.yaml @README.md summarize these files[/dim]\n"
        )
    
    def _cmd_clear(self) -> None:
        """Clear chat area."""
        self.query_one("#chat-area", RichLog).clear()
        self.log_message("[dim]Chat cleared[/dim]")
    
    def _cmd_config(self) -> None:
        """Show current configuration."""
        if not self.config:
            self.log_message("[yellow]No configuration loaded. Please run /setup.[/yellow]")
            return
        
        # Get dynamic config info
        from parishad.config.modes import get_mode_config
        try:
            mode_data = get_mode_config(self.config.sabha)
            sabha_display = f"{mode_data.sabha_name} ({mode_data.description})"
        except ValueError:
            sabha_display = f"{self.config.sabha} (Unknown Configuration)"

        mode = self.config.get_mode()
        pipeline = self.config.get_pipeline_config()
        
        # Get system profile and model directory
        from parishad.models.downloader import ModelManager, get_default_model_dir
        from parishad.config.user_config import load_user_config
        
        try:
            user_cfg = load_user_config()
            profile = user_cfg.default_profile
        except Exception:
            profile = "minimal_council"
        
        model_dir = get_default_model_dir()
        
        # Highlight current model
        current_model_display = f"[green]{self.model}[/green]"
        
        config_text = (
            f"\n[bold cyan]═══ Current Configuration ═══[/bold cyan]\n\n"
            f"[bold]Sabha Council:[/bold]\n"
            f"  Sabha:        {sabha_display}\n"
            f"  Mode:         {mode}\n"
            f"  Pipeline:     {pipeline}\n\n"
            f"[bold]Model Backend:[/bold]\n"
            f"  Profile:      [cyan]{profile}[/cyan]\n"
            f"  Current Model: {current_model_display}\n"
            f"  Model Dir:    {model_dir}\n\n"
        )
        
        # Show downloaded models (deduplicated)
        try:
            manager = ModelManager()
            all_models = manager.list_models()
            
            # Deduplicate by name
            unique_models = {}
            for m in all_models:
                if m.name not in unique_models:
                    unique_models[m.name] = m
            
            models = list(unique_models.values())
            
            if models:
                config_text += "[bold]Downloaded Models:[/bold]\n"
                for model in models[:10]:  # Show first 10
                    marker = "★ " if model.name == self.model else "• "
                    style = "[bold green]" if model.name == self.model else ""
                    end_style = "[/bold green]" if model.name == self.model else ""
                    config_text += f"  {marker}{style}{model.name:30} [{model.format.value:12}] {model.size_human}{end_style}\n"
                if len(models) > 10:
                    config_text += f"  [dim]... and {len(models) - 10} more[/dim]\n"
            else:
                config_text += "[yellow]No models downloaded yet.[/yellow]\n"
                config_text += "[dim]Use /setup to download models.[/dim]\n"
        except Exception as e:
            config_text += f"[yellow]Could not list models: {e}[/yellow]\n"
        
        config_text += (
            f"\n[bold]Working Directory:[/bold]\n"
            f"  {self.cwd}\n\n"
            f"[dim]Type /help for available commands.[/dim]\n"
        )
        
        self.log_message(config_text)
    
    def _cmd_setup(self) -> None:
        """Re-run setup to change configuration."""
        self.log_message("[dim]Opening setup...[/dim]")
        self.push_screen(SetupScreen(initial_config=self.config), callback=self._on_reconfig_complete)
    
    def _on_reconfig_complete(self, config: Optional[ParishadConfig]) -> None:
        """Callback when re-configuration is completed or aborted."""
        if config and config != self.config:
            # Config changed - reload
            self.config = config
            self.model = config.model
            self.backend = config.backend
            self.sabha = config.sabha
            self.cwd = Path(config.cwd) if config.cwd else Path.cwd()
            
            self.log_message("[green]✓ Configuration updated. Reloading Sabha...[/green]")
            self._initialize_chat()
        else:
            # Aborted or no change
            self.log_message("[dim]Setup cancelled - keeping current configuration[/dim]")
    
    def _cmd_model(self, args: List[str]) -> None:
        """Show or change model."""
        if args:
            new_model = args[0]
            # Check if model changed
            if self.model != new_model:
                self.model = new_model
                # Update config if exists
                if self.config:
                    self.config.model = new_model
                    try:
                        save_parishad_config(self.config)
                        self.log_message(f"[green]✓ Model changed to: {self.model}[/green]")
                        # Trigger re-initialization if needed
                        # self._initialize_chat() 
                    except Exception as e:
                        self.log_message(f"[red]Failed to save config: {e}[/red]")
                else:
                    self.log_message(f"[dim]Model changed to: {self.model} (runtime only)[/dim]")
            else:
                self.log_message(f"[dim]Model is already: {self.model}[/dim]")
        else:
            self.log_message(f"[dim]Current model: {self.model}[/dim]")
    
    def _cmd_scan(self) -> None:
        """Scan for models on disk."""
        from parishad.models.downloader import ModelManager
        try:
            manager = ModelManager()
            self.log_message("[dim]Scanning for models...[/dim]")
            
            # This updates the registry (models.json)
            new_models = manager.scan_for_models()
            
            # Also read valid models
            all_models = manager.list_models()
            
            msg = f"[green]✓ Scan complete.[/green]\n\n"
            if new_models:
                 msg += f"[bold]Found {len(new_models)} new models:[/bold]\n"
                 for m in new_models:
                     msg += f"  • {m.name}\n"
            else:
                 msg += "[dim]No new models found.[/dim]\n"
            
            msg += f"\n[dim]Total models available: {len(all_models)}[/dim]"
            self.log_message(msg)

        except Exception as e:
            self.log_message(f"[red]Error scanning models: {e}[/red]")

    def _cmd_save(self) -> None:
        """Manually save configuration."""
        if self.config:
            try:
                if save_parishad_config(self.config):
                    self.log_message("[green]✓ Configuration saved to disk.[/green]")
                else:
                    self.log_message("[red]Failed to save configuration.[/red]")
            except Exception as e:
                self.log_message(f"[red]Error saving configuration: {e}[/red]")
        else:
             self.log_message("[yellow]No configuration to save.[/yellow]")

    def _cmd_sabha(self, args: List[str]) -> None:
        """Switch active Sabha council."""
        valid_sabhas = ["laghu", "madhyam", "maha"]
        
        if not args:
            current = self.sabha or "unknown"
            self.log_message(f"[dim]Current Sabha: {current}[/dim]")
            self.log_message(f"[dim]Usage: /sabha [{'|'.join(valid_sabhas)}][/dim]")
            return
            
        new_sabha = args[0].lower()
        
        # Handle aliases
        aliases = {"fast": "laghu", "core": "madhyam", "balanced": "madhyam", "extended": "maha", "thorough": "maha"}
        new_sabha = aliases.get(new_sabha, new_sabha)
        
        if new_sabha not in valid_sabhas:
            self.log_message(f"[red]Invalid Sabha: {new_sabha}[/red]")
            self.log_message(f"[dim]Valid options: {', '.join(valid_sabhas)}[/dim]")
            return
            
        if new_sabha == self.sabha:
            self.log_message(f"[yellow]Already using {new_sabha}[/yellow]")
            return

        self.log_message(f"[cyan]Switching to {new_sabha}...[/cyan]")
        
        # Update config
        try:
            self.sabha = new_sabha
            if self.config:
                self.config.sabha = new_sabha
                save_parishad_config(self.config)
            
            # Re-initialize
            self._initialize_chat()
            
        except Exception as e:
            self.log_message(f"[red]Error switching Sabha: {e}[/red]")
    
    def _smart_assign_models(self, models: List[str]) -> None:
        """Smartly assign models to slots based on size."""
        from parishad.models.downloader import ModelManager
        try:
            manager = ModelManager()
            all_models = manager.list_models()
            
            # Find matching model objects
            selected = []
            for name in models:
                # Find best match
                match = None
                for m in all_models:
                    if m.name == name:
                        match = m
                        break
                # Fallback partial match if exact failing
                if not match:
                    for m in all_models:
                        if name in m.name:
                            match = m
                            break
                
                if match:
                    selected.append(match)
                else:
                    self.log_message(f"[yellow]Warning: Model '{name}' not found.[/yellow]")
            
            if not selected:
                self.log_message("[red]No valid models found for assignment.[/red]")
                return

            # Sort by size (descending)
            selected.sort(key=lambda x: x.size_bytes, reverse=True)
            
            updates = {}
            count = len(selected)
            
            if count == 1:
                # One model -> Assign to BIG (Primary)
                # Leaves other slots to default/previous
                updates = {"big": selected[0].name}
                
            elif count == 2:
                # Two models -> Big (Largest), Small (Smallest)
                updates = {
                    "big": selected[0].name,
                    "small": selected[1].name
                }
                
            else:
                # Three+ models -> Big, Mid, Small
                mid_idx = count // 2
                updates = {
                    "big": selected[0].name,
                    "mid": selected[mid_idx].name,
                    "small": selected[-1].name
                }
            
            if not self.config.model_map:
                self.config.model_map = {}
            
            self.config.model_map.update(updates)
            save_parishad_config(self.config)
            
            # Formatted log
            msg = "[green]Smartly assigned models:[/green]\n"
            for slot, model in updates.items():
                msg += f"  • {slot.upper():5}: {model}\n"
            self.log_message(msg)
            
            # Auto-reload
            self.log_message("[cyan]Reloading council with new assignments...[/cyan]")
            self._initialize_chat()

        except Exception as e:
            self.log_message(f"[red]Smart assignment failed: {e}[/red]")

    def _cmd_assign(self, args: List[str]) -> None:
        """Assign models to slots (Explicit or Smart)."""
        args_str = " ".join(args) if args else ""
        if not args_str:
            self.log_message("[yellow]Usage: /assign [big=model]... or /assign [model1] [model2]...[/yellow]")
            return
            
        # Check for explicit assignment
        if "=" in args_str:
            updates = {}
            parts = args_str.split()
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    if k in ["big", "mid", "small", "planner", "judge"]:
                         updates[k] = v
                    else:
                         self.log_message(f"[red]Unknown slot: {k}[/red]")
            
            if updates:
                if not self.config.model_map:
                    self.config.model_map = {}
                self.config.model_map.update(updates)
                save_parishad_config(self.config)
                self.log_message(f"[green]Assigned: {updates}[/green]")
                self.log_message("[cyan]Reloading council...[/cyan]")
                self._initialize_chat()
        else:
            # Smart Assignment
            self._smart_assign_models(args)

    def _cmd_redownload(self) -> None:
        """Force re-download of current model."""
        if not self.config:
            self.log_message("[yellow]No model configured. Use /setup first.[/yellow]")
            return
        
        from parishad.models.downloader import ModelManager
        
        source = self.config.backend
        model_id = self.config.model
        
        self.log_message(
            f"[yellow]Re-downloading {model_id} from {source}...[/yellow]\\n"
            f"[dim]This will delete the existing model file and download fresh.[/dim]\\n"
        )
        
        try:
            manager = ModelManager()
            
            # Remove existing model from registry
            if model_id in manager.registry._models:
                old_path = manager.registry._models[model_id].path
                if old_path.exists():
                    old_path.unlink()
                    self.log_message(f"[dim]Deleted old model file: {old_path}[/dim]\\n")
                del manager.registry._models[model_id]
                manager.registry._save_registry()
            
            # Trigger re-initialization which will download
            self.log_message("[cyan]Starting download...[/cyan]\\n")
            self._initialize_chat()
            
        except Exception as e:
            self.log_message(f"[red]Error during re-download: {e}[/red]\\n")
    
    def action_quit(self) -> None:
        """Handle Ctrl+C - cancel download if in progress, or require double press to exit."""
        # If download is in progress, cancel it
        if self.download_cancel_event and not self.download_cancel_event.is_set():
            self.download_cancel_event.set()
            self.log_message("\n[yellow]Download cancelled. Press Ctrl+C again to exit.[/yellow]\n")
            self.download_cancel_event = None
            return
        
        # Otherwise, require double press to exit
        if self.ctrl_c_pressed:
            # Kill subprocess if still running
            if self._subprocess and self._subprocess.poll() is None:
                try:
                    self._subprocess.terminate()
                    self._subprocess.wait(timeout=2)
                except:
                    try:
                        self._subprocess.kill()
                    except:
                        pass
            self.exit()
        else:
            self.ctrl_c_pressed = True
            # Show in status bar instead of chat
            self.query_one("#status", Static).update("[yellow]↳ Press Ctrl+C again to exit[/yellow]")
            # Reset after 2 seconds
            self.set_timer(2.0, self.reset_ctrl_c)
    
    def reset_ctrl_c(self) -> None:
        """Reset Ctrl+C state."""
        self.ctrl_c_pressed = False
        # Restore status bar
        self.query_one("#status", Static).update("[dim]? for help · Ctrl+C to exit[/dim]")
    
    def action_clear(self) -> None:
        """Clear chat area."""
        self.query_one("#chat-area", RichLog).clear()


def run_code_cli(
    backend: str = "ollama",
    model: str = "llama3.2:3b", 
    cwd: Optional[str] = None,
    sabha: Optional[str] = None,  # Sabha ID: "laghu"/"madhyam"/"maha" 
    mode: Optional[str] = None,   # Mode key: "fast"/"balanced"/"thorough"
):
    """
    Run Parishad Code TUI.
    
    Args:
        backend: Backend to use (ollama, lmstudio, etc.)
        model: Model name/ID
        cwd: Working directory
        sabha: Sabha ID to use (if specified, overrides config)
        mode: Mode key to use (converted to sabha internally)
    """
    import platform
    import logging
    
    # Configure logging to file for debugging
    log_file = os.path.expanduser("~/parishad_debug.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    logging.info("Parishad CLI starting...")
    
    # Windows-specific terminal fixes
    if platform.system() == "Windows":
        # Enable ANSI escape sequences on Windows
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass
        
        # Force UTF-8 encoding for Windows console
        if sys.stdout.encoding != 'utf-8':
            try:
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                if hasattr(sys.stderr, 'reconfigure'):
                    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
    
    # If mode is specified, convert to sabha
    if mode and not sabha:
        mode_to_sabha = {
            "fast": "laghu",
            "balanced": "madhyam",
            "thorough": "maha"
        }
        sabha = mode_to_sabha.get(mode, "laghu")
    
    working_dir = Path(cwd).resolve() if cwd else Path.cwd()
    
    # Pass sabha directly to constructor (app.config is ParishadConfig dataclass, not dict)
    app = ParishadApp(model=model, sabha=sabha, cwd=str(working_dir))
    
    # Run with inline driver for better Windows compatibility
    try:
        app.run()
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C
        pass
    except Exception as e:
        # Show error and exit cleanly
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_code_cli()
