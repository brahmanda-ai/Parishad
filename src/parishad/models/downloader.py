"""
Model Download Manager for Parishad.

Provides unified interface for downloading and managing LLM models from:
- HuggingFace Hub (GGUF files)
- Ollama (via ollama pull)
- LM Studio (symlinks to existing models)

Models are stored in a central directory for easy management.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

# Environment variable for custom model directory
PARISHAD_MODELS_DIR_ENV = "PARISHAD_MODELS_DIR"


def get_config_file_path() -> Path:
    """Get the path to the parishad config file."""
    if platform.system() == "Darwin":
        return Path.home() / ".parishad" / "config.json"
    elif platform.system() == "Windows":
        app_data = os.environ.get("LOCALAPPDATA", os.environ.get("APPDATA", ""))
        if app_data:
            return Path(app_data) / "parishad" / "config.json"
        return Path.home() / ".parishad" / "config.json"
    else:
        return Path.home() / ".parishad" / "config.json"


def get_user_configured_model_dir() -> Optional[Path]:
    """
    Get user-configured model directory from environment or config file.
    
    Priority:
    1. PARISHAD_MODELS_DIR environment variable
    2. model_dir in ~/.parishad/config.json
    3. None (use default)
    """
    # Check environment variable first
    env_dir = os.environ.get(PARISHAD_MODELS_DIR_ENV)
    if env_dir:
        return Path(env_dir)
    
    # Check config file
    config_file = get_config_file_path()
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
                if "model_dir" in config and config["model_dir"]:
                    return Path(config["model_dir"])
        except (json.JSONDecodeError, IOError):
            pass
    
    return None


def set_model_dir(path: str | Path) -> None:
    """
    Set the custom model directory in config file.
    
    Args:
        path: Path to the directory where models should be stored
    """
    config_file = get_config_file_path()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing config or create new
    config = {}
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Update model_dir
    config["model_dir"] = str(Path(path).resolve())
    
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Model directory set to: {path}")


def get_platform_default_model_dir() -> Path:
    """
    Get the default model directory.
    
    Always uses ~/.parishad/models for consistency across all platforms.
    This ensures a single, predictable location for all Parishad data.
    """
    return Path.home() / ".parishad" / "models"


def get_default_model_dir() -> Path:
    """
    Get the model directory, checking user config first.
    
    Priority:
    1. PARISHAD_MODELS_DIR environment variable
    2. model_dir in ~/.parishad/config.json
    3. Platform-specific default
    
    To set a custom directory:
    - Set PARISHAD_MODELS_DIR environment variable, OR
    - Run: parishad config set-model-dir /path/to/models
    """
    user_dir = get_user_configured_model_dir()
    if user_dir:
        return user_dir
    return get_platform_default_model_dir()


# Default paths
DEFAULT_MODEL_DIR = get_default_model_dir()

# Ollama models directory (platform-specific)
if platform.system() == "Windows":
    # Ollama on Windows stores models in USERPROFILE\.ollama\models
    OLLAMA_MODELS_DIR = Path(os.environ.get("USERPROFILE", Path.home())) / ".ollama" / "models"
else:
    OLLAMA_MODELS_DIR = Path.home() / ".ollama" / "models"

LMSTUDIO_MODELS_DIR = Path.home() / ".lmstudio" / "models"

# Alternative LM Studio paths (varies by platform)
LMSTUDIO_ALT_PATHS = [
    Path.home() / ".cache" / "lm-studio" / "models",
    Path.home() / "Library" / "Application Support" / "LM Studio" / "models",  # macOS
    Path(os.environ.get("LOCALAPPDATA", "")) / "LM Studio" / "models" if platform.system() == "Windows" else Path("/nonexistent"),
]


class ModelSource(Enum):
    """Source of the model."""
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    LOCAL = "local"
    UNKNOWN = "unknown"


class ModelFormat(Enum):
    """Model file format."""
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    OLLAMA = "ollama"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """Information about a downloaded model."""
    name: str
    source: ModelSource
    format: ModelFormat
    path: Path
    size_bytes: int = 0
    downloaded_at: Optional[datetime] = None
    quantization: Optional[str] = None
    base_model: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "source": self.source.value,
            "format": self.format.value,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "downloaded_at": self.downloaded_at.isoformat() if self.downloaded_at else None,
            "quantization": self.quantization,
            "base_model": self.base_model,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            source=ModelSource(data["source"]),
            format=ModelFormat(data["format"]),
            path=Path(data["path"]),
            size_bytes=data.get("size_bytes", 0),
            downloaded_at=datetime.fromisoformat(data["downloaded_at"]) if data.get("downloaded_at") else None,
            quantization=data.get("quantization"),
            base_model=data.get("base_model"),
            metadata=data.get("metadata", {}),
        )
    
    @property
    def size_human(self) -> str:
        """Human-readable size."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


@dataclass
class DownloadProgress:
    """Progress information for downloads."""
    total_bytes: int
    downloaded_bytes: int
    speed_bps: float = 0.0
    eta_seconds: float = 0.0
    model_name: str = ""
    
    @property
    def percentage(self) -> float:
        """Download percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100
    
    @property
    def speed_human(self) -> str:
        """Human-readable speed."""
        speed = self.speed_bps
        for unit in ["B/s", "KB/s", "MB/s", "GB/s"]:
            if speed < 1024:
                return f"{speed:.1f} {unit}"
            speed /= 1024
        return f"{speed:.1f} TB/s"


# Progress callback type
ProgressCallback = Callable[[DownloadProgress], None]


# =============================================================================
# Model Registry
# =============================================================================


class ModelRegistry:
    """
    Registry of downloaded models.
    
    Tracks all models downloaded via the download manager.
    Uses the unified ~/.parishad/config.json file.
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize registry.
        
        Args:
            model_dir: Directory for models (default: platform-specific)
        """
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        # Use unified config file instead of separate registry
        self.config_file = Path.home() / ".parishad" / "config.json"
        self._models: dict[str, ModelInfo] = {}
        
        # Ensure directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self._load()
    
    def _load(self) -> None:
        """Load registry from unified config file."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    data = json.load(f)
                    # Models are stored under 'models' key
                    self._models = {
                        name: ModelInfo.from_dict(info)
                        for name, info in data.get("models", {}).items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load registry from config: {e}")
                self._models = {}
    
    def _save(self) -> None:
        """Save registry to unified config file."""
        try:
            # Load existing config to preserve other fields
            config = {}
            if self.config_file.exists():
                with open(self.config_file) as f:
                    config = json.load(f)
            
            # Update models section
            config["models"] = {
                name: info.to_dict()
                for name, info in self._models.items()
            }
            
            # Write back
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def add(self, model: ModelInfo) -> None:
        """Add or update a model in the registry."""
        self._models[model.name] = model
        self._save()
    
    def remove(self, name: str) -> Optional[ModelInfo]:
        """Remove a model from the registry."""
        model = self._models.pop(name, None)
        if model:
            self._save()
        return model
    
    def get(self, name: str) -> Optional[ModelInfo]:
        """Get model by name."""
        return self._models.get(name)
    
    def list(self, source: Optional[ModelSource] = None) -> list[ModelInfo]:
        """List all models, optionally filtered by source."""
        models = list(self._models.values())
        if source:
            models = [m for m in models if m.source == source]
        return sorted(models, key=lambda m: m.name)
    
    def find_by_path(self, path: Path) -> Optional[ModelInfo]:
        """Find model by path."""
        path = path.resolve()
        for model in self._models.values():
            if model.path.resolve() == path:
                return model
        return None
    
    def exists(self, name: str) -> bool:
        """Check if model exists in registry."""
        return name in self._models

    def verify_integrity(self) -> int:
        """
        Verify that all registered models physically exist.
        Removes invalid entries.
        
        Returns:
            Number of removed entries.
        """
        to_remove = []
        for name, model in self._models.items():
            if model.source == ModelSource.OLLAMA:
                # Ollama models are managed by ollama service, check via list
                # This check is expensive so we might skip or do a lightweight check
                # For now assume if referenced file (json) exists it's ok, 
                # or trust OllamaManager.scan_for_models logic.
                if not model.path.exists():
                     to_remove.append(name)
            elif not model.path.exists():
                to_remove.append(name)
        
        for name in to_remove:
            self.remove(name)
            
        return len(to_remove)


# =============================================================================
# HuggingFace Downloader
# =============================================================================


class HuggingFaceDownloader:
    """
    Download GGUF models from HuggingFace Hub.
    
    Supports downloading specific quantization variants.
    """
    
    # Popular GGUF model repositories
    POPULAR_MODELS = {
        "qwen2.5:0.5b": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        "qwen2.5:1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        "qwen2.5:3b": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
        "qwen2.5:7b": ("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q4_k_m.gguf"),
        "llama3.2:1b": ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
        "llama3.2:3b": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
        "llama3.1:8b": ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
        "phi3:mini": ("microsoft/Phi-3-mini-4k-instruct-gguf", "Phi-3-mini-4k-instruct-q4.gguf"),
        "mistral:7b": ("mistralai/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
        "gemma2:2b": ("bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q4_K_M.gguf"),
        "deepseek:1.5b": ("bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"),
    }
    
    def __init__(self, model_dir: Path):
        """
        Initialize downloader.
        
        Args:
            model_dir: Directory to store downloaded models
        """
        self.model_dir = model_dir
        self.hf_dir = model_dir / "huggingface"
        self.hf_dir.mkdir(parents=True, exist_ok=True)
    
    def list_available(self) -> dict[str, tuple[str, str]]:
        """List available models with shortcuts."""
        return self.POPULAR_MODELS.copy()
    
    def resolve_model(self, model_spec: str) -> tuple[str, str]:
        """
        Resolve model specification to repo and filename.
        
        Args:
            model_spec: Model name (e.g., "qwen2.5:1.5b") or repo/file
            
        Returns:
            Tuple of (repo_id, filename)
        """
        # Check shortcuts first
        if model_spec in self.POPULAR_MODELS:
            return self.POPULAR_MODELS[model_spec]
        
        # Parse as repo/filename
        if "/" in model_spec:
            parts = model_spec.split("/")
            if len(parts) >= 3:
                # Format: owner/repo/filename
                repo_id = f"{parts[0]}/{parts[1]}"
                filename = "/".join(parts[2:])
                return repo_id, filename
            elif len(parts) == 2:
                # Just repo, need to find GGUF file
                return model_spec, ""
        
        raise ValueError(f"Unknown model: {model_spec}. Use format 'owner/repo/file.gguf' or a shortcut like 'qwen2.5:1.5b'")
    
    def download(
        self,
        model_spec: str,
        quantization: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ModelInfo:
        """
        Download a model from HuggingFace.
        
        Args:
            model_spec: Model specification (shortcut or repo/file)
            quantization: Preferred quantization (e.g., "q4_k_m", "q8_0")
            progress_callback: Callback for progress updates
            
        Returns:
            ModelInfo for the downloaded model
        """
        repo_id, filename = self.resolve_model(model_spec)
        
        # If no filename, try to find one
        if not filename:
            filename = self._find_gguf_file(repo_id, quantization)
        
        # Construct download URL
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        
        # Determine local path
        safe_name = repo_id.replace("/", "_")
        local_path = self.hf_dir / safe_name / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        self._download_file(url, local_path, progress_callback)
        
        # Determine quantization from filename
        quant = self._extract_quantization(filename)
        
        # Create model info
        model_name = model_spec if model_spec in self.POPULAR_MODELS else f"hf:{repo_id}/{filename}"
        
        return ModelInfo(
            name=model_name,
            source=ModelSource.HUGGINGFACE,
            format=ModelFormat.GGUF,
            path=local_path,
            size_bytes=local_path.stat().st_size,
            downloaded_at=datetime.now(),
            quantization=quant,
            base_model=repo_id,
            metadata={
                "repo_id": repo_id,
                "filename": filename,
                "url": url,
            },
        )
    
    def _find_gguf_file(self, repo_id: str, quantization: Optional[str] = None) -> str:
        """Find a GGUF file in the repository."""
        try:
            # Try using huggingface_hub if available
            from huggingface_hub import list_repo_files
            
            # Wrap in try-except to handle API failures (rate limits, auth, network)
            try:
                files = list_repo_files(repo_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]
                
                if not gguf_files:
                    raise ValueError(f"No GGUF files found in {repo_id}")
                
                # Prefer requested quantization
                if quantization:
                    for f in gguf_files:
                        if quantization.lower() in f.lower():
                            return f
                
                # Prefer Q4_K_M as default
                for f in gguf_files:
                    if "q4_k_m" in f.lower():
                        return f
                
                # Return first GGUF file
                return gguf_files[0]
                
            except Exception as e:
                # If listing fails (or import works but call fails), fall back to guessing
                # This protects against API flakiness
                if isinstance(e, ValueError) and "No GGUF files found" in str(e):
                    raise # Re-raise valid empty repo errors
                    
                logger.warning(f"Failed to list files in {repo_id}: {e}. Falling back to filename guessing.")
                raise ImportError("Force fallback") # Trigger fallback logic
                
        except (ImportError, Exception):
            # Fallback: Guess the filename based on repo name
            # Most GGUF repos (like bartowski) follow: {ModelName}-{Quant}.gguf
            repo_name = repo_id.split("/")[-1]
            
            # Strip '-GGUF' suffix if present (common in repo names but not filenames)
            if repo_name.lower().endswith("-gguf"):
                repo_name = repo_name[:-5]
                
            quant_suffix = quantization if quantization else "Q4_K_M"
            
            # Construct standard guess
            # Example: Llama-3.2-3B-Instruct -> Llama-3.2-3B-Instruct-Q4_K_M.gguf
            guessed_filename = f"{repo_name}-{quant_suffix}.gguf"
            
            logger.info(f"Guessed filename: {guessed_filename}")
            return guessed_filename
    
    def _extract_quantization(self, filename: str) -> Optional[str]:
        """Extract quantization from filename."""
        filename_lower = filename.lower()
        
        quantizations = [
            "q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
            "q4_0", "q4_1", "q4_k_s", "q4_k_m",
            "q5_0", "q5_1", "q5_k_s", "q5_k_m",
            "q6_k", "q8_0", "f16", "f32",
        ]
        
        for quant in quantizations:
            if quant in filename_lower:
                return quant.upper()
        
        return None
    
    def _download_file(
        self,
        url: str,
        dest: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Download file with progress tracking."""
        import urllib.request
        
        # Check if already exists
        if dest.exists():
            logger.info(f"Model already exists: {dest}")
            return
        
        logger.info(f"Downloading from {url}")
        
        # Get file size
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get("Content-Length", 0))
        except Exception as e:
            logger.error(f"Failed to get file size from {url}: {e}")
            raise RuntimeError(f"Cannot access model at {url}. Error: {e}")
        
        # Download with progress
        downloaded = 0
        start_time = datetime.now()
        
        temp_dest = dest.with_suffix(".download")
        
        try:
            with urllib.request.urlopen(url) as response:
                with open(temp_dest, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback:
                            elapsed = (datetime.now() - start_time).total_seconds()
                            speed = downloaded / elapsed if elapsed > 0 else 0
                            eta = (total_size - downloaded) / speed if speed > 0 else 0
                            
                            progress_callback(DownloadProgress(
                                total_bytes=total_size,
                                downloaded_bytes=downloaded,
                                speed_bps=speed,
                                eta_seconds=eta,
                            ))
            
            # Move to final location
            temp_dest.rename(dest)
            logger.info(f"Downloaded: {dest}")
            
        except Exception as e:
            # Clean up partial download
            if temp_dest.exists():
                temp_dest.unlink()
            raise


# =============================================================================
# Ollama Integration
# =============================================================================


class OllamaManager:
    """
    Manage models through Ollama.
    
    Uses the ollama CLI to pull and manage models.
    """
    
    def __init__(self, model_dir: Path):
        """
        Initialize Ollama manager.
        
        Args:
            model_dir: Directory for model symlinks/info
        """
        self.model_dir = model_dir
        self.ollama_dir = model_dir / "ollama"
        self.ollama_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def is_available() -> bool:
        """Check if Ollama is installed and running."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def list_models(self) -> list[dict]:
        """List models available in Ollama."""
        if not self.is_available():
            return []
        
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                return []
            
            models = []
            lines = result.stdout.strip().split("\n")
            
            # Skip header
            for line in lines[1:]:
                if not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    models.append({
                        "name": parts[0],
                        "size": parts[1] if len(parts) > 1 else "unknown",
                    })
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    def pull(
        self,
        model_name: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ModelInfo:
        """
        Pull a model using Ollama.
        
        Args:
            model_name: Model name (e.g., "llama3.2:1b", "qwen2.5:0.5b")
            progress_callback: Callback for progress updates
            
        Returns:
            ModelInfo for the pulled model
        """
        import re
        
        if not self.is_available():
            raise RuntimeError("Ollama is not installed or not running. Install from https://ollama.ai")
        
        logger.info(f"Pulling model via Ollama: {model_name}")
        
        # Run ollama pull with output suppressed (we'll parse stderr for progress)
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )
        
        # Parse Ollama's progress output and update our callback
        # Ollama outputs lines like: "pulling 2bada8a74506:   2% ▕  ▏  75 MB/4.7 GB"
        size_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(MB|GB)/(\d+(?:\.\d+)?)\s*(MB|GB)')
        percent_pattern = re.compile(r'(\d+)%')
        
        last_progress = None
        
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse progress from Ollama output
            if progress_callback:
                # Extract percentage
                percent_match = percent_pattern.search(line)
                size_match = size_pattern.search(line)
                
                if size_match:
                    # Parse downloaded and total size
                    downloaded = float(size_match.group(1))
                    downloaded_unit = size_match.group(2)
                    total = float(size_match.group(3))
                    total_unit = size_match.group(4)
                    
                    # Convert to bytes
                    downloaded_bytes = int(downloaded * (1024**3 if downloaded_unit == "GB" else 1024**2))
                    total_bytes = int(total * (1024**3 if total_unit == "GB" else 1024**2))
                    
                    progress = DownloadProgress(
                        downloaded_bytes=downloaded_bytes,
                        total_bytes=total_bytes,
                        speed_bps=0.0,
                        model_name=model_name,
                    )
                    
                    # Only update if progress changed significantly
                    if last_progress is None or downloaded_bytes - last_progress >= 1024 * 1024:  # 1MB
                        progress_callback(progress)
                        last_progress = downloaded_bytes
            
            # Log non-progress lines (like "pulling manifest", "verifying sha256")
            if not percent_pattern.search(line):
                logger.debug(f"Ollama: {line}")
        
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"Failed to pull model: {model_name}")
        
        # Signal completion
        if progress_callback:
            progress_callback(DownloadProgress(
                downloaded_bytes=1,
                total_bytes=1,
                speed_bps=0.0,
                model_name=model_name,
            ))
        
        # Get model info
        models = self.list_models()
        model_info = next((m for m in models if m["name"].startswith(model_name)), None)
        
        # Create a reference file
        ref_file = self.ollama_dir / f"{model_name.replace(':', '_')}.json"
        with open(ref_file, "w") as f:
            json.dump({
                "name": model_name,
                "source": "ollama",
                "pulled_at": datetime.now().isoformat(),
            }, f)
        
        return ModelInfo(
            name=f"ollama:{model_name}",
            source=ModelSource.OLLAMA,
            format=ModelFormat.OLLAMA,
            path=ref_file,  # Reference file, actual model in Ollama's storage
            size_bytes=0,  # Ollama manages this
            downloaded_at=datetime.now(),
            metadata={
                "ollama_name": model_name,
                "size": model_info.get("size") if model_info else "unknown",
            },
        )
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get the path to an Ollama model (for direct access)."""
        # Ollama stores models in a specific structure
        # This is for information only - use ollama CLI for actual inference
        blob_dir = OLLAMA_MODELS_DIR / "blobs"
        
        if blob_dir.exists():
            # Models are stored as blobs with SHA256 names
            # We can't easily map name to blob, so return the dir
            return blob_dir
        
        return None


# =============================================================================
# LM Studio Integration
# =============================================================================


class LMStudioManager:
    """
    Manage models from LM Studio.
    
    LM Studio stores GGUF models in a standard directory.
    This manager finds and symlinks those models.
    """
    
    def __init__(self, model_dir: Path):
        """
        Initialize LM Studio manager.
        
        Args:
            model_dir: Directory for model symlinks
        """
        self.model_dir = model_dir
        self.lmstudio_dir = model_dir / "lmstudio"
        self.lmstudio_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def find_lmstudio_dir(cls) -> Optional[Path]:
        """Find the LM Studio models directory."""
        # Check primary path
        if LMSTUDIO_MODELS_DIR.exists():
            return LMSTUDIO_MODELS_DIR
        
        # Check alternative paths
        for alt_path in LMSTUDIO_ALT_PATHS:
            if alt_path.exists():
                return alt_path
        
        return None
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if LM Studio models are available."""
        return cls.find_lmstudio_dir() is not None
    
    def list_models(self) -> list[dict]:
        """List models available in LM Studio."""
        lmstudio_dir = self.find_lmstudio_dir()
        
        if not lmstudio_dir:
            return []
        
        models = []
        
        # Walk through the models directory
        for gguf_file in lmstudio_dir.rglob("*.gguf"):
            try:
                size = gguf_file.stat().st_size
                rel_path = gguf_file.relative_to(lmstudio_dir)
                
                models.append({
                    "name": str(rel_path),
                    "path": gguf_file,
                    "size_bytes": size,
                })
            except Exception as e:
                logger.warning(f"Failed to read model {gguf_file}: {e}")
        
        return models
    
    def import_model(self, model_path: str) -> ModelInfo:
        """
        Import a model from LM Studio.
        
        Args:
            model_path: Relative path within LM Studio models dir
            
        Returns:
            ModelInfo for the imported model
        """
        lmstudio_dir = self.find_lmstudio_dir()
        
        if not lmstudio_dir:
            raise RuntimeError("LM Studio models directory not found")
        
        source_path = lmstudio_dir / model_path
        
        if not source_path.exists():
            # Try as absolute path
            source_path = Path(model_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create symlink in our directory
        safe_name = Path(model_path).name
        link_path = self.lmstudio_dir / safe_name
        
        if link_path.exists():
            link_path.unlink()
        
        link_path.symlink_to(source_path)
        
        return ModelInfo(
            name=f"lmstudio:{safe_name}",
            source=ModelSource.LMSTUDIO,
            format=ModelFormat.GGUF,
            path=link_path,
            size_bytes=source_path.stat().st_size,
            downloaded_at=datetime.now(),
            quantization=self._extract_quantization(safe_name),
            metadata={
                "original_path": str(source_path),
                "lmstudio_path": model_path,
            },
        )
    
    def _extract_quantization(self, filename: str) -> Optional[str]:
        """Extract quantization from filename."""
        filename_lower = filename.lower()
        
        quantizations = [
            "q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
            "q4_0", "q4_1", "q4_k_s", "q4_k_m",
            "q5_0", "q5_1", "q5_k_s", "q5_k_m",
            "q6_k", "q8_0", "f16", "f32",
        ]
        
        for quant in quantizations:
            if quant in filename_lower:
                return quant.upper()
        
        return None


# =============================================================================
# Unified Model Manager
# =============================================================================


class ModelManager:
    """
    Unified interface for managing LLM models.
    
    Provides a single entry point for:
    - Downloading from HuggingFace
    - Pulling from Ollama
    - Importing from LM Studio
    - Listing all available models
    
    Usage:
        manager = ModelManager()
        
        # Download from HuggingFace
        model = manager.download("qwen2.5:1.5b", source="huggingface")
        
        # Pull from Ollama
        model = manager.download("llama3.2:1b", source="ollama")
        
        # Import from LM Studio
        model = manager.download("author/model.gguf", source="lmstudio")
        
        # List all models
        models = manager.list_models()
        
        # Get model path for inference
        path = manager.get_model_path("qwen2.5:1.5b")
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize model manager.
        
        Args:
            model_dir: Directory for models (default: platform-specific)
        """
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.registry = ModelRegistry(self.model_dir)
        self.huggingface = HuggingFaceDownloader(self.model_dir)
        self.ollama = OllamaManager(self.model_dir)
        self.lmstudio = LMStudioManager(self.model_dir)
        
        logger.info(f"Model manager initialized: {self.model_dir}")
    
    def download(
        self,
        model_spec: str,
        source: str = "auto",
        quantization: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ModelInfo:
        """
        Download or import a model.
        
        Tries multiple sources as fallbacks if the primary source fails.
        
        Args:
            model_spec: Model specification (name, path, or URL)
            source: Source to use ("huggingface", "ollama", "lmstudio", or "auto")
            quantization: Preferred quantization (for HuggingFace)
            progress_callback: Progress callback function
            
        Returns:
            ModelInfo for the downloaded model
        """
        source = source.lower()
        
        # Auto-detect source
        if source == "auto":
            source = self._detect_source(model_spec)
        
        # Check if already downloaded
        existing = self.registry.get(model_spec)
        if existing and existing.path.exists():
            logger.info(f"Model already available: {model_spec}")
            return existing
        
        # Define fallback order based on primary source
        if source == "huggingface":
            sources_to_try = ["huggingface", "ollama", "lmstudio"]
        elif source == "ollama":
            sources_to_try = ["ollama", "huggingface", "lmstudio"]
        elif source == "lmstudio":
            sources_to_try = ["lmstudio", "huggingface", "ollama"]
        else:
            sources_to_try = [source]
        
        errors = []
        
        for try_source in sources_to_try:
            try:
                model = self._download_from_source(
                    model_spec, try_source, quantization, progress_callback
                )
                # Register the model
                self.registry.add(model)
                return model
            except Exception as e:
                error_msg = f"{try_source}: {e}"
                errors.append(error_msg)
                logger.warning(f"Failed to download from {try_source}: {e}")
                continue
        
        # All sources failed
        raise RuntimeError(
            f"Failed to download '{model_spec}' from all sources:\n" +
            "\n".join(f"  - {err}" for err in errors)
        )
    
    def _download_from_source(
        self,
        model_spec: str,
        source: str,
        quantization: Optional[str],
        progress_callback: Optional[ProgressCallback],
    ) -> ModelInfo:
        """Download from a specific source."""
        if source == "huggingface":
            return self.huggingface.download(model_spec, quantization, progress_callback)
        elif source == "ollama":
            # Convert model spec to Ollama format if needed
            ollama_name = self._convert_to_ollama_name(model_spec)
            return self.ollama.pull(ollama_name, progress_callback)
        elif source == "lmstudio":
            return self.lmstudio.import_model(model_spec)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def _convert_to_ollama_name(self, model_spec: str) -> str:
        """Convert a model specification to Ollama format."""
        # Common mappings from shortcut to Ollama model name
        ollama_mappings = {
            "qwen2.5:0.5b": "qwen2.5:0.5b",
            "qwen2.5:1.5b": "qwen2.5:1.5b",
            "qwen2.5:3b": "qwen2.5:3b",
            "qwen2.5:7b": "qwen2.5:7b",
            "llama3.2:1b": "llama3.2:1b",
            "llama3.2:3b": "llama3.2:3b",
            "llama3.1:8b": "llama3.1:8b",
            "phi3:mini": "phi3:mini",
            "mistral:7b": "mistral:7b-instruct",
            "gemma2:2b": "gemma2:2b",
            "deepseek:1.5b": "deepseek-r1:1.5b",
            "deepseek:7b": "deepseek-r1:7b",
        }
        return ollama_mappings.get(model_spec, model_spec)
    
    def _detect_source(self, model_spec: str) -> str:
        """Auto-detect the source for a model specification."""
        # Check shortcuts first
        if model_spec in self.huggingface.POPULAR_MODELS:
            return "huggingface"
        
        # Check prefixes
        if model_spec.startswith("hf:") or model_spec.startswith("huggingface:"):
            return "huggingface"
        if model_spec.startswith("ollama:"):
            return "ollama"
        if model_spec.startswith("lmstudio:"):
            return "lmstudio"
        
        # Check if it looks like a HuggingFace repo
        if "/" in model_spec and ".gguf" in model_spec.lower():
            return "huggingface"
        
        # Check if Ollama has it
        if self.ollama.is_available():
            return "ollama"
        
        # Default to HuggingFace
        return "huggingface"
    
    def list_models(self, source: Optional[str] = None) -> list[ModelInfo]:
        """
        List all downloaded models.
        
        Args:
            source: Filter by source (optional)
            
        Returns:
            List of ModelInfo objects
        """
        source_enum = ModelSource(source) if source else None
        return self.registry.list(source_enum)
    
    def get_model_path(self, name: str) -> Optional[Path]:
        """
        Get the path to a model for inference.
        
        Args:
            name: Model name (e.g., "qwen2.5:7b" or "ollama:qwen2.5:7b")
            
        Returns:
            Path to model file, or None if not found
        """
        # Try exact name first
        model = self.registry.get(name)
        
        # Try with ollama: prefix
        if not model:
            model = self.registry.get(f"ollama:{name}")
        
        # Try with hf: prefix  
        if not model:
            model = self.registry.get(f"hf:{name}")
        
        # Try searching by partial name match
        if not model:
            for registered_name, registered_model in self.registry._models.items():
                if name in registered_name or registered_name.endswith(name):
                    model = registered_model
                    break
        
        # Check if model exists and return path
        if model and model.path.exists():
            # For symlinks, return the symlink path (not resolved)
            # This allows llama.cpp to load from symlink
            return model.path
        
        # Try finding directly in ollama folder
        ollama_symlink = self.model_dir / "ollama" / f"{name.replace(':', '_')}.gguf"
        if ollama_symlink.exists():
            return ollama_symlink
        
        return None
    
    def remove_model(self, name: str, delete_files: bool = True) -> bool:
        """
        Remove a model from the registry.
        
        Args:
            name: Model name
            delete_files: Also delete the model files
            
        Returns:
            True if model was removed
        """
        model = self.registry.remove(name)
        
        if model and delete_files:
            try:
                if model.path.exists():
                    if model.path.is_dir():
                        shutil.rmtree(model.path)
                    else:
                        model.path.unlink()
                logger.info(f"Deleted model files: {model.path}")
            except Exception as e:
                logger.error(f"Failed to delete model files: {e}")
        
        return model is not None
    
    def get_available_sources(self) -> dict[str, bool]:
        """Get availability status of each source."""
        return {
            "huggingface": True,  # Always available (uses urllib)
            "ollama": self.ollama.is_available(),
            "lmstudio": self.lmstudio.is_available(),
        }
    
    def scan_for_models(self) -> list[ModelInfo]:
        """
        Scan for models that aren't in the registry.
        
        Finds GGUF files in the model directory and LM Studio.
        
        Returns:
            List of newly discovered models
        """
        discovered = []
        
        # Scan model directory for GGUF files
        search_paths = [self.model_dir]
        
        # Add default HuggingFace Hub cache
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            search_paths.append(hf_cache)
            
        # Add default LM Studio cache (MacOS)
        lms_cache = Path.home() / ".cache" / "lm-studio" / "models" 
        if lms_cache.exists():
            search_paths.append(lms_cache)
        
        # Also check standard MacOS Application Support 
        # (Where LM Studio often stores them by default on Mac)
        lms_app_support = Path.home() / "Library" / "Application Support" / "LM Studio" / "models"
        if lms_app_support.exists():
             search_paths.append(lms_app_support)

        for base_path in search_paths:
            # Scan for GGUF and Transformers weights
            extensions = ["*.gguf", "*.safetensors", "pytorch_model.bin"]
            candidates = []
            for ext in extensions:
                candidates.extend(base_path.rglob(ext))

            for model_file in candidates:
                if "blobs" in str(model_file): continue 
                
                # Check if already registered
                if self.registry.find_by_path(model_file):
                    continue
                
                try:
                    size = model_file.stat().st_size
                    
                    # Determine format/backend hint
                    name_prefix = "local"
                    fmt = ModelFormat.GGUF
                    src = ModelSource.LOCAL
                    
                    if model_file.suffix in [".safetensors", ".bin"]:
                        fmt = ModelFormat.SAFETENSORS if model_file.suffix == ".safetensors" else ModelFormat.PYTORCH
                        
                        # Extract repo name from HF cache path structure
                        # Path looks like: .../models--Organization--Repo/snapshots/hash/model.safetensors
                        repo_name = None
                        for p in model_file.parts:
                            if p.startswith("models--"):
                                repo_name = p.replace("models--", "").replace("--", "/")
                                break
                        
                        if repo_name:
                            name_prefix = repo_name
                        else:
                            name_prefix = "hf_local"
                    
                    # Improve name if filename is generic
                    stem = model_file.stem
                    if stem in ["model", "pytorch_model", "adapter_model"] or stem.startswith("model-00"):
                        # Use repo name if we have it, otherwise parent dir
                        if name_prefix and name_prefix != "local" and name_prefix != "hf_local":
                            stem = name_prefix.split("/")[-1]  # Use just the model name part
                        else:
                            stem = model_file.parent.name
                    
                    # Filter out non-generative models (BERT, encoders, etc.)
                    # Also filter out MLX-quantized models (incompatible with transformers)
                    # Skip models that are clearly not for text generation or incompatible
                    skip_patterns = ["bert", "roberta", "albert", "electra", "deberta", "muril", "xlm-roberta"]
                    model_name_lower = stem.lower()
                    path_lower = str(model_file).lower()
                    
                    # Skip non-generative models
                    if any(pattern in model_name_lower for pattern in skip_patterns):
                        continue
                    
                    # Skip MLX models (they're incompatible with transformers backend)
                    if "mlx" in path_lower:
                        continue

                    model = ModelInfo(
                        name=f"{name_prefix}:{stem}",
                        source=src,
                        format=fmt,
                        path=model_file,
                        size_bytes=size,
                    )
                    self.registry.add(model)
                    discovered.append(model)
                except OSError:
                    continue
        
        # Scan LM Studio
        for lms_model in self.lmstudio.list_models():
            name = f"lmstudio:{lms_model['name']}"
            if not self.registry.get(name):
                model = ModelInfo(
                    name=name,
                    source=ModelSource.LMSTUDIO,
                    format=ModelFormat.GGUF,
                    path=lms_model["path"],
                    size_bytes=lms_model["size_bytes"],
                )
                self.registry.add(model)
                discovered.append(model)
        
        # Reconcile Ollama
        try:
            live_ollama = {m['name']: m for m in self.ollama.list_models()}
            live_names = set(live_ollama.keys())
            
            # Prune stale
            for existing in self.registry.list(ModelSource.OLLAMA):
                # Check if existing.metadata['ollama_name'] is in live_names
                # Or try to parse from name
                o_name = existing.name.replace("ollama:", "")
                if o_name not in live_names:
                    # Stale entry, remove it
                    self.registry.remove(existing.name)
                    # Also try to remove the JSON proxy file if we created it
                    if existing.path and existing.path.suffix == ".json" and "ollama" in str(existing.path):
                         try: existing.path.unlink()
                         except OSError: pass
            
            # Add new/current
            for name, o_model in live_ollama.items():
                reg_name = f"ollama:{name}"
                if not self.registry.get(reg_name):
                     ref_file = self.ollama.ollama_dir / f"{name.replace(':', '_')}.json"
                     if not ref_file.exists():
                        try:
                            with open(ref_file, "w") as f:
                                json.dump({
                                    "name": name,
                                    "source": "ollama",
                                    "auto_discovered": True
                                }, f)
                        except OSError: continue
                    
                     model = ModelInfo(
                        name=reg_name,
                        source=ModelSource.OLLAMA,
                        format=ModelFormat.OLLAMA,
                        path=ref_file,
                        size_bytes=0, # Unknown
                        downloaded_at=datetime.now()
                     )
                     self.registry.add(model)
                     discovered.append(model)
        except Exception as e:
            logger.warning(f"Ollama reconciliation failed: {e}")
                


        return discovered


# =============================================================================
# CLI Helper Functions
# =============================================================================


def print_progress(progress: DownloadProgress) -> None:
    """Print download progress to terminal with in-place update."""
    bar_width = 40
    filled = int(bar_width * progress.percentage / 100)
    bar = "=" * filled + "-" * (bar_width - filled)
    
    # Build the progress line
    line = (
        f"[{bar}] {progress.percentage:.1f}% "
        f"({progress.downloaded_bytes / 1024 / 1024:.1f}MB) "
        f"{progress.speed_human}"
    )
    
    # Clear line and write progress (use ANSI escape to clear to end of line)
    sys.stdout.write(f"\r\033[K{line}")
    sys.stdout.flush()
    
    if progress.percentage >= 100:
        sys.stdout.write("\n")  # Newline when done
        sys.stdout.flush()


def interactive_download(manager: ModelManager) -> Optional[ModelInfo]:
    """Interactive model download wizard."""
    print("\n=== Parishad Model Download ===\n")
    
    # Show available sources
    sources = manager.get_available_sources()
    print("Available sources:")
    print(f"  1. HuggingFace (GGUF models) - {'✓ Available' if sources['huggingface'] else '✗ Not available'}")
    print(f"  2. Ollama                    - {'✓ Available' if sources['ollama'] else '✗ Not installed'}")
    print(f"  3. LM Studio                 - {'✓ Available' if sources['lmstudio'] else '✗ Not found'}")
    
    print("\nSelect source (1-3): ", end="")
    choice = input().strip()
    
    if choice == "1":
        return _download_from_huggingface(manager)
    elif choice == "2":
        return _download_from_ollama(manager)
    elif choice == "3":
        return _download_from_lmstudio(manager)
    else:
        print("Invalid choice")
        return None


def _download_from_huggingface(manager: ModelManager) -> Optional[ModelInfo]:
    """Interactive HuggingFace download."""
    print("\nPopular models:")
    models = list(manager.huggingface.POPULAR_MODELS.keys())
    for i, name in enumerate(models, 1):
        print(f"  {i}. {name}")
    print(f"  {len(models) + 1}. Custom (enter repo/filename)")
    
    print("\nSelect model: ", end="")
    choice = input().strip()
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            model_spec = models[idx]
        elif idx == len(models):
            print("Enter HuggingFace repo/filename: ", end="")
            model_spec = input().strip()
        else:
            print("Invalid choice")
            return None
    except ValueError:
        # Treat as model name directly
        model_spec = choice
    
    print(f"\nDownloading {model_spec}...")
    return manager.download(model_spec, source="huggingface", progress_callback=print_progress)


def _download_from_ollama(manager: ModelManager) -> Optional[ModelInfo]:
    """Interactive Ollama download."""
    if not manager.ollama.is_available():
        print("\nOllama is not installed. Install from: https://ollama.ai")
        return None
    
    # Show existing models
    existing = manager.ollama.list_models()
    if existing:
        print("\nModels already in Ollama:")
        for m in existing:
            print(f"  - {m['name']} ({m['size']})")
    
    print("\nEnter model name to pull (e.g., 'llama3.2:1b', 'qwen2.5:0.5b'): ", end="")
    model_name = input().strip()
    
    if not model_name:
        return None
    
    print(f"\nPulling {model_name} via Ollama...")
    return manager.download(model_name, source="ollama")


def _download_from_lmstudio(manager: ModelManager) -> Optional[ModelInfo]:
    """Interactive LM Studio import."""
    if not manager.lmstudio.is_available():
        print("\nLM Studio models directory not found.")
        print("Expected locations:")
        print(f"  - {LMSTUDIO_MODELS_DIR}")
        for path in LMSTUDIO_ALT_PATHS:
            print(f"  - {path}")
        return None
    
    models = manager.lmstudio.list_models()
    
    if not models:
        print("\nNo GGUF models found in LM Studio.")
        return None
    
    print("\nModels in LM Studio:")
    for i, m in enumerate(models, 1):
        size_mb = m["size_bytes"] / 1024 / 1024
        print(f"  {i}. {m['name']} ({size_mb:.1f} MB)")
    
    print("\nSelect model to import: ", end="")
    choice = input().strip()
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            model = models[idx]
            print(f"\nImporting {model['name']}...")
            return manager.download(model["name"], source="lmstudio")
    except ValueError:
        pass
    
    print("Invalid choice")
    return None
