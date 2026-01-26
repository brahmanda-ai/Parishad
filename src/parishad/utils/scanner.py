"""
Smart Model Scanner for Parishad.
Scans standard locations for existing LLMs to avoid redundant downloads.
"""
import shutil
import subprocess
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FoundModel:
    name: str
    source: str  # "ollama", "huggingface", "lmstudio"
    path: str = ""
    size_gb: float = 0.0

    def __str__(self):
        return f"{self.name} ({self.source}, {self.size_gb:.1f}GB)"


class ModelScanner:
    """Scans system for local LLMs."""

    def scan_all(self) -> List[FoundModel]:
        """Run all scans and return combined results."""
        models = []
        models.extend(self.scan_ollama())
        models.extend(self.scan_huggingface())
        return models

    def scan_ollama(self) -> List[FoundModel]:
        """Check for Ollama models via CLI."""
        models = []
        if not shutil.which("ollama"):
            return []

        try:
            # Run 'ollama list'
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode != 0:
                return []

            # Parse output (skip header)
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return []

            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 1:
                    name = parts[0]
                    # Simple size parsing if available, else 0
                    size_gb = 0.0
                    # Parse size (handles "2.0 GB" and "2.0GB")
                    size_gb = 0.0
                    try:
                        # Case 1: "2.0 GB" (separate tokens)
                        if len(parts) >= 4 and parts[3] == 'GB':
                            size_gb = float(parts[2])
                        # Case 2: "2.0GB" (merged token)
                        elif len(parts) >= 3 and 'GB' in parts[2]:
                            size_gb = float(parts[2].replace('GB', ''))
                        # Case 3: MB support
                        elif len(parts) >= 4 and parts[3] == 'MB':
                            size_gb = float(parts[2]) / 1024
                    except ValueError:
                        pass
                    
                    models.append(FoundModel(
                        name=name,
                        source="ollama",
                        path=str(Path.home() / ".ollama" / "models"),
                        size_gb=size_gb
                    ))
                    
        except Exception as e:
            logger.warning(f"Ollama scan failed: {e}")
            
        return models

    def scan_huggingface(self) -> List[FoundModel]:
        """Scan ~/.cache/huggingface/hub for models."""
        models = []
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        
        if not cache_dir.exists():
            return []

        try:
            for item in cache_dir.iterdir():
                if item.is_dir() and item.name.startswith("models--"):
                    # Format: models--org--modelname
                    # Convert to org/modelname
                    raw_name = item.name.replace("models--", "")
                    clean_name = raw_name.replace("--", "/")
                    
                    # Calculate size
                    size_gb = sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024**3)
                    
                    models.append(FoundModel(
                        name=clean_name,
                        source="huggingface",
                        path=str(item),
                        size_gb=size_gb
                    ))
        except Exception as e:
            logger.warning(f"HF scan failed: {e}")
            
        return models

    def scan_directory(self, root_path: Path, min_size_gb: float = 0.5) -> List[FoundModel]:
        """
        Deep scan a directory recursively for large model files.
        Looks for .gguf, .bin, .safetensors > min_size_gb.
        """
        models = []
        extensions = {'.gguf', '.bin', '.safetensors', '.pt', '.pth'}
        
        # Skip these common potentially huge/slow dirs
        skip_dirs = {'.git', '.vscode', '.idea', '__pycache__', 'node_modules', 'Library', 'AppData'}
        
        if not root_path.exists():
            return []
            
        try:
            # Walk the tree
            for path in root_path.rglob('*'):
                # 1. Check if we should skip this directory (optimization)
                if path.is_dir():
                    if path.name in skip_dirs or path.name.startswith('.'):
                        continue
                        
                # 2. Check file criteria
                if path.is_file() and path.suffix.lower() in extensions:
                    try:
                        size_bytes = path.stat().st_size
                        size_gb = size_bytes / (1024**3)
                        
                        if size_gb >= min_size_gb:
                            # Found a potential model!
                            models.append(FoundModel(
                                name=path.name,
                                source="local_file",
                                path=str(path),
                                size_gb=size_gb
                            ))
                    except (PermissionError, OSError):
                        continue
                        
        except Exception as e:
            logger.warning(f"Deep scan error at {root_path}: {e}")
            
        return models
