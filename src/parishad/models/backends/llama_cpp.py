"""
Llama.cpp backend for GGUF models.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import Optional

from .base import BackendConfig, BackendError, BackendResult, BaseBackend

logger = logging.getLogger(__name__)

# Suppress verbose llama.cpp logging (ggml_metal_init messages)
os.environ.setdefault("GGML_METAL_LOG_LEVEL", "0")
os.environ.setdefault("LLAMA_CPP_LOG_LEVEL", "0")

# Lazy imports
_llama_cpp = None
_suppress_output = None


def _get_llama_cpp():
    """Lazy import of llama-cpp-python."""
    global _llama_cpp, _suppress_output
    if _llama_cpp is None:
        try:
            import llama_cpp
            from llama_cpp import suppress_stdout_stderr
            _llama_cpp = llama_cpp
            _suppress_output = suppress_stdout_stderr
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppBackend. "
                "Install with: pip install llama-cpp-python"
            )
    return _llama_cpp


def resolve_model_path(model_id: str) -> Optional[Path]:
    """Resolve a model ID to a file path."""
    # Direct path
    direct = Path(model_id)
    if direct.exists():
        return direct
    
    # Try model manager
    try:
        from ..downloader import ModelManager, get_default_model_dir
        manager = ModelManager()
        
        # Check registry
        path = manager.get_model_path(model_id)
        if path and path.exists():
            return path
        
        # Check ollama symlinks
        ollama_dir = get_default_model_dir() / "ollama"
        if ollama_dir.exists():
            safe_name = model_id.replace(":", "_").replace("/", "_")
            for suffix in [".gguf", ""]:
                candidate = ollama_dir / f"{safe_name}{suffix}"
                if candidate.exists():
                    return candidate
        
        # Search for matching GGUF files
        for model in manager.list_models():
            if model.format.value == "gguf":
                if model_id in model.name or model_id in str(model.path):
                    return model.path
    except Exception as e:
        logger.debug(f"Model manager lookup failed: {e}")
    
    # Common locations
    search_paths = [
        Path.cwd() / "models",
        Path.home() / ".cache" / "parishad" / "models",
        Path.home() / ".local" / "share" / "parishad" / "models",
    ]
    
    for search_dir in search_paths:
        if search_dir.exists():
            candidate = search_dir / model_id
            if candidate.exists():
                return candidate
            for gguf_file in search_dir.rglob("*.gguf"):
                if model_id in gguf_file.name:
                    return gguf_file
    
    return None


class LlamaCppBackend(BaseBackend):
    """Backend for GGUF models using llama-cpp-python."""
    
    _name = "llama_cpp"
    
    def __init__(self):
        """Initialize LlamaCppBackend."""
        super().__init__()
        self._llm = None
    
    def load(self, config: BackendConfig) -> None:
        """Load a GGUF model."""
        llama_cpp = _get_llama_cpp()
        
        model_path = resolve_model_path(config.model_id)
        
        if model_path is None:
            raise BackendError(
                f"Model not found: {config.model_id}. "
                "Download with: parishad download <model_name>",
                backend_name=self._name,
                model_id=config.model_id,
            )
        
        extra = config.extra or {}
        n_gpu_layers = extra.get("n_gpu_layers", -1)
        n_ctx = extra.get("n_ctx", config.context_length)
        n_batch = extra.get("n_batch", 512)
        verbose = extra.get("verbose", False)
        chat_format = extra.get("chat_format", None)
        
        try:
            suppress_ctx = _suppress_output(disable=False) if _suppress_output else None
            if suppress_ctx:
                with suppress_ctx:
                    self._llm = llama_cpp.Llama(
                        model_path=str(model_path),
                        n_gpu_layers=n_gpu_layers,
                        n_ctx=n_ctx,
                        n_batch=n_batch,
                        verbose=verbose,
                        chat_format=chat_format,
                    )
            else:
                self._llm = llama_cpp.Llama(
                    model_path=str(model_path),
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    n_batch=n_batch,
                    verbose=verbose,
                    chat_format=chat_format,
                )
            self._config = config
            self._model_id = config.model_id
            self._loaded = True
            
        except Exception as e:
            raise BackendError(
                f"Failed to load model: {e}",
                backend_name=self._name,
                model_id=config.model_id,
                original_error=e,
            )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
    ) -> BackendResult:
        """Generate text using llama.cpp."""
        if not self._loaded or self._llm is None:
            raise BackendError(
                "Model not loaded",
                backend_name=self._name,
                model_id=self._model_id,
            )
        
        start_time = time.perf_counter()
        
        try:
            logger.debug(f"Calling llama_cpp with prompt len={len(prompt)}, max_tokens={max_tokens}, temp={temperature}")
            
            result = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                stop=stop or [],
                echo=False,
            )
            
            logger.debug(f"llama_cpp raw result keys: {result.keys()}")
            if "choices" in result and result["choices"]:
                logger.debug(f"First choice keys: {result['choices'][0].keys()}")
                logger.debug(f"Finish reason: {result['choices'][0].get('finish_reason')}")
            else:
                logger.error(f"No choices in result: {result}")
            
            text = result["choices"][0]["text"]
            finish_reason = result["choices"][0].get("finish_reason", "stop")
            
            usage = result.get("usage", {})
            tokens_in = usage.get("prompt_tokens", self._estimate_tokens(prompt))
            tokens_out = usage.get("completion_tokens", self._estimate_tokens(text))
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return BackendResult(
                text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model_id=self._model_id,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                extra={"total_tokens": usage.get("total_tokens", tokens_in + tokens_out)},
            )
            
        except Exception as e:
            raise BackendError(
                f"Generation failed: {e}",
                backend_name=self._name,
                model_id=self._model_id,
                original_error=e,
            )
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        
        super().unload()
        gc.collect()
    
    @property
    def context_length(self) -> int:
        """Get the model's context length."""
        if self._llm is not None:
            return self._llm.n_ctx()
        return self._config.context_length if self._config else 4096
