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
        """Load a GGUF model with comprehensive logging and VRAM checking."""
        import time
        start_time = time.perf_counter()
        
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"🚀 LLAMA.CPP MODEL LOADING STARTED")
        logger.info(f"{'='*80}")
        logger.info(f"Model ID: {config.model_id}")
        logger.info(f"Context Length: {config.context_length}")
        
        llama_cpp = _get_llama_cpp()
        logger.info(f"✅ llama-cpp-python loaded successfully")
        
        # Resolve model path
        logger.info(f"🔍 Resolving model path for: {config.model_id}")
        model_path = resolve_model_path(config.model_id)
        
        if model_path is None:
            logger.error(f"❌ Model not found: {config.model_id}")
            raise BackendError(
                f"Model not found: {config.model_id}. "
                "Download with: parishad download <model_name>",
                backend_name=self._name,
                model_id=config.model_id,
            )
        
        logger.info(f"✅ Model path resolved: {model_path}")
        model_size_mb = model_path.stat().st_size / (1024**2)
        logger.info(f"   File size: {model_size_mb:.2f} MB")
        
       # VRAM compatibility check
        try:
            from ..memory_estimation import check_model_compatibility, detect_vram
            
            logger.info(f"")
            logger.info(f"🔍 Checking GPU/VRAM compatibility...")
            vram_info = detect_vram()
            
            if not vram_info.is_gpu:
                logger.error(f"❌ No GPU detected! This project requires GPU.")
                logger.error(f"   GPU Type: {vram_info.gpu_type}")
                logger.error(f"   Please install CUDA drivers and PyTorch with CUDA support.")
            
            is_compatible, message, mem_req = check_model_compatibility(
                config.model_id,
                vram_info=vram_info,
                context_length=config.context_length
            )
            
            if not is_compatible and not vram_info.is_gpu:
                raise BackendError(
                    message,
                    backend_name=self._name,
                    model_id=config.model_id,
                )
            elif not is_compatible:
                logger.warning(f"⚠️  VRAM WARNING: {message}")
                logger.warning(f"   Attempting to load anyway, but may fail...")
        except ImportError as e:
            logger.warning(f"⚠️  Could not check VRAM compatibility: {e}")
        except Exception as e:
            logger.warning(f"⚠️  VRAM check failed: {e}")
        
        extra = config.extra or {}
        n_gpu_layers = extra.get("n_gpu_layers", -1)
        n_ctx = extra.get("n_ctx", config.context_length)
        n_batch = extra.get("n_batch", 512)
        verbose = extra.get("verbose", False)
        chat_format = extra.get("chat_format", None)
        
        logger.info(f"")
        logger.info(f"📋 Loading configuration:")
        logger.info(f"   GPU Layers: {n_gpu_layers} (-1 = all layers on GPU)")
        logger.info(f"   Context Size: {n_ctx}")
        logger.info(f"   Batch Size: {n_batch}")
        logger.info(f"   Verbose: {verbose}")
        logger.info(f"   Chat Format: {chat_format or 'auto'}")
        
        try:
            logger.info(f"")
            logger.info(f"⏳ Loading model into memory... (this may take 30-90 seconds)")
            load_start = time.perf_counter()
            
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
            
            load_duration = time.perf_counter() - load_start
            logger.info(f"✅ Model loaded successfully in {load_duration:.2f}s")
            
            self._config = config
            self._model_id = config.model_id
            self._loaded = True
            
            total_duration = time.perf_counter() - start_time
            logger.info(f"")
            logger.info(f"{'='*80}")
            logger.info(f"✅ LLAMA.CPP LOADING COMPLETE - Total time: {total_duration:.2f}s")
            logger.info(f"{'='*80}")
            logger.info(f"")
            
        except Exception as e:
            load_duration = time.perf_counter() - start_time
            logger.error(f"")
            logger.error(f"{'='*80}")
            logger.error(f"❌ LLAMA.CPP LOADING FAILED after {load_duration:.2f}s")
            logger.error(f"{'='*80}")
            logger.error(f"Model: {config.model_id}")
            logger.error(f"Path: {model_path}")
            logger.error(f"Error: {type(e).__name__}: {e}")
            logger.error(f"")
            
            # Provide helpful error messages
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                logger.error(f"💡 This appears to be a VRAM/memory issue.")
                logger.error(f"   Try a smaller model or more aggressive quantization.")
            elif "file" in error_msg or "not found" in error_msg:
                logger.error(f"💡 Model file issue. Try re-downloading the model.")
            
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
        """Generate text using llama.cpp with comprehensive logging."""
        if not self._loaded or self._llm is None:
            logger.error(f"❌ Cannot generate: Model not loaded")
            raise BackendError(
                "Model not loaded",
                backend_name=self._name,
                model_id=self._model_id,
            )
        
        start_time = time.perf_counter()
        
        logger.info(f"")
        logger.info(f"{'─'*80}")
        logger.info(f"🎯 GENERATION STARTED")
        logger.info(f"{'─'*80}")
        logger.info(f"Model: {self._model_id}")
        logger.info(f"Prompt length: {len(prompt)} chars")
        logger.info(f"Max tokens: {max_tokens}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Top-p: {top_p}")
        logger.info(f"Stop sequences: {stop or 'none'}")
        
        try:
            logger.info(f"⏳ Generating response...")
            gen_start = time.perf_counter()
            
            result = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                stop=stop or [],
                echo=False,
            )
            
            gen_duration = time.perf_counter() - gen_start
            
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
            total_tokens = tokens_in + tokens_out
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            tokens_per_sec = tokens_out / gen_duration if gen_duration > 0 else 0
            
            logger.info(f"✅ Generation complete in {gen_duration:.2f}s")
            logger.info(f"   Tokens: {tokens_in} in + {tokens_out} out = {total_tokens} total")
            logger.info(f"   Speed: {tokens_per_sec:.1f} tokens/sec")
            logger.info(f"   Finish reason: {finish_reason}")
            logger.info(f"   Response length: {len(text)} chars")
            logger.info(f"   Response preview: {text[:100]}...")
            logger.info(f"{'─'*80}")
            logger.info(f"")
            
            return BackendResult(
                text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model_id=self._model_id,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                extra={
                    "total_tokens": total_tokens,
                    "tokens_per_sec": tokens_per_sec,
                    "generation_time_sec": gen_duration
                },
            )
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"")
            logger.error(f"{'─'*80}")
            logger.error(f"❌ GENERATION FAILED after {duration:.2f}s")
            logger.error(f"{'─'*80}")
            logger.error(f"Error: {type(e).__name__}: {e}")
            logger.error(f"")
            
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
