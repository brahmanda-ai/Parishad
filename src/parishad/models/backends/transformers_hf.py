"""
HuggingFace Transformers backend.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .base import BackendConfig, BackendError, BackendResult, BaseBackend

logger = logging.getLogger(__name__)

# Lazy imports
_transformers = None
_torch = None


def _get_transformers():
    """Lazy import of transformers and torch."""
    global _transformers, _torch
    if _transformers is None:
        try:
            import transformers
            import torch
            _transformers = transformers
            _torch = torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for TransformersBackend. "
                "Install with: pip install transformers torch"
            )
    return _transformers, _torch


class TransformersBackend(BaseBackend):
    """Backend for HuggingFace Transformers models."""
    
    _name = "transformers"
    
    def __init__(self):
        """Initialize TransformersBackend."""
        super().__init__()
        self._model = None
        self._tokenizer = None
    
    def load(self, config: BackendConfig) -> None:
        """Load a Transformers model with comprehensive logging and VRAM checking."""
        import time
        start_time = time.perf_counter()
        
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"🚀 TRANSFORMERS MODEL LOADING STARTED")
        logger.info(f"{'='*80}")
        logger.info(f"Model ID: {config.model_id}")
        logger.info(f"Context Length: {config.context_length}")
        
        transformers, torch = _get_transformers()
        logger.info(f"✅ transformers and torch loaded successfully")
        logger.info(f"   PyTorch version: {torch.__version__}")
        logger.info(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"   CUDA version: {torch.version.cuda}")
            logger.info(f"   GPU count: {torch.cuda.device_count()}")
        
        # VRAM compatibility check
        try:
            from ..memory_estimation import check_model_compatibility, detect_vram
            
            logger.info(f"")
            logger.info(f"🔍 Checking GPU/VRAM compatibility...")
            vram_info = detect_vram()
            
            if not vram_info.is_gpu:
                logger.error(f"❌ No GPU detected! This project requires GPU.")
                logger.error(f"   GPU Type: {vram_info.gpu_type}")
                logger.error(f"   For transformers backend, CUDA GPU is required.")
            
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
        
        model_id_or_path = config.model_id
        p = Path(model_id_or_path)
        if p.exists() and p.is_file():
            model_id_or_path = str(p.parent)
            logger.info(f"Loading from local path: {model_id_or_path}")
        
        logger.info(f"")
        logger.info(f"📋 Loading configuration:")
        device_map = extra.get("device_map", "cuda")
        logger.info(f"   Device map: {device_map}")
        
        dtype_str = extra.get("torch_dtype", "float16")
        logger.info(f"   Data type: {dtype_str}")
        
        quantization = extra.get("quantization")
        if quantization:
            logger.info(f"   Quantization: {quantization}")
        
        trust_remote_code = extra.get("trust_remote_code", False)
        logger.info(f"   Trust remote code: {trust_remote_code}")
            
        try:
            logger.info(f"")
            logger.info(f"⏳ Loading tokenizer...")
            tokenizer_start = time.perf_counter()
            
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id_or_path,
                trust_remote_code=trust_remote_code,
            )
            
            tokenizer_duration = time.perf_counter() - tokenizer_start
            logger.info(f"✅ Tokenizer loaded in {tokenizer_duration:.2f}s")
            logger.info(f"   Vocab size: {len(self._tokenizer)}")
            
            model_kwargs = {
                "device_map": device_map,
                "trust_remote_code": trust_remote_code,
            }
            
            if dtype_str == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif dtype_str == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif dtype_str == "float32":
                model_kwargs["torch_dtype"] = torch.float32
            
            if quantization in ("4bit", "8bit"):
                try:
                    from transformers import BitsAndBytesConfig
                    
                    logger.info(f"🔧 Configuring {quantization} quantization...")
                    
                    if quantization == "4bit":
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            llm_int8_enable_fp32_cpu_offload=False,  # GPU-only
                        )
                        logger.info(f"   4-bit quantization configured (GPU-only)")
                    else:
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_enable_fp32_cpu_offload=False,  # GPU-only
                        )
                        logger.info(f"   8-bit quantization configured (GPU-only)")
                except ImportError:
                    logger.warning(f"⚠️  bitsandbytes not available, ignoring quantization")
            
            logger.info(f"")
            logger.info(f"⏳ Loading model into GPU... (this may take 1-3 minutes)")
            model_start = time.perf_counter()
            
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                **model_kwargs,
            )
            
            model_duration = time.perf_counter() - model_start
            logger.info(f"✅ Model loaded successfully in {model_duration:.2f}s")
            
            # Log model info
            param_count = sum(p.numel() for p in self._model.parameters())
            param_billion = param_count / 1e9
            logger.info(f"   Parameters: {param_billion:.2f}B")
            logger.info(f"   Device: {self._model.device}")
            
            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"   GPU memory allocated: {allocated_gb:.2f} GB")
                logger.info(f"   GPU memory reserved: {reserved_gb:.2f} GB")
            
            self._config = config
            self._model_id = config.model_id
            self._loaded = True
            
            total_duration = time.perf_counter() - start_time
            logger.info(f"")
            logger.info(f"{'='*80}")
            logger.info(f"✅ TRANSFORMERS LOADING COMPLETE - Total time: {total_duration:.2f}s")
            logger.info(f"{'='*80}")
            logger.info(f"")
            
        except Exception as e:
            load_duration = time.perf_counter() - start_time
            logger.error(f"")
            logger.error(f"{'='*80}")
            logger.error(f"❌ TRANSFORMERS LOADING FAILED after {load_duration:.2f}s")
            logger.error(f"{'='*80}")
            logger.error(f"Model: {config.model_id}")
            logger.error(f"Error: {type(e).__name__}: {e}")
            logger.error(f"")
            
            # Provide helpful error messages
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                logger.error(f"💡 This appears to be a VRAM/memory issue.")
                logger.error(f"   Try a smaller model or more aggressive quantization.")
                logger.error(f"   Also try closing other GPU-intensive applications.")
            elif "connection" in error_msg or "download" in error_msg:
                logger.error(f"💡 Network issue downloading model. Check internet connection.")
            
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
        """Generate text using Transformers with comprehensive logging."""
        if not self._loaded or self._model is None or self._tokenizer is None:
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
            logger.info(f"⏳ Tokenizing input...")
            tokenize_start = time.perf_counter()
            
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(self._model.device)
            
            input_len = inputs.input_ids.shape[1]
            tokenize_duration = time.perf_counter() - tokenize_start
            
            logger.info(f"✅ Tokenized in {tokenize_duration:.3f}s")
            logger.info(f"   Input tokens: {input_len}")
            
            logger.info(f"⏳ Generating response...")
            gen_start = time.perf_counter()
            
            with _torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            
            gen_duration = time.perf_counter() - gen_start
            
            generated_ids = outputs[0][input_len:]
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            tokens_out = len(generated_ids)
            tokens_per_sec = tokens_out / gen_duration if gen_duration > 0 else 0
            
            if "cuda" in str(self._model.device):
                _torch.cuda.empty_cache()
            
            finish_reason = "length" if len(generated_ids) >= max_tokens else "stop"
            if stop:
                for s in stop:
                    if s in text:
                        text = text.split(s)[0]
                        finish_reason = "stop"
                        break
            
            logger.info(f"✅ Generation complete in {gen_duration:.2f}s")
            logger.info(f"   Tokens: {input_len} in + {tokens_out} out = {input_len + tokens_out} total")
            logger.info(f"   Speed: {tokens_per_sec:.1f} tokens/sec")
            logger.info(f"   Finish reason: {finish_reason}")
            logger.info(f"   Response length: {len(text)} chars")
            logger.info(f"   Response preview: {text[:100]}...")
            logger.info(f"{'─'*80}")
            logger.info(f"")
            
            return BackendResult(
                text=text,
                tokens_in=input_len,
                tokens_out=tokens_out,
                model_id=self._model_id,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                extra={
                    "total_tokens": input_len + tokens_out,
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
            
            raise BackendError(
                f"Generation failed: {e}",
                backend_name=self._name,
                model_id=self._model_id,
                original_error=e,
            )
    
    def unload(self) -> None:
        """Unload model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        if _torch is not None:
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            elif _torch.backends.mps.is_available():
                _torch.mps.empty_cache()
                
        super().unload()
