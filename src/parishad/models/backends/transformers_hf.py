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
        """Load a Transformers model."""
        transformers, torch = _get_transformers()
        
        extra = config.extra or {}
        
        model_id_or_path = config.model_id
        p = Path(model_id_or_path)
        if p.exists() and p.is_file():
            model_id_or_path = str(p.parent)
            
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id_or_path,
                trust_remote_code=extra.get("trust_remote_code", False),
            )
            
            model_kwargs = {
                "device_map": extra.get("device_map", "auto"),
                "trust_remote_code": extra.get("trust_remote_code", False),
            }
            
            dtype_str = extra.get("torch_dtype", "float16")
            if dtype_str == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif dtype_str == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif dtype_str == "float32":
                model_kwargs["torch_dtype"] = torch.float32
            
            quantization = extra.get("quantization")
            if quantization in ("4bit", "8bit"):
                try:
                    from transformers import BitsAndBytesConfig
                    
                    cpu_offload = extra.get("cpu_offload", False)
                    
                    if quantization == "4bit":
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            llm_int8_enable_fp32_cpu_offload=cpu_offload,
                        )
                    else:
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_enable_fp32_cpu_offload=cpu_offload,
                        )
                except ImportError:
                    pass
            
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                **model_kwargs,
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
        """Generate text using Transformers."""
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise BackendError(
                "Model not loaded",
                backend_name=self._name,
                model_id=self._model_id,
            )
        
        start_time = time.perf_counter()
        
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(self._model.device)
            
            input_len = inputs.input_ids.shape[1]
            
            with _torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            
            generated_ids = outputs[0][input_len:]
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if "cuda" in str(self._model.device):
                _torch.cuda.empty_cache()
            
            finish_reason = "length" if len(generated_ids) >= max_tokens else "stop"
            if stop:
                for s in stop:
                    if s in text:
                        text = text.split(s)[0]
                        finish_reason = "stop"
                        break
            
            return BackendResult(
                text=text,
                tokens_in=input_len,
                tokens_out=len(generated_ids),
                model_id=self._model_id,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
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
