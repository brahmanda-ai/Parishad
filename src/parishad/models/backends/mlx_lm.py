"""
MLX backend for Apple Silicon.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .base import BackendConfig, BackendError, BackendResult, BaseBackend

logger = logging.getLogger(__name__)

# Lazy imports
_mlx_lm = None


def _get_mlx_lm():
    """Lazy import of mlx-lm."""
    global _mlx_lm
    if _mlx_lm is None:
        try:
            import mlx_lm
            _mlx_lm = mlx_lm
        except ImportError:
            raise ImportError(
                "mlx-lm is required for MlxBackend. "
                "Install with: pip install mlx-lm"
            )
    return _mlx_lm


class MlxBackend(BaseBackend):
    """Backend for MLX models on Apple Silicon."""
    
    _name = "mlx"
    
    def __init__(self):
        """Initialize MlxBackend."""
        super().__init__()
        self._model = None
        self._tokenizer = None
    
    def load(self, config: BackendConfig) -> None:
        """Load an MLX model."""
        mlx_lm = _get_mlx_lm()
        
        model_id_or_path = config.model_id
        
        p = Path(model_id_or_path)
        if p.exists():
            if p.is_file():
                model_id_or_path = str(p.parent)
            else:
                model_id_or_path = str(p)
                
        try:
            self._model, self._tokenizer = mlx_lm.load(
                model_id_or_path,
                tokenizer_config={"trust_remote_code": True}
            )
            
            self._config = config
            self._model_id = config.model_id
            self._loaded = True
            
        except Exception as e:
            raise BackendError(
                f"Failed to load MLX model: {e}",
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
        """Generate text using MLX."""
        if not self._loaded or self._model is None:
            raise BackendError(
                "Model not loaded",
                backend_name=self._name,
                model_id=self._model_id,
            )
            
        mlx_lm = _get_mlx_lm()
        start_time = time.perf_counter()
        
        try:
            tokens_in = len(self._tokenizer.encode(prompt))
            
            text = mlx_lm.generate(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            
            finish_reason = "length"
            if stop:
                for s in stop:
                    if s in text:
                        text = text.split(s)[0]
                        finish_reason = "stop"
                        break
            
            tokens_out = len(self._tokenizer.encode(text))
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return BackendResult(
                text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model_id=self._model_id,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            raise BackendError(
                f"MLX generation failed: {e}",
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
        super().unload()
