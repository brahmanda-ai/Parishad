"""
Ollama backend using native API.

Supports both OpenAI-compatible mode (via inherited OllamaBackend in openai_api.py)
and native Ollama API mode (this file).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .base import BackendConfig, BackendError, BackendResult, BaseBackend

logger = logging.getLogger(__name__)

# Lazy import
_requests = None


def _get_requests():
    """Lazy import of requests."""
    global _requests
    if _requests is None:
        try:
            import requests
            _requests = requests
        except ImportError:
            raise ImportError(
                "requests package is required for OllamaNativeBackend. "
                "Install with: pip install requests"
            )
    return _requests


class OllamaNativeBackend(BaseBackend):
    """
    Backend for Ollama using native API.
    
    Uses Ollama's /api/generate endpoint directly instead of OpenAI compatibility layer.
    This provides access to Ollama-specific features like:
    - Raw mode for exact prompts
    - System prompts
    - Streaming
    - Context management
    """
    
    _name = "ollama_native"
    
    def __init__(self):
        """Initialize OllamaNativeBackend."""
        super().__init__()
        self._base_url = "http://localhost:11434"
        self._session = None
    
    def load(self, config: BackendConfig) -> None:
        """Initialize Ollama connection."""
        requests = _get_requests()
        
        extra = config.extra or {}
        self._base_url = extra.get("base_url", "http://localhost:11434")
        
        # Strip ollama: prefix if present
        model_id = config.model_id
        if model_id.startswith("ollama:"):
            model_id = model_id.replace("ollama:", "", 1)
        
        try:
            # Test connection
            self._session = requests.Session()
            response = self._session.get(f"{self._base_url}/api/tags", timeout=5)
            
            if response.status_code != 200:
                raise BackendError(
                    f"Ollama server not responding at {self._base_url}",
                    backend_name=self._name
                )
            
            # Check if model is available
            tags = response.json()
            available_models = [m["name"] for m in tags.get("models", [])]
            
            # Check for exact match or partial match
            model_found = False
            for m in available_models:
                if model_id in m or m in model_id:
                    model_found = True
                    break
            
            if not model_found and available_models:
                logger.warning(
                    f"Model '{model_id}' not found in Ollama. "
                    f"Available: {available_models[:5]}..."
                )
            
            self._config = config
            self._model_id = model_id
            self._loaded = True
            
            logger.info(f"✅ Connected to Ollama at {self._base_url}")
            
        except Exception as e:
            raise BackendError(
                f"Failed to connect to Ollama: {e}",
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
        """Generate text using Ollama native API."""
        if not self._loaded or self._session is None:
            raise BackendError(
                "Backend not loaded",
                backend_name=self._name,
                model_id=self._model_id,
            )
        
        start_time = time.perf_counter()
        
        try:
            # Parse system/user if present in prompt
            system_prompt = ""
            user_prompt = prompt
            
            if "<|start_header_id|>system<|end_header_id|>" in prompt:
                # Llama-3 format - extract parts
                parts = prompt.split("<|start_header_id|>")
                for part in parts:
                    if part.startswith("system"):
                        system_prompt = part.split("<|end_header_id|>")[1].split("<|eot_id|>")[0].strip()
                    elif part.startswith("user"):
                        user_prompt = part.split("<|end_header_id|>")[1].split("<|eot_id|>")[0].strip()
            
            # Build request payload
            payload: dict[str, Any] = {
                "model": self._model_id,
                "prompt": user_prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    # Increase context window for large documents (default 32k)
                    "num_ctx": self._config.extra.get("num_ctx", 32768) if self._config and self._config.extra else 32768,
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if stop:
                payload["options"]["stop"] = stop
            
            # Make request
            response = self._session.post(
                f"{self._base_url}/api/generate",
                json=payload,
                # Increase default timeout to 300s (5m) for slower models/hardware
                timeout=self._config.timeout if self._config and self._config.timeout else 300
            )
            
            if response.status_code != 200:
                raise BackendError(
                    f"Ollama API error: {response.status_code} - {response.text}",
                    backend_name=self._name,
                    model_id=self._model_id
                )
            
            result = response.json()
            
            text = result.get("response", "")
            
            # Get token counts from Ollama response
            tokens_in = result.get("prompt_eval_count", self._estimate_tokens(prompt))
            tokens_out = result.get("eval_count", self._estimate_tokens(text))
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Determine finish reason
            done_reason = result.get("done_reason", "stop")
            if done_reason == "length":
                finish_reason = "length"
            else:
                finish_reason = "stop"
            
            return BackendResult(
                text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model_id=self._model_id,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                extra={
                    "total_duration": result.get("total_duration"),
                    "load_duration": result.get("load_duration"),
                    "eval_duration": result.get("eval_duration"),
                }
            )
            
        except Exception as e:
            if isinstance(e, BackendError):
                raise
            raise BackendError(
                f"Ollama generation failed: {e}",
                backend_name=self._name,
                model_id=self._model_id,
                original_error=e,
            )
    
    def unload(self) -> None:
        """Close the session."""
        if self._session:
            self._session.close()
            self._session = None
        super().unload()
    
    def list_models(self) -> list[str]:
        """List available Ollama models."""
        if not self._session:
            return []
        
        try:
            response = self._session.get(f"{self._base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                tags = response.json()
                return [m["name"] for m in tags.get("models", [])]
        except Exception:
            pass
        return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        if not self._session:
            return False
        
        try:
            response = self._session.post(
                f"{self._base_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=600  # Models can take a while to download
            )
            return response.status_code == 200
        except Exception:
            return False
