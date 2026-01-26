"""
OpenAI and Ollama backends for API-based inference.
"""

from __future__ import annotations

import logging
import os
import time

from .base import BackendConfig, BackendError, BackendResult, BaseBackend

logger = logging.getLogger(__name__)

# Lazy import
_openai = None


def _get_openai():
    """Lazy import of openai package."""
    global _openai
    if _openai is None:
        try:
            import openai
            _openai = openai
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIBackend. "
                "Install with: pip install openai"
            )
    return _openai


class OpenAIBackend(BaseBackend):
    """Backend for OpenAI API and compatible endpoints."""
    
    _name = "openai"
    
    def __init__(self):
        """Initialize OpenAIBackend."""
        super().__init__()
        self._client = None
    
    def load(self, config: BackendConfig) -> None:
        """Initialize OpenAI client."""
        openai = _get_openai()
        
        extra = config.extra or {}
        
        api_key_env = extra.get("api_key_env", "OPENAI_API_KEY")
        api_key = extra.get("api_key") or os.environ.get(api_key_env)
        
        if not api_key:
            raise BackendError(
                f"OpenAI API key not found. Set {api_key_env} environment variable.",
                backend_name=self._name,
                model_id=config.model_id,
            )
        
        client_kwargs = {
            "api_key": api_key,
            "timeout": extra.get("timeout", config.timeout),
        }
        
        if "base_url" in extra:
            client_kwargs["base_url"] = extra["base_url"]
        
        if "organization" in extra:
            client_kwargs["organization"] = extra["organization"]
        
        try:
            self._client = openai.OpenAI(**client_kwargs)
            self._config = config
            self._model_id = config.model_id
            self._loaded = True
            
        except Exception as e:
            raise BackendError(
                f"Failed to initialize OpenAI client: {e}",
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
        """Generate text using OpenAI API."""
        if not self._loaded or self._client is None:
            raise BackendError(
                "Client not initialized",
                backend_name=self._name,
                model_id=self._model_id,
            )
        
        start_time = time.perf_counter()
        
        try:
            messages = self._parse_prompt_to_messages(prompt)
            
            request_kwargs = {
                "model": self._model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            
            if stop:
                request_kwargs["stop"] = stop
            
            response = self._client.chat.completions.create(**request_kwargs)
            
            choice = response.choices[0]
            text = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"
            
            usage = response.usage
            if usage:
                tokens_in = usage.prompt_tokens
                tokens_out = usage.completion_tokens
            else:
                tokens_in = self._estimate_tokens(prompt)
                tokens_out = self._estimate_tokens(text)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return BackendResult(
                text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model_id=self._model_id,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                extra={"response_id": response.id, "created": response.created},
            )
            
        except Exception as e:
            raise BackendError(
                f"OpenAI API call failed: {e}",
                backend_name=self._name,
                model_id=self._model_id,
                original_error=e,
            )
    
    def _parse_prompt_to_messages(self, prompt: str) -> list[dict[str, str]]:
        """Parse a prompt string into chat messages."""
        messages = []
        
        if "System:" in prompt and "User:" in prompt:
            parts = prompt.split("User:", 1)
            system_part = parts[0].replace("System:", "").strip()
            user_part = parts[1].strip() if len(parts) > 1 else ""
            
            if system_part:
                messages.append({"role": "system", "content": system_part})
            if user_part:
                if "Assistant:" in user_part:
                    user_part = user_part.split("Assistant:")[0].strip()
                messages.append({"role": "user", "content": user_part})
        else:
            messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def unload(self) -> None:
        """Close the client."""
        self._client = None
        super().unload()


class OllamaBackend(OpenAIBackend):
    """Backend for Ollama (via OpenAI compatibility layer)."""
    
    _name = "ollama"
    
    def load(self, config: BackendConfig) -> None:
        """Load Ollama backend with defaults."""
        if config.extra is None:
            config.extra = {}
            
        config.extra.setdefault("base_url", "http://localhost:11434/v1")
        config.extra.setdefault("api_key", "ollama")
        
        if config.model_id.startswith("ollama:"):
            config.model_id = config.model_id.replace("ollama:", "", 1)
            
        super().load(config)
