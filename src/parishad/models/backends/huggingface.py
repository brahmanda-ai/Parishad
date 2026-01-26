"""
HuggingFace backends for inference.

Provides:
- HuggingFaceBackend: Uses HuggingFace Inference API (cloud)
- HuggingFaceLocalBackend: Uses local transformers (alias for TransformersBackend)
"""

from __future__ import annotations

import logging
import os
import time

from .base import BackendConfig, BackendError, BackendResult, BaseBackend

logger = logging.getLogger(__name__)

# Lazy imports
_huggingface_hub = None


def _get_huggingface_hub():
    """Lazy import of huggingface_hub."""
    global _huggingface_hub
    if _huggingface_hub is None:
        try:
            import huggingface_hub
            _huggingface_hub = huggingface_hub
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for HuggingFaceBackend. "
                "Install with: pip install huggingface_hub"
            )
    return _huggingface_hub


class HuggingFaceBackend(BaseBackend):
    """
    Backend for HuggingFace Inference API (cloud-based).
    
    Uses HuggingFace's serverless Inference API or dedicated Inference Endpoints.
    Requires HF_TOKEN environment variable or token in config.
    """
    
    _name = "huggingface"
    
    def __init__(self):
        """Initialize HuggingFaceBackend."""
        super().__init__()
        self._client = None
    
    def load(self, config: BackendConfig) -> None:
        """Initialize HuggingFace Inference client."""
        hf = _get_huggingface_hub()
        
        extra = config.extra or {}
        
        # Get token
        token = extra.get("token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        if not token:
            logger.warning(
                "No HuggingFace token found. Some models may not be accessible. "
                "Set HF_TOKEN environment variable."
            )
        
        try:
            # Check if it's an Inference Endpoint URL or model ID
            model_id = config.model_id
            
            if model_id.startswith("https://"):
                # Dedicated Inference Endpoint
                self._client = hf.InferenceClient(
                    model=model_id,
                    token=token,
                    timeout=config.timeout,
                )
                logger.info(f"✅ Connected to HuggingFace Inference Endpoint")
            else:
                # Serverless Inference API
                self._client = hf.InferenceClient(
                    model=model_id,
                    token=token,
                    timeout=config.timeout,
                )
                logger.info(f"✅ Using HuggingFace Serverless API for {model_id}")
            
            self._config = config
            self._model_id = model_id
            self._loaded = True
            
        except Exception as e:
            raise BackendError(
                f"Failed to initialize HuggingFace client: {e}",
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
        """Generate text using HuggingFace Inference API."""
        if not self._loaded or self._client is None:
            raise BackendError(
                "Client not initialized",
                backend_name=self._name,
                model_id=self._model_id,
            )
        
        start_time = time.perf_counter()
        
        try:
            # Use text_generation for LLMs
            response = self._client.text_generation(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                stop_sequences=stop or [],
                return_full_text=False,  # Only return generated text
                details=True,  # Get token counts
            )
            
            # Extract text and details
            if hasattr(response, 'generated_text'):
                text = response.generated_text
                tokens_out = response.details.generated_tokens if hasattr(response, 'details') else self._estimate_tokens(text)
                finish_reason = response.details.finish_reason if hasattr(response, 'details') else "stop"
            else:
                # Simple string response
                text = str(response)
                tokens_out = self._estimate_tokens(text)
                finish_reason = "stop"
            
            tokens_in = self._estimate_tokens(prompt)
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
            # Check for common HF API errors
            error_msg = str(e)
            if "429" in error_msg:
                error_msg = f"Rate limited by HuggingFace API. Try again later. {e}"
            elif "401" in error_msg or "403" in error_msg:
                error_msg = f"Authentication failed. Check your HF_TOKEN. {e}"
            elif "Model is currently loading" in error_msg:
                error_msg = f"Model is loading on HuggingFace servers. Retry in ~30s. {e}"
            
            raise BackendError(
                f"HuggingFace generation failed: {error_msg}",
                backend_name=self._name,
                model_id=self._model_id,
                original_error=e,
            )
    
    def unload(self) -> None:
        """Close the client."""
        self._client = None
        super().unload()


class HuggingFaceChatBackend(BaseBackend):
    """
    Backend for HuggingFace Inference API with chat/conversation support.
    
    Uses the chat_completion endpoint for models that support it.
    """
    
    _name = "huggingface_chat"
    
    def __init__(self):
        """Initialize HuggingFaceChatBackend."""
        super().__init__()
        self._client = None
    
    def load(self, config: BackendConfig) -> None:
        """Initialize HuggingFace Inference client."""
        hf = _get_huggingface_hub()
        
        extra = config.extra or {}
        token = extra.get("token") or os.environ.get("HF_TOKEN")
        
        try:
            self._client = hf.InferenceClient(
                model=config.model_id,
                token=token,
                timeout=config.timeout,
            )
            
            self._config = config
            self._model_id = config.model_id
            self._loaded = True
            
        except Exception as e:
            raise BackendError(
                f"Failed to initialize HuggingFace chat client: {e}",
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
        """Generate text using HuggingFace chat completion."""
        if not self._loaded or self._client is None:
            raise BackendError(
                "Client not initialized",
                backend_name=self._name,
                model_id=self._model_id,
            )
        
        start_time = time.perf_counter()
        
        try:
            # Parse prompt into messages
            messages = self._parse_prompt_to_messages(prompt)
            
            response = self._client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                stop=stop,
            )
            
            # Extract response
            choice = response.choices[0]
            text = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"
            
            # Token counts
            if hasattr(response, 'usage') and response.usage:
                tokens_in = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens
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
            )
            
        except Exception as e:
            raise BackendError(
                f"HuggingFace chat generation failed: {e}",
                backend_name=self._name,
                model_id=self._model_id,
                original_error=e,
            )
    
    def _parse_prompt_to_messages(self, prompt: str) -> list[dict]:
        """Parse prompt string into message format."""
        messages = []
        
        # Try to parse Llama-3 format
        if "<|start_header_id|>" in prompt:
            parts = prompt.split("<|start_header_id|>")
            for part in parts:
                if part.startswith("system"):
                    content = part.split("<|end_header_id|>")[1].split("<|eot_id|>")[0].strip()
                    if content:
                        messages.append({"role": "system", "content": content})
                elif part.startswith("user"):
                    content = part.split("<|end_header_id|>")[1].split("<|eot_id|>")[0].strip()
                    if content:
                        messages.append({"role": "user", "content": content})
                elif part.startswith("assistant"):
                    content = part.split("<|end_header_id|>")[1].split("<|eot_id|>")[0].strip()
                    if content:
                        messages.append({"role": "assistant", "content": content})
        elif "System:" in prompt and "User:" in prompt:
            # Simple format
            parts = prompt.split("User:", 1)
            system = parts[0].replace("System:", "").strip()
            user = parts[1].strip() if len(parts) > 1 else ""
            
            if system:
                messages.append({"role": "system", "content": system})
            if user:
                messages.append({"role": "user", "content": user})
        else:
            # Single user message
            messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def unload(self) -> None:
        """Close client."""
        self._client = None
        super().unload()
