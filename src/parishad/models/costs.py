"""
Cost estimation for Parishad model inference.

Provides cost estimation for different backends and models:
- API-based models (OpenAI, Anthropic) have per-token costs
- Local models (llama.cpp, transformers) are "free" (compute cost only)

This module tracks:
- Dollar cost estimates for API models
- Token usage metrics
- Approximate FLOP estimates for local models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Cost Data
# =============================================================================


@dataclass
class ModelPricing:
    """
    Pricing information for a model.
    
    Prices are in USD per 1M tokens.
    """
    
    input_price: float = 0.0
    """Price per 1M input tokens in USD."""
    
    output_price: float = 0.0
    """Price per 1M output tokens in USD."""
    
    name: str = ""
    """Model name for display."""
    
    def cost_for_tokens(self, tokens_in: int, tokens_out: int) -> float:
        """
        Calculate cost for a given token usage.
        
        Args:
            tokens_in: Number of input tokens
            tokens_out: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        input_cost = (tokens_in / 1_000_000) * self.input_price
        output_cost = (tokens_out / 1_000_000) * self.output_price
        return input_cost + output_cost


# Default pricing for known models (as of Dec 2024)
# Prices in USD per 1M tokens
MODEL_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(input_price=2.50, output_price=10.00, name="GPT-4o"),
    "gpt-4o-mini": ModelPricing(input_price=0.15, output_price=0.60, name="GPT-4o Mini"),
    "gpt-4-turbo": ModelPricing(input_price=10.00, output_price=30.00, name="GPT-4 Turbo"),
    "gpt-4": ModelPricing(input_price=30.00, output_price=60.00, name="GPT-4"),
    "gpt-3.5-turbo": ModelPricing(input_price=0.50, output_price=1.50, name="GPT-3.5 Turbo"),
    
    # Anthropic
    "claude-3-5-sonnet-20241022": ModelPricing(input_price=3.00, output_price=15.00, name="Claude 3.5 Sonnet"),
    "claude-3-opus-20240229": ModelPricing(input_price=15.00, output_price=75.00, name="Claude 3 Opus"),
    "claude-3-haiku-20240307": ModelPricing(input_price=0.25, output_price=1.25, name="Claude 3 Haiku"),
    
    # Local models (free)
    "stub": ModelPricing(input_price=0.0, output_price=0.0, name="Stub"),
    "mock": ModelPricing(input_price=0.0, output_price=0.0, name="Mock"),
}

# Backend-level defaults (for unknown models)
BACKEND_DEFAULT_PRICING: dict[str, ModelPricing] = {
    "openai": ModelPricing(input_price=0.50, output_price=1.50, name="OpenAI Default"),
    "anthropic": ModelPricing(input_price=3.00, output_price=15.00, name="Anthropic Default"),
    "stub": ModelPricing(input_price=0.0, output_price=0.0, name="Stub"),
    "mock": ModelPricing(input_price=0.0, output_price=0.0, name="Mock"),
    "llama_cpp": ModelPricing(input_price=0.0, output_price=0.0, name="Local"),
    "transformers": ModelPricing(input_price=0.0, output_price=0.0, name="Local"),
}


def get_model_pricing(model_id: str, backend: str = "") -> ModelPricing:
    """
    Get pricing for a model.
    
    Looks up in order:
    1. Exact model_id match
    2. Backend default
    3. Zero pricing (local/unknown)
    
    Args:
        model_id: Model identifier
        backend: Optional backend name
        
    Returns:
        ModelPricing instance
    """
    # Check exact model match
    if model_id in MODEL_PRICING:
        return MODEL_PRICING[model_id]
    
    # Check backend default
    if backend in BACKEND_DEFAULT_PRICING:
        return BACKEND_DEFAULT_PRICING[backend]
    
    # Default to free (local models)
    return ModelPricing(name=model_id or "Unknown")


# =============================================================================
# Cost Tracking
# =============================================================================


@dataclass
class CostMetrics:
    """
    Accumulated cost metrics for a session or query.
    """
    
    total_tokens_in: int = 0
    """Total input tokens used."""
    
    total_tokens_out: int = 0
    """Total output tokens generated."""
    
    total_cost_usd: float = 0.0
    """Total estimated cost in USD."""
    
    calls: int = 0
    """Number of API/inference calls."""
    
    latency_ms: float = 0.0
    """Total latency in milliseconds."""
    
    # Per-slot breakdown
    slot_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Metrics broken down by slot."""
    
    def add_call(
        self,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        latency_ms: float = 0.0,
        slot: str = "",
    ) -> None:
        """
        Add metrics from a call.
        
        Args:
            tokens_in: Input tokens for this call
            tokens_out: Output tokens for this call
            cost_usd: Cost in USD for this call
            latency_ms: Latency in milliseconds
            slot: Optional slot name (small/mid/big)
        """
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.total_cost_usd += cost_usd
        self.calls += 1
        self.latency_ms += latency_ms
        
        if slot:
            if slot not in self.slot_metrics:
                self.slot_metrics[slot] = {
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                    "calls": 0,
                }
            self.slot_metrics[slot]["tokens_in"] += tokens_in
            self.slot_metrics[slot]["tokens_out"] += tokens_out
            self.slot_metrics[slot]["cost_usd"] += cost_usd
            self.slot_metrics[slot]["calls"] += 1
    
    @property
    def total_tokens(self) -> int:
        """Total tokens (in + out)."""
        return self.total_tokens_in + self.total_tokens_out
    
    @property
    def avg_latency_ms(self) -> float:
        """Average latency per call."""
        if self.calls == 0:
            return 0.0
        return self.latency_ms / self.calls
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "calls": self.calls,
            "total_latency_ms": round(self.latency_ms, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "slot_metrics": self.slot_metrics,
        }


# =============================================================================
# Cost Estimation Functions
# =============================================================================


def estimate_cost(
    backend: str,
    model_id: str,
    tokens_in: int,
    tokens_out: int,
) -> float:
    """
    Estimate cost for a generation call.
    
    Args:
        backend: Backend name (e.g., 'openai', 'llama_cpp')
        model_id: Model identifier
        tokens_in: Number of input tokens
        tokens_out: Number of output tokens
        
    Returns:
        Estimated cost in USD (0.0 for local models)
    """
    pricing = get_model_pricing(model_id, backend)
    return pricing.cost_for_tokens(tokens_in, tokens_out)


def estimate_query_cost(
    backend: str,
    model_id: str,
    prompt_tokens: int,
    estimated_output_tokens: int = 500,
) -> float:
    """
    Estimate cost for a query before running it.
    
    Useful for budget checks before making API calls.
    
    Args:
        backend: Backend name
        model_id: Model identifier
        prompt_tokens: Number of input tokens
        estimated_output_tokens: Expected output tokens
        
    Returns:
        Estimated cost in USD
    """
    return estimate_cost(backend, model_id, prompt_tokens, estimated_output_tokens)


# =============================================================================
# FLOP Estimation (for local models)
# =============================================================================


def estimate_flops(
    model_params_billions: float,
    tokens: int,
    is_generation: bool = True,
) -> float:
    """
    Estimate FLOPs for local model inference.
    
    Uses the approximation: FLOPs ≈ 2 * params * tokens
    For generation, each token requires a forward pass.
    
    Args:
        model_params_billions: Model size in billions of parameters
        tokens: Number of tokens processed/generated
        is_generation: Whether this is generation (vs prefill)
        
    Returns:
        Estimated FLOPs
    """
    params = model_params_billions * 1e9
    
    if is_generation:
        # Each generated token requires ~2 * params FLOPs
        return 2 * params * tokens
    else:
        # Prefill is more efficient (parallelized)
        return 2 * params * tokens


def estimate_local_inference_time_ms(
    model_params_billions: float,
    tokens_in: int,
    tokens_out: int,
    tflops: float = 100.0,
) -> float:
    """
    Estimate inference time for local models.
    
    Args:
        model_params_billions: Model size in billions
        tokens_in: Input tokens (prefill)
        tokens_out: Output tokens (generation)
        tflops: Hardware capability in TFLOPS
        
    Returns:
        Estimated time in milliseconds
    """
    # Prefill FLOPs
    prefill_flops = estimate_flops(model_params_billions, tokens_in, is_generation=False)
    
    # Generation FLOPs
    gen_flops = estimate_flops(model_params_billions, tokens_out, is_generation=True)
    
    total_flops = prefill_flops + gen_flops
    
    # Convert TFLOPS to FLOPS
    flops_per_second = tflops * 1e12
    
    # Time in seconds, then milliseconds
    time_seconds = total_flops / flops_per_second
    return time_seconds * 1000


# =============================================================================
# Model Size Registry
# =============================================================================


# Known model sizes in billions of parameters
MODEL_SIZES: dict[str, float] = {
    # Qwen
    "qwen2.5-0.5b": 0.5,
    "qwen2.5-1.5b": 1.5,
    "qwen2.5-3b": 3.0,
    "qwen2.5-7b": 7.0,
    "qwen2.5-14b": 14.0,
    "qwen2.5-32b": 32.0,
    "qwen2.5-72b": 72.0,
    
    # Llama
    "llama-3.2-1b": 1.0,
    "llama-3.2-3b": 3.0,
    "llama-3.1-8b": 8.0,
    "llama-3.1-70b": 70.0,
    
    # Mistral
    "mistral-7b": 7.0,
    "mixtral-8x7b": 47.0,  # Sparse
    
    # Stubs
    "stub": 0.0,
    "mock": 0.0,
}


def get_model_size(model_id: str) -> float:
    """
    Get model size in billions of parameters.
    
    Tries to parse from model_id if not in registry.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Model size in billions (0.0 if unknown)
    """
    model_lower = model_id.lower()
    
    # Check registry
    for key, size in MODEL_SIZES.items():
        if key in model_lower:
            return size
    
    # Try to parse from name (e.g., "7b", "14b")
    import re
    match = re.search(r'(\d+(?:\.\d+)?)\s*b', model_lower)
    if match:
        return float(match.group(1))
    
    return 0.0
