"""
Model Catalog for Parishad Marketplace.
Defines recommended models and sabha configurations.
"""
from dataclasses import dataclass, field
from typing import Literal, List, Dict

@dataclass
class ModelEntry:
    """A recommended model."""
    name: str  # Display name
    backend: Literal["llama_cpp", "mlx", "transformers"] # Format-aware backends: llama_cpp for GGUF, mlx for MLX, transformers for Safetensors
    model_id: str  # The ID used by the backend or the lookup key
    min_ram_gb: int
    description: str
    format: Literal["gguf", "mlx", "safetensors"] = "gguf"  # Model format
    hw_tags: List[str] = field(default_factory=list) # e.g. ["cpu", "cuda", "mlx", "metal"]
    download_info: Dict[str, str] = field(default_factory=dict) # Hints for downloader: repo_id, filename

@dataclass
class SabhaConfig:
    """Configuration for a Sabha tier."""
    name: str
    roles: List[str]
    description: str
    min_tokens_req: int 

# --- SABHA DEFINITIONS ---
SABHAS = {
    "laghu": SabhaConfig(
        name="Laghu Sabha",
        description="A concise council of 5 core roles. Fast and efficient.",
        roles=["raja", "dandadhyaksha", "sacheev", "prerak", "sainik"],
        min_tokens_req=4096
    ),
    "mantri": SabhaConfig(
        name="Mantri Parishad",
        description="Expanded council with 8 roles for better planning.",
        roles=[
            "raja", "dandadhyaksha", "sacheev", "prerak", "majumdar",
            "pantapradhan", "darbari", "sainik"
        ],
        min_tokens_req=8192
    ),
    "maha": SabhaConfig(
        name="Maha Sabha",
        description="The full royal court of 10 roles for maximum capability.",
        roles=[
            "raja", "dandadhyaksha", "sacheev", "prerak", "majumdar",
            "pantapradhan", "darbari", "sar_senapati", "guptachar", "sainik"
        ],
        min_tokens_req=16384
    )
}

# --- MODEL RECOMMENDATIONS ---

# Note: We now support three formats: GGUF (llama_cpp), MLX (mlx backend for Mac), and Safetensors (transformers)
# 'model_id' acts as the 'spec' for ModelManager.download()

MODELS = {
    "entry": [
        # GGUF Models - Efficient quantized models for CPU/GPU
        ModelEntry(
            name="Qwen 2.5 (0.5B) GGUF", backend="llama_cpp", model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            min_ram_gb=2, description="Ultra-lightweight GGUF. Good for basic testing.", 
            format="gguf", hw_tags=["cpu", "cuda", "metal"]
        ),
        ModelEntry(
            name="Llama 3.2 (1B) GGUF", backend="llama_cpp", model_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
            min_ram_gb=4, description="Meta's smallest instruction model in GGUF.", 
            format="gguf", hw_tags=["cpu", "cuda", "metal"]
        ),
        # MLX Models - Optimized for Apple Silicon
        ModelEntry(
            name="Qwen 2.5 (0.5B) MLX", backend="mlx", model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            min_ram_gb=2, description="Ultra-lightweight MLX for Mac. Blazing fast.", 
            format="mlx", hw_tags=["mlx", "mac"]
        ),
        ModelEntry(
            name="Llama 3.2 (1B) MLX", backend="mlx", model_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
            min_ram_gb=4, description="Meta's smallest model optimized for Apple Silicon.", 
            format="mlx", hw_tags=["mlx", "mac"]
        ),
        # Safetensors - Full precision models
        ModelEntry(
            name="Qwen 2.5 (0.5B) FP16", backend="transformers", model_id="Qwen/Qwen2.5-0.5B-Instruct",
            min_ram_gb=4, description="Full precision tiny model. Best quality.", 
            format="safetensors", hw_tags=["cuda", "cpu"]
        ),
    ],
    "mid": [
        # GGUF Models
        ModelEntry(
            name="Qwen 2.5 (7B) GGUF", backend="llama_cpp", model_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
            min_ram_gb=16, description="Excellent all-rounder GGUF. Best in class 7B.", 
            format="gguf", hw_tags=["cuda", "metal", "cpu"]
        ),
        ModelEntry(
            name="Llama 3.1 (8B) GGUF", backend="llama_cpp", model_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            min_ram_gb=16, description="Meta's state-of-the-art 8B GGUF.", 
            format="gguf", hw_tags=["cuda", "metal"]
        ),
        ModelEntry(
            name="Mistral 7B (v0.3) GGUF", backend="llama_cpp", model_id="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            min_ram_gb=16, description="Reliable workhorse GGUF from Mistral AI.", 
            format="gguf", hw_tags=["cuda", "metal"]
        ),
        # MLX Models
        ModelEntry(
            name="Qwen 2.5 (7B) MLX", backend="mlx", model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
            min_ram_gb=16, description="Excellent MLX reasoning model for Mac.", 
            format="mlx", hw_tags=["mlx", "mac"]
        ),
        ModelEntry(
            name="Llama 3.1 (8B) MLX", backend="mlx", model_id="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            min_ram_gb=16, description="Powerful MLX model optimized for Apple Silicon.", 
            format="mlx", hw_tags=["mlx", "mac"]
        ),
        # Safetensors Models
        ModelEntry(
            name="Qwen 2.5 (7B) FP16", backend="transformers", model_id="Qwen/Qwen2.5-7B-Instruct",
            min_ram_gb=32, description="Full precision 7B. Best quality, needs more VRAM.", 
            format="safetensors", hw_tags=["cuda"]
        ),
    ],
    "high": [
        # GGUF Models
        ModelEntry(
            name="Qwen 2.5 (14B) GGUF", backend="llama_cpp", model_id="Qwen/Qwen2.5-14B-Instruct-GGUF",
            min_ram_gb=28, description="Heavyweight reasoning GGUF. Great for coding.", 
            format="gguf", hw_tags=["cuda", "metal"]
        ),
        ModelEntry(
            name="Mixtral 8x7B GGUF", backend="llama_cpp", model_id="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
            min_ram_gb=26, description="Top-tier sparse MoE GGUF. Very fast.", 
            format="gguf", hw_tags=["cuda", "metal"]
        ),
        # MLX Models
        ModelEntry(
            name="Qwen 2.5 (14B) MLX", backend="mlx", model_id="mlx-community/Qwen2.5-14B-Instruct-4bit",
            min_ram_gb=28, description="Large MLX reasoning model for Mac Studio.", 
            format="mlx", hw_tags=["mlx", "mac"]
        ),
        ModelEntry(
            name="Llama 3.1 (70B) MLX", backend="mlx", model_id="mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
            min_ram_gb=64, description="Flagship MLX model for Mac Studio/Pro with lots of RAM.", 
            format="mlx", hw_tags=["mlx", "mac"]
        ),
        # Safetensors Models
        ModelEntry(
            name="Qwen 2.5 (14B) FP16", backend="transformers", model_id="Qwen/Qwen2.5-14B-Instruct",
            min_ram_gb=56, description="Full precision 14B. Best quality for high-end GPUs.", 
            format="safetensors", hw_tags=["cuda"]
        ),
    ]
}
