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
    backend: Literal["llama_cpp", "ollama", "lm_studio"] # Defaulting to GGUF-compatible backends
    model_id: str  # The ID used by the backend or the lookup key
    min_ram_gb: int
    description: str
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

# Note: We prioritize GGUF (llama_cpp) for local ease (no gated repos, lower VRAM reqs).
# 'model_id' here acts as the 'spec' for ModelManager.download()

MODELS = {
    "entry": [
        ModelEntry(
            name="Qwen 2.5 (0.5B)", backend="llama_cpp", model_id="qwen2.5:0.5b",
            min_ram_gb=2, description="Ultra-lightweight. Good for basic testing.", 
            hw_tags=["cpu", "cuda", "mlx"]
        ),
        ModelEntry(
            name="Llama 3.2 (1B)", backend="llama_cpp", model_id="llama3.2:1b",
            min_ram_gb=4, description="Meta's smallest instruction model.", 
            hw_tags=["cpu", "cuda", "mlx"]
        ),
        ModelEntry(
            name="Phi-3.5 Mini (3.8B)", backend="llama_cpp", model_id="phi3:mini", # Using phi3:mini as close approx or update downloader
            min_ram_gb=8, description="High capability for size. Strong reasoning.", 
            hw_tags=["cpu", "cuda", "mlx"],
            download_info={"repo": "microsoft/Phi-3-mini-4k-instruct-gguf", "file": "Phi-3-mini-4k-instruct-q4.gguf"}
        ),
        ModelEntry(
            name="Gemma 2 (2B)", backend="llama_cpp", model_id="gemma2:2b",
            min_ram_gb=6, description="Google's lightweight efficient model.", 
            hw_tags=["cpu", "cuda", "mlx"]
        ),
        ModelEntry(
            name="Qwen 2.5 (1.5B)", backend="llama_cpp", model_id="qwen2.5:1.5b",
            min_ram_gb=4, description="Balanced small model.", 
            hw_tags=["cpu", "cuda", "mlx"]
        ),
    ],
    "mid": [
        ModelEntry(
            name="Qwen 2.5 (7B)", backend="llama_cpp", model_id="qwen2.5:7b",
            min_ram_gb=16, description="Excellent all-rounder. Best in class 7B.", 
            hw_tags=["cuda", "mlx"]
        ),
        ModelEntry(
            name="Llama 3.1 (8B)", backend="llama_cpp", model_id="llama3.1:8b",
            min_ram_gb=16, description="Meta's state-of-the-art 8B model.", 
            hw_tags=["cuda", "metal"]
        ),
        ModelEntry(
            name="Mistral 7B (v0.3)", backend="llama_cpp", model_id="mistral:7b",
            min_ram_gb=16, description="Reliable workhorse from Mistral AI.", 
            hw_tags=["cuda", "mlx"]
        ),
    ],
    "high": [
        ModelEntry(
            name="Qwen 2.5 (14B)", backend="llama_cpp", model_id="Qwen/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m.gguf",
            min_ram_gb=28, description="Heavyweight reasoning. Great for coding.", 
            hw_tags=["cuda", "mlx"]
        ),
        ModelEntry(
            name="Mixtral 8x7B", backend="llama_cpp", model_id="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
            min_ram_gb=26, description="Top-tier sparse MoE. Very fast.", 
            hw_tags=["cuda", "metal"]
        ),
    ]
}
