"""
Production-grade memory estimation and VRAM checking for model loading.

Provides utilities to:
- Detect available GPU VRAM
- Estimate model memory requirements
- Check compatibility before loading
- Provide user-friendly error messages

Mimics LM Studio's behavior for hardware compatibility checking.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import json

logger = logging.getLogger(__name__)


@dataclass
class VRAMInfo:
    """GPU VRAM information."""
    available_gb: float
    total_gb: float
    gpu_name: str
    gpu_type: str  # "cuda", "apple_silicon", "cpu"
    
    @property
    def is_gpu(self) -> bool:
        """Check if actual GPU is available."""
        return self.gpu_type in ("cuda", "apple_silicon")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "available_gb": round(self.available_gb, 2),
            "total_gb": round(self.total_gb, 2),
            "gpu_name": self.gpu_name,
            "gpu_type": self.gpu_type,
            "is_gpu": self.is_gpu
        }


@dataclass
class ModelMemoryRequirement:
    """Estimated memory requirements for a model."""
    model_id: str
    params_billion: float
    quantization: str
    estimated_gb: float
    min_required_gb: float  # Minimum VRAM needed
    recommended_gb: float  # Recommended VRAM for good performance
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "model_id": self.model_id,
            "params_billion": self.params_billion,
            "quantization": self.quantization,
            "estimated_gb": round(self.estimated_gb, 2),
            "min_required_gb": round(self.min_required_gb, 2),
            "recommended_gb": round(self.recommended_gb, 2)
        }


def detect_vram() -> VRAMInfo:
    """
    Detect available GPU VRAM with production-grade error handling.
    
    Returns:
        VRAMInfo with current GPU/VRAM status
    """
    logger.info("🔍 Detecting GPU and VRAM...")
    
    # Try NVIDIA CUDA first
    try:
        import torch
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            total_vram_gb = device_props.total_memory / (1024**3)
            
            # Get available VRAM (total - currently used)
            torch.cuda.empty_cache()  # Clear cache first
            allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
            available_gb = total_vram_gb - reserved_gb
            
            gpu_name = torch.cuda.get_device_name(0)
            
            logger.info(f"✅ NVIDIA GPU detected: {gpu_name}")
            logger.info(f"   Total VRAM: {total_vram_gb:.2f} GB")
            logger.info(f"   Available: {available_gb:.2f} GB")
            logger.info(f"   Allocated: {allocated_gb:.2f} GB")
            logger.info(f"   Reserved: {reserved_gb:.2f} GB")
            
            return VRAMInfo(
                available_gb=available_gb,
                total_gb=total_vram_gb,
                gpu_name=gpu_name,
                gpu_type="cuda"
            )
    except ImportError:
        logger.debug("PyTorch not available, trying nvidia-smi...")
    except Exception as e:
        logger.debug(f"PyTorch CUDA check failed: {e}")
    
    # Try nvidia-smi as fallback
    try:
        import shutil
        import platform
        
        smi_cmd = "nvidia-smi"
        if platform.system() == "Windows":
            if not shutil.which("nvidia-smi"):
                candidate = r"C:\Windows\System32\nvidia-smi.exe"
                if Path(candidate).exists():
                    smi_cmd = candidate
        
        # Query GPU memory
        cmd = [
            smi_cmd,
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                parts = lines[0].split(',')
                gpu_name = parts[0].strip()
                total_mb = float(parts[1].strip())
                free_mb = float(parts[2].strip())
                
                total_gb = total_mb / 1024
                available_gb = free_mb / 1024
                
                logger.info(f"✅ NVIDIA GPU detected (via nvidia-smi): {gpu_name}")
                logger.info(f"   Total VRAM: {total_gb:.2f} GB")
                logger.info(f"   Available: {available_gb:.2f} GB")
                
                return VRAMInfo(
                    available_gb=available_gb,
                    total_gb=total_gb,
                    gpu_name=gpu_name,
                    gpu_type="cuda"
                )
    except Exception as e:
        logger.debug(f"nvidia-smi check failed: {e}")
    
    # Check for Apple Silicon
    try:
        import platform
        if platform.system() == "Darwin" and "arm" in platform.machine().lower():
            # Apple Silicon uses unified memory
            import psutil
            total_ram = psutil.virtual_memory().total / (1024**3)
            available_ram = psutil.virtual_memory().available / (1024**3)
            
            # Estimate usable GPU memory (roughly 70% of total RAM)
            gpu_memory = total_ram * 0.7
            available_gpu = available_ram * 0.7
            
            logger.info(f"✅ Apple Silicon detected")
            logger.info(f"   Unified Memory: {total_ram:.2f} GB")
            logger.info(f"   Available for GPU: {available_gpu:.2f} GB")
            
            return VRAMInfo(
                available_gb=available_gpu,
                total_gb=gpu_memory,
                gpu_name="Apple Silicon",
                gpu_type="apple_silicon"
            )
    except Exception as e:
        logger.debug(f"Apple Silicon check failed: {e}")
    
    # No GPU detected - CPU only
    logger.warning("⚠️  No GPU detected - CPU-only mode")
    logger.warning("   This project requires GPU for optimal performance")
    
    return VRAMInfo(
        available_gb=0.0,
        total_gb=0.0,
        gpu_name="CPU",
        gpu_type="cpu"
    )


def estimate_model_memory(
    params_billion: float,
    quantization: str = "Q4_K_M",
    context_length: int = 4096
) -> ModelMemoryRequirement:
    """
    Estimate model memory requirements with high accuracy.
    
    Based on:
    - Parameter count
    - Quantization level
    - Context window size
    - Overhead for KV cache and inference
    
    Args:
        params_billion: Model size in billions of parameters
        quantization: Quantization format ("Q4_K_M", "Q8_0", "FP16", "4bit", "8bit")
        context_length: Context window size
        
    Returns:
        ModelMemoryRequirement with detailed estimates
    """
    # Bits per parameter for different quantization levels
    quantization_bits = {
        # GGUF quants
        "Q2_K": 2.5,
        "Q3_K_S": 3.0,
        "Q3_K_M": 3.5,
        "Q3_K_L": 3.8,
        "Q4_0": 4.5,
        "Q4_K_S": 4.5,
        "Q4_K_M": 5.0,
        "Q4_K_L": 5.5,
        "Q5_0": 5.5,
        "Q5_K_S": 5.5,
        "Q5_K_M": 6.0,
        "Q5_K_L": 6.5,
        "Q6_K": 6.5,
        "Q8_0": 8.5,
        # Standard formats
        "4bit": 4.5,
        "8bit": 8.5,
        "FP16": 16.0,
        "BF16": 16.0,
        "FP32": 32.0,
        # MLX formats
        "4bit-mlx": 4.5,
        "8bit-mlx": 8.5,
    }
    
    # Normalize quantization string
    quant_key = quantization.upper().replace("-", "_")
    if "4BIT" in quant_key or "4-BIT" in quantization.lower():
        quant_key = "4bit"
    elif "8BIT" in quant_key or "8-BIT" in quantization.lower():
        quant_key = "8bit"
    
    bits_per_param = quantization_bits.get(quant_key, 5.0)  # Default to Q4_K_M equivalent
    
    # Base model memory: params × bits_per_param / 8 (convert to bytes)
    model_memory_gb = (params_billion * 1e9 * bits_per_param) / (8 * 1024**3)
    
    # KV cache memory: Simple approximation
    # For Q4 models: ~0.04 GB per 1B params per 1K context
    # For FP16: ~0.08 GB per 1B params per 1K context
    kv_cache_multiplier = 0.08 if bits_per_param >= 16 else 0.04
    context_k = context_length / 1024  # Convert to K tokens
    kv_cache_gb = params_billion * context_k * kv_cache_multiplier
    
    # Inference overhead: activations, batching, etc.
    # Roughly 15-20% of model size
    overhead_multiplier = 1.20
    
    # Total estimated memory
    total_estimated_gb = (model_memory_gb + kv_cache_gb) * overhead_multiplier
    
    # Minimum required: model + minimal context (2K)
    min_context = 2048
    min_context_k = min_context / 1024
    min_kv_cache_gb = params_billion * min_context_k * kv_cache_multiplier
    min_required_gb = (model_memory_gb + min_kv_cache_gb) * 1.15
    
    # Recommended: comfortable margins for good performance
    recommended_gb = total_estimated_gb * 1.3
    
    logger.debug(f"Memory estimation for {params_billion}B model ({quantization}):")
    logger.debug(f"  Model weights: {model_memory_gb:.2f} GB")
    logger.debug(f"  KV cache ({context_length} ctx): {kv_cache_gb:.2f} GB")
    logger.debug(f"  Total estimated: {total_estimated_gb:.2f} GB")
    logger.debug(f"  Minimum required: {min_required_gb:.2f} GB")
    logger.debug(f"  Recommended: {recommended_gb:.2f} GB")
    
    return ModelMemoryRequirement(
        model_id="",  # Will be set by caller
        params_billion=params_billion,
        quantization=quantization,
        estimated_gb=total_estimated_gb,
        min_required_gb=min_required_gb,
        recommended_gb=recommended_gb
    )


def parse_model_size(model_id: str, catalog_path: Optional[Path] = None) -> Tuple[float, str]:
    """
    Parse model size and quantization from model ID or catalog.
    
    Args:
        model_id: Model identifier
        catalog_path: Optional path to models.json catalog
        
    Returns:
        Tuple of (params_billion, quantization)
    """
    # Try to load from catalog first
    if catalog_path is None:
        catalog_path = Path(__file__).parent.parent / "data" / "models.json"
    
    if catalog_path.exists():
        try:
            with open(catalog_path, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
            
            # Search all sources
            for source_data in catalog.get("sources", {}).values():
                for model in source_data.get("models", []):
                    if model_id in model.get("shortcut", "") or model_id == model.get("name", ""):
                        params_str = model.get("params", "")
                        quant = model.get("quantization", "Q4_K_M")
                        
                        # Parse params (e.g., "0.5B", "7B", "8x7B")
                        from ..models.size_detection import parse_params_from_string
                        params_billion = parse_params_from_string(params_str)
                        
                        return params_billion, quant
        except Exception as e:
            logger.debug(f"Failed to load model catalog: {e}")
    
    # Fallback: parse from model_id string
    import re
    
    # Look for patterns like "3B", "7B", "0.5B", "1.5B"
    match = re.search(r'(\d+\.?\d*)[Bb]', model_id)
    if match:
        params_billion = float(match.group(1))
    else:
        # Default assumption for unknown models
        params_billion = 7.0
        logger.warning(f"Could not determine model size from '{model_id}', assuming {params_billion}B")
    
    # Look for quantization hints
    quant = "Q4_K_M"  # Default
    if "Q2" in model_id.upper():
        quant = "Q2_K"
    elif "Q3" in model_id.upper():
        quant = "Q3_K_M"
    elif "Q4" in model_id.upper():
        quant = "Q4_K_M"
    elif "Q5" in model_id.upper():
        quant = "Q5_K_M"
    elif "Q6" in model_id.upper():
        quant = "Q6_K"
    elif "Q8" in model_id.upper():
        quant = "Q8_0"
    elif "4bit" in model_id.lower() or "4BIT" in model_id:
        quant = "4bit"
    elif "8bit" in model_id.lower() or "8BIT" in model_id:
        quant = "8bit"
    elif "FP16" in model_id.upper() or "fp16" in model_id.lower():
        quant = "FP16"
    
    return params_billion, quant


def check_model_compatibility(
    model_id: str,
    vram_info: Optional[VRAMInfo] = None,
    context_length: int = 4096
) -> Tuple[bool, str, Optional[ModelMemoryRequirement]]:
    """
    Check if a model is compatible with available hardware.
    
    Args:
        model_id: Model identifier
        vram_info: GPU/VRAM information (will detect if None)
        context_length: Desired context window
        
    Returns:
        Tuple of (is_compatible, message, memory_requirement)
    """
    if vram_info is None:
        vram_info = detect_vram()
    
    # Parse model size
    params_billion, quantization = parse_model_size(model_id)
    
    # Estimate memory requirement
    mem_req = estimate_model_memory(params_billion, quantization, context_length)
    mem_req.model_id = model_id
    
    logger.info(f"🔍 Checking compatibility for {model_id}")
    logger.info(f"   Model: {params_billion}B parameters ({quantization})")
    logger.info(f"   Required VRAM: {mem_req.min_required_gb:.2f} GB (min), {mem_req.recommended_gb:.2f} GB (recommended)")
    logger.info(f"   Available VRAM: {vram_info.available_gb:.2f} GB")
    
    # Check if GPU is available
    if not vram_info.is_gpu:
        message = (
            f"❌ GPU Required\n\n"
            f"This project requires a GPU but none was detected.\n"
            f"Model '{model_id}' ({params_billion}B parameters) needs at least "
            f"{mem_req.min_required_gb:.1f} GB VRAM.\n\n"
            f"Please ensure:\n"
            f"• NVIDIA GPU with CUDA support is installed\n"
            f"• GPU drivers are up to date\n"
            f"• PyTorch with CUDA is installed\n"
        )
        logger.error(message)
        return False, message, mem_req
    
    # Check if enough VRAM
    if vram_info.available_gb < mem_req.min_required_gb:
        message = (
            f"❌ Insufficient VRAM\n\n"
            f"Model: {model_id} ({params_billion}B parameters)\n"
            f"Required: {mem_req.min_required_gb:.1f} GB VRAM (minimum)\n"
            f"Available: {vram_info.available_gb:.1f} GB on {vram_info.gpu_name}\n\n"
            f"This model is too large for your GPU.\n\n"
            f"Suggestions:\n"
            f"• Try a smaller model (0.5B, 1B, or 3B)\n"
            f"• Use a more aggressive quantization (Q2, Q3, Q4)\n"
            f"• Close other GPU-intensive applications\n"
            f"• Upgrade to a GPU with more VRAM\n"
        )
        logger.warning(message)
        return False, message, mem_req
    
    # Warn if below recommended
    if vram_info.available_gb < mem_req.recommended_gb:
        message = (
            f"⚠️  Limited VRAM\n\n"
            f"Model: {model_id} ({params_billion}B parameters)\n"
            f"Available: {vram_info.available_gb:.1f} GB\n"
            f"Recommended: {mem_req.recommended_gb:.1f} GB\n\n"
            f"The model will load but may be slow or have reduced context.\n"
            f"Consider using a smaller model for better performance.\n"
        )
        logger.warning(message)
        return True, message, mem_req
    
    # All good!
    message = (
        f"✅ Compatible\n\n"
        f"Model: {model_id} ({params_billion}B parameters)\n"
        f"GPU: {vram_info.gpu_name}\n"
        f"Available VRAM: {vram_info.available_gb:.1f} GB\n"
        f"Required: {mem_req.estimated_gb:.1f} GB\n"
    )
    logger.info(message)
    return True, message, mem_req
