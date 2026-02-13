# Production-Grade Debugging & VRAM Checking Guide

## Overview

This guide documents the comprehensive production-ready debugging, logging, and VRAM checking system implemented in Parishad. The system provides:

✅ **Automatic VRAM/GPU detection**  
✅ **Pre-flight compatibility checks before model loading**  
✅ **Comprehensive debug logging throughout the entire pipeline**  
✅ **User-friendly error messages (LM Studio-style)**  
✅ **Memory estimation for models from 0.5B to 70B+**  
✅ **GPU-only execution (no CPU fallback)**

---

## Features Implemented

### 1. Memory Estimation & VRAM Checking

**File:** `src/parishad/models/memory_estimation.py` (NEW)

#### Capabilities:
- Detects GPU VRAM using PyTorch and nvidia-smi
- Supports NVIDIA CUDA and Apple Silicon
- Estimates model memory requirements based on:
  - Parameter count (0.5B to 70B+)
  - Quantization format (Q2, Q3, Q4, Q5, Q6, Q8, FP16, etc.)
  - Context window size
  - KV cache requirements
  - Inference overhead

#### API:
```python
from parishad.models.memory_estimation import (
    detect_vram,
    estimate_model_memory,
    check_model_compatibility
)

# Detect VRAM
vram_info = detect_vram()
# Returns: VRAMInfo(available_gb, total_gb, gpu_name, gpu_type)

# Estimate model memory
mem_req = estimate_model_memory(
    params_billion=7.0,
    quantization="Q4_K_M",
    context_length=4096
)
# Returns: ModelMemoryRequirement with min/recommended VRAM

# Check compatibility (combines both)
is_compatible, message, mem_req = check_model_compatibility(
    model_id="Llama-3.2-3B",
    context_length=4096
)
```

---

### 2. Comprehensive Logging System

#### Backend Logging

**Files Modified:**
- `src/parishad/models/backends/llama_cpp.py`
- `src/parishad/models/backends/transformers_hf.py`

#### What's Logged:

**Model Loading:**
```
================================================================================
🚀 LLAMA.CPP MODEL LOADING STARTED
================================================================================
Model ID: Llama-3.2-3B-Instruct-GGUF
Context Length: 4096

🔍 Resolving model path for: Llama-3.2-3B-Instruct-GGUF
✅ Model path resolved: /path/to/model.gguf
   File size: 2048.50 MB

🔍 Checking GPU/VRAM compatibility...
✅ NVIDIA GPU detected: NVIDIA GeForce RTX 3090
   Total VRAM: 24.00 GB
   Available: 22.50 GB

🔍 Checking compatibility for Llama-3.2-3B-Instruct-GGUF
   Model: 3.0B parameters (Q4_K_M)
   Required VRAM: 2.50 GB (min), 3.25 GB (recommended)
   Available VRAM: 22.50 GB
✅ Compatible

📋 Loading configuration:
   GPU Layers: -1 (-1 = all layers on GPU)
   Context Size: 4096
   Batch Size: 512
   Verbose: False
   Chat Format: auto

⏳ Loading model into memory... (this may take 30-90 seconds)
✅ Model loaded successfully in 12.34s

================================================================================
✅ LLAMA.CPP LOADING COMPLETE - Total time: 15.67s
================================================================================
```

**Generation:**
```
────────────────────────────────────────────────────────────────────────────────
🎯 GENERATION STARTED
────────────────────────────────────────────────────────────────────────────────
Model: Llama-3.2-3B-Instruct-GGUF
Prompt length: 245 chars
Max tokens: 512
Temperature: 0.7
Top-p: 0.9
Stop sequences: none

⏳ Generating response...
✅ Generation complete in 8.45s
   Tokens: 60 in + 142 out = 202 total
   Speed: 16.8 tokens/sec
   Finish reason: stop
   Response length: 567 chars
   Response preview: The capital of France is Paris...
────────────────────────────────────────────────────────────────────────────────
```

**Errors:**
```
================================================================================
❌ LLAMA.CPP LOADING FAILED after 5.23s
================================================================================
Model: Llama-3.1-70B-Instruct-GGUF
Path: /path/to/model.gguf
Error: RuntimeError: CUDA out of memory

💡 This appears to be a VRAM/memory issue.
   Try a smaller model or more aggressive quantization.
```

---

### 3. User-Friendly Error Messages

**File Modified:** `src/parishad/cli/code.py`

#### Error Categories:

**1. GPU Memory Issues:**
```
⚠️  GPU Memory Issue Detected
The selected model is too large for your GPU.

Solutions:
  1. Try a smaller model (0.5B, 1B, or 3B parameters)
  2. Use more aggressive quantization (Q2, Q3, Q4)
  3. Close other GPU-intensive applications
  4. Check VRAM usage: nvidia-smi

Use /setup to select a different model.
```

**2. Model Not Found:**
```
⚠️  Model Not Found
The model could not be located on this system.

Solutions:
  1. Run /setup to download the model
  2. Check: parishad models list
  3. Download manually: parishad download <model>
```

**3. No GPU Detected:**
```
⚠️  No GPU Detected
This project requires a CUDA-capable GPU.

Requirements:
  • NVIDIA GPU with CUDA support
  • Updated GPU drivers
  • PyTorch with CUDA installed

Check: nvidia-smi
```

---

### 4. Enhanced Logging Configuration

**File Modified:** `src/parishad/utils/logging.py`

**Changes:**
- Default log level changed from `WARNING` to `INFO`
- Enhanced format includes function name and line numbers:
  ```
  2026-02-11 14:23:45 | INFO     | parishad.models.backends | load:123  | Loading model...
  ```

---

## Testing Guide

### Test Different Model Sizes

#### 1. Tiny Models (0.5B - 1B)
```bash
# Should work on any GPU with 2GB+ VRAM
parishad run "What is 2+2?" --model Qwen2.5-0.5B-Instruct-GGUF
```

#### 2. Small Models (1B - 3B)
```bash
# Should work on GPUs with 4GB+ VRAM
parishad run "What is the capital of France?" --model Llama-3.2-3B-Instruct-GGUF
```

#### 3. Medium Models (7B)
```bash
# Requires 8GB+ VRAM
parishad run "Explain quantum computing" --model Llama-3.1-8B-Instruct-GGUF
```

#### 4. Large Models (70B+)
```bash
# Requires 40GB+ VRAM - should fail gracefully with helpful message
parishad run "Test" --model Llama-3.1-70B-Instruct-GGUF
```

### Check Logs

**1. Enable DEBUG logging:**
```python
# In your code or config
from parishad.utils.logging import setup_logging
setup_logging(level="DEBUG")
```

**2. Check log output:**
```bash
# Logs will show in terminal with full context:
2026-02-11 14:23:45 | INFO     | parishad.models.backends.llama_cpp | load:125  | Model path resolved
2026-02-11 14:23:45 | INFO     | parishad.models.memory_estimation | detect_vram:98  | ✅ NVIDIA GPU detected
```

### Test VRAM Checking

```python
from parishad.models.memory_estimation import detect_vram, check_model_compatibility

# Check your GPU
vram = detect_vram()
print(f"GPU: {vram.gpu_name}")
print(f"VRAM: {vram.available_gb:.2f} GB available / {vram.total_gb:.2f} GB total")

# Check if a model will fit
is_compatible, message, mem_req = check_model_compatibility(
    model_id="Llama-3.1-8B-Instruct-GGUF",
    context_length=4096
)
print(f"Compatible: {is_compatible}")
print(f"Message: {message}")
print(f"Estimated VRAM needed: {mem_req.estimated_gb:.2f} GB")
```

---

## Configuration

### Enable Full Debug Logging

**Option 1: Environment Variable**
```bash
export PARISHAD_LOG_LEVEL=DEBUG
```

**Option 2: In Code**
```python
from parishad.utils.logging import setup_logging
setup_logging(level="DEBUG", log_file="parishad_debug.log")
```

### Adjust VRAM Safety Margins

Edit `src/parishad/models/memory_estimation.py`:
```python
# Line 250-260: Adjust safety multipliers
overhead_multiplier = 1.20  # 20% overhead (default)
recommended_gb = total_estimated_gb * 1.3  # 30% margin (default)

# For tighter margins:
overhead_multiplier = 1.15  # 15% overhead
recommended_gb = total_estimated_gb * 1.2  # 20% margin
```

---

## Architecture

### VRAM Checking Flow

```
1. User selects model in TUI/CLI
           ↓
2. detect_vram() → Queries GPU (PyTorch/nvidia-smi)
           ↓  
3. parse_model_size() → Extracts params & quantization from model ID
           ↓
4. estimate_model_memory() → Calculates VRAM needed
           ↓
5. check_model_compatibility() → Compares available vs needed
           ↓
6. If incompatible → Show error message, don't load
   If compatible → Proceed with model loading
           ↓
7. Backend.load() → Loads model with full logging
           ↓
8. Backend.generate() → Generates response with performance metrics
```

### Supported Quantization Formats

| Format | Bits/Param | VRAM Multiplier | Quality |
|--------|------------|-----------------|---------|
| Q2_K | 2.5 | 0.31x | Low |
| Q3_K_M | 3.5 | 0.44x | Medium-Low |
| Q4_K_M | 5.0 | 0.63x | Good (Recommended) |
| Q5_K_M | 6.0 | 0.75x | Very Good |
| Q6_K | 6.5 | 0.81x | Excellent |
| Q8_0 | 8.5 | 1.06x | Near-Perfect |
| FP16 | 16.0 | 2.00x | Perfect |

---

## Troubleshooting

### Issue: Logs not showing

**Solution:**
```python
from parishad.utils.logging import setup_logging
setup_logging(level="INFO")  # or "DEBUG"
```

### Issue: VRAM check reports wrong values

**Check:**
1. Is `nvidia-smi` working? Run: `nvidia-smi`
2. Is PyTorch detecting CUDA? Run:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

### Issue: Model loads despite "incompatible" warning

**Explanation:** Warnings allow loading (may be slow), Errors prevent loading.
- **Warning:** Available < Recommended (will load but may be slow)
- **Error:** Available < Minimum (won't load)

---

## Performance Metrics

All generation calls now log:
- **Tokens/second** - Speed of generation
- **Total tokens** - Input + output count
- **Latency** - Total time including preprocessing
- **Finish reason** - How generation ended (stop/length)

Example output:
```
✅ Generation complete in 8.45s
   Tokens: 60 in + 142 out = 202 total
   Speed: 16.8 tokens/sec
   Finish reason: stop
```

---

## Files Modified

### New Files:
- `src/parishad/models/memory_estimation.py` - VRAM detection & estimation

### Modified Files:
- `src/parishad/models/backends/llama_cpp.py` - Added comprehensive logging
- `src/parishad/models/backends/transformers_hf.py` - Added comprehensive logging  
- `src/parishad/models/runner.py` - Changed device_map default to "cuda"
- `src/parishad/cli/code.py` - Added user-friendly error messages + fixed Sabha role count bug
- `src/parishad/utils/logging.py` - Enhanced logging format & changed default to INFO

---

## Key Fixes Summary

### 1. Sabha Role Count Bug ✅
**Problem:** Laghu Sabha (5 roles) was executing 8 roles due to hardcoded `config="core"`  
**Solution:** Made config dynamic based on Sabha selection:
- Laghu Sabha → `pipeline.fast.yaml` (5 roles)
- Madhyam Sabha → `pipeline.core.yaml` (8 roles)
- Maha Sabha → `pipeline.extended.yaml` (10 roles)

### 2. GPU-Only Execution ✅
**Problem:** Code allowed CPU fallback via `device_map="auto"`  
**Solution:** 
- Changed `device_map` default from `"auto"` to `"cuda"`
- Disabled CPU offload in quantization config
- Set `n_gpu_layers=-1` (all layers on GPU)

### 3. Comprehensive Logging ✅
**Added:** Detailed logging for every step of model loading and generation

### 4. VRAM Checking ✅
**Added:** Pre-flight checks to prevent loading models that won't fit

### 5. User-Friendly Errors ✅
**Added:** LM Studio-style error messages with actionable solutions

---

## Next Steps for Testing

1. **Test with 0.5B model** - Should work on any GPU
2. **Test with 3B model** - Should work on 4GB+ GPUs
3. **Test with 7B model** - Should work on 8GB+ GPUs
4. **Test with 70B model** - Should fail gracefully with helpful message
5. **Check logs** - Verify detailed logging appears
6. **Verify Sabha roles** - Confirm correct role count for each Sabha

---

## Support

For issues or questions about the debugging system:
1. Check logs with `level="DEBUG"`
2. Run `nvidia-smi` to verify GPU
3. Check `parishad models list`
4. Review error messages for solutions

---

*Last Updated: February 11, 2026*
*Version: Production v1.0*
