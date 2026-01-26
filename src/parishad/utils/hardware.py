"""
Hardware detection utility for Parishad.
Detects system capabilities (RAM, GPU, VRAM) to populate config.json.
"""
import platform
import psutil
import subprocess
import logging
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    name: str
    vram_gb: float
    type: str  # "apple_silicon", "cuda", "cpu"

@dataclass
class SystemInfo:
    os: str
    arch: str
    ram_gb: float
    gpu: GPUInfo

    def to_dict(self):
        return asdict(self)

def get_system_info() -> SystemInfo:
    """Detect all system information."""
    
    # 1. OS & Arch
    os_name = platform.system()
    arch = platform.machine()
    
    # 2. RAM
    try:
        ram_bytes = psutil.virtual_memory().total
        ram_gb = round(ram_bytes / (1024**3), 1)
    except:
        ram_gb = 8.0 # Fallback
        
    # 3. GPU Detection
    gpu = _detect_gpu(os_name, arch, ram_gb)
    
    return SystemInfo(
        os=os_name,
        arch=arch,
        ram_gb=ram_gb,
        gpu=gpu
    )

def _detect_gpu(os_name: str, arch: str, system_ram_gb: float) -> GPUInfo:
    """Detect GPU specifics."""
    
    # A. Apple Silicon
    if os_name == "Darwin" and "arm" in arch.lower():
        # Try to get specific chip name via sysctl
        try:
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            chip_name = result.stdout.strip()
        except:
            chip_name = "Apple Silicon"
            
        return GPUInfo(
            name=chip_name,
            vram_gb=system_ram_gb * 0.7, # Unified memory heuristic (approx usable)
            type="apple_silicon"
        )
        
    # B. NVIDIA (Priority: Torch -> nvidia-smi)
    
    # 1. Try Torch execution (Most reliable for Python environment)
    try:
        import torch
        if torch.cuda.is_available():
             vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
             return GPUInfo(
                name=torch.cuda.get_device_name(0),
                vram_gb=round(vram_gb, 1),
                type="cuda"
            )
    except ImportError:
        pass

    # 2. Try nvidia-smi (System level)
    try:
        # Check standard paths on Windows if not in PATH
        smi_cmd = "nvidia-smi"
        if os_name == "Windows":
             import shutil
             if not shutil.which("nvidia-smi"):
                 # Common install path
                 candidate = r"C:\Windows\System32\nvidia-smi.exe"
                 if os.path.exists(candidate):
                     smi_cmd = candidate
        
        # Query details
        cmd = [smi_cmd, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                parts = lines[0].split(',')
                name = parts[0].strip()
                vram_mb = float(parts[1].strip())
                return GPUInfo(
                    name=name,
                    vram_gb=round(vram_mb / 1024, 1),
                    type="cuda"
                )
    except Exception as e:
        logger.debug(f"nvidia-smi check failed: {e}")
        
    # C. Fallback CPU
    return GPUInfo(
        name="CPU",
        vram_gb=0.0,
        type="cpu"
    )
