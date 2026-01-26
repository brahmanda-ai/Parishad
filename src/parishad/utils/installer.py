
import subprocess
import sys
import platform
import os
import importlib.util
from rich.console import Console

console = Console()

def check_and_install_backend():
    """
    Check if llama-cpp-python is installed.
    If not, automatically install it using pre-built wheels or optimal settings.
    """
    if importlib.util.find_spec("llama_cpp") is not None:
        return

    console.print("\n[bold yellow]⚠️  Core backend (llama-cpp-python) is missing.[/bold yellow]")
    console.print("[dim]Parishad needs this to run local models.[/dim]")
    
    system = platform.system()
    
    # 1. Windows: The main pain point. Use pre-built wheels.
    if system == "Windows":
        console.print("\n[cyan]🪟 Windows detected. Scanning for NVIDIA GPU...[/cyan]")
        
        use_cuda = False
        try:
            # Simple check for nvcc
            subprocess.run(["nvcc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            console.print("[green]✓ CUDA Toolkit detected (nvcc found).[/green]")
            use_cuda = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("[yellow]! CUDA not found. Defaulting to CPU.[/yellow]")
            console.print("[dim](If you have an NVIDIA GPU, install CUDA Toolkit 12.x for 10x speed)[/dim]")
        
        console.print(f"\n[bold green]🚀 Auto-installing optimized backend for {'CUDA 12.x' if use_cuda else 'CPU'}...[/bold green]")
        console.print("[dim]This may take a minute...[/dim]\n")
        
        cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python"]
        cmd.extend(["--prefer-binary", "--extra-index-url"])
        
        if use_cuda:
            # Use cu124 wheels (compatible with most modern 12.x)
            cmd.append("https://abetlen.github.io/llama-cpp-python/whl/cu124")
        else:
            cmd.append("https://abetlen.github.io/llama-cpp-python/whl/cpu")
            
        try:
            subprocess.check_call(cmd)
            console.print("\n[bold green]✓ Backend installed successfully![/bold green]")
        except subprocess.CalledProcessError:
            console.print("\n[bold red]❌ Installation failed.[/bold red]")
            console.print("Please copy-paste this command manually:")
            console.print(f"  {' '.join(cmd)}")
            sys.exit(1)

    # 2. Mac: Enable Metal
    elif system == "Darwin":
        console.print("\n[cyan]🍎 Mac detected. Installing with Metal (GPU) support...[/cyan]")
        
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_METAL=on"
        
        cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python"]
        
        try:
            subprocess.check_call(cmd, env=env)
            console.print("\n[bold green]✓ Backend installed successfully![/bold green]")
        except subprocess.CalledProcessError:
            console.print("\n[bold red]❌ Installation failed.[/bold red]")
            sys.exit(1)

    # 3. Linux: Standard install
    else:
        console.print("\n[cyan]🐧 Linux detected. Installing from PyPI...[/cyan]")
        cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python"]
        try:
            subprocess.check_call(cmd)
            console.print("\n[bold green]✓ Backend installed successfully![/bold green]")
        except subprocess.CalledProcessError:
            sys.exit(1)
            
    # clear some space
    print()
