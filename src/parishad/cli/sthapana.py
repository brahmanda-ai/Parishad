"""
Parishad Sthapana (Enhanced): The Council Setup Wizard v3.
Features: Dashboard, Local-First Auto-Config, Smart Allocator, Claude-Code Tier TUI.
"""
import sys
import time
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import rich
from rich.console import Console, Group
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, 
    DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
)
from rich import box
from rich.layout import Layout
from rich.text import Text
from rich.align import Align

from ..utils.hardware import get_hardware_stats, HardwareStats
from ..data.catalog import SABHAS, MODELS, ModelEntry, SabhaConfig
from ..models.downloader import ModelManager, DownloadProgress, ModelInfo
from ..cli.prarambh import main as run_prarambh

console = Console()
DEFAULT_CONFIG_PATH = Path("models.yaml")

# --- VISUALS ---

def print_dashboard(stats: HardwareStats, manager: ModelManager):
    """Print the "Control Center" dashboard."""
    console.clear()
    
    # 1. Hardware Panel
    hw_table = Table.grid(padding=(0, 2), expand=True)
    hw_table.add_column(style="cyan", justify="left", ratio=1)
    hw_table.add_column(style="white", justify="right", ratio=2)
    hw_table.add_row("OS/Arch", f"{stats.os} ({stats.arch})")
    hw_table.add_row("RAM", f"{stats.ram_total_gb} GB")
    gpu = stats.gpu_name or "None"
    if stats.gpu_vram_gb: gpu += f" ({stats.gpu_vram_gb} GB)"
    hw_table.add_row("GPU", gpu)
    
    tier_style = {"entry": "yellow", "mid": "blue", "high": "green"}
    tier_txt = f"[{tier_style[stats.tier]}] {stats.tier.upper()} TIER [/]"
    
    # 2. Local Library Panel
    local_models = manager.list_models()
    lib_table = Table.grid(padding=(0, 2), expand=True)
    lib_table.add_column(style="magenta", justify="left", ratio=1)
    lib_table.add_column(style="white", justify="right", ratio=2)
    lib_table.add_row("Installed Models", str(len(local_models)))
    
    sources = set(m.source.value for m in local_models)
    source_txt = ", ".join(sources) if sources else "[dim]None[/dim]"
    lib_table.add_row("Sources", source_txt)
    
    # Grid Layout
    grid = Table.grid(expand=True, padding=2)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    
    p_hw = Panel(
        Group(
            Align.center(tier_txt),
            hw_table
        ),
        title="[bold]System Hardware[/bold]",
        border_style="cyan",
        padding=(1, 2)
    )
    
    p_lib = Panel(
        lib_table,
        title="[bold]Local Library[/bold]",
        border_style="magenta",
        padding=(1, 2)
    )
    
    grid.add_row(p_hw, p_lib)
    
    console.print(Panel(
        Align.center("[bold magenta]PARISHAD STHAPANA[/bold magenta] [dim]v3[/dim]"),
        box=box.DOUBLE_EDGE,
        border_style="magenta"
    ))
    console.print(grid)
    
    if stats.is_apple_silicon:
        console.print(Align.center("[dim]⚡ Apple Silicon Optimization Active (MLX)[/dim]"))
    console.print()

# --- LOGIC ---

def suggest_configuration_from_local(local_models: List[ModelInfo], mode: str, tier: str) -> Optional[Dict[str, ModelEntry]]:
    """Try to auto-allocate local models to slots."""
    if not local_models:
        return None
        
    # Heuristic: Sort by size (bytes)
    # This is rough but generally bigger file = smarter model
    sorted_models = sorted(local_models, key=lambda m: m.size_bytes, reverse=True)
    
    # Create pseudo catalog entries for local models so rest of logic works
    candidates = []
    for m in sorted_models:
        # Infer RAM usage roughly (size in GB + overhead)
        size_gb = m.size_bytes / (1024**3)
        min_ram = int(size_gb * 1.2) 
        
        # Determine backend from format
        backend = "transformers"
        if m.format.value == "gguf":
            backend = "llama_cpp"
        elif m.format.value == "ollama":
            backend = "ollama"
            
        candidates.append(ModelEntry(
            name=m.name,
            backend=backend, # Correctly mapped backend
            model_id=m.name, # Use registry name as ID
            min_ram_gb=min_ram,
            description="Locally installed model",
            hw_tags=["cpu", "cuda", "mlx"] if m.format.value=="gguf" else ["cuda"]
        ))

    mapping = {}
    
    if mode == "single":
        best = candidates[0]
        mapping = {"small": best, "mid": best, "big": best}
        
    elif mode == "dual":
        if len(candidates) < 2: return None
        big = candidates[0]
        mid = candidates[1]
        mapping = {"small": mid, "mid": mid, "big": big}

    elif mode == "triple":
        if len(candidates) < 3: return None
        big = candidates[0]
        mid = candidates[1]
        small = candidates[-1] # Smallest
        mapping = {"small": small, "mid": mid, "big": big}
        
    return mapping

def pick_model_smart(prompt: str, tier: str, hw_stats: HardwareStats, manager: ModelManager) -> ModelEntry:
    """Smart selection table mixing Local and Marketplace."""
    
    # 1. Market Candidates
    candidates = MODELS.get(tier, MODELS["entry"])
    preferred_tag = "mlx" if hw_stats.is_apple_silicon else "cuda" if hw_stats.gpu_name else "cpu"
    
    market_primary = [m for m in candidates if preferred_tag in m.hw_tags]
    market_secondary = [m for m in candidates if preferred_tag not in m.hw_tags and "cpu" in m.hw_tags]
    market_list = market_primary + market_secondary
    
    # 2. Local Candidates (that match heuristic for this slot?)
    # For simplicity, we show relevant ones.
    # Actually, let's just show the Market list but mark if installed.
    # AND add any "Other Local" option? No, stick to curated for stability + "Custom Local"
    
    console.print(f"\n[bold underline]{prompt}[/bold underline]")
    
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("ID", style="dim", width=3)
    table.add_column("Model Name", style="bold")
    table.add_column("Type", width=8)
    table.add_column("Status", width=12)
    table.add_column("Description")
    
    options = []
    
    # Process Market Options
    for i, m in enumerate(market_list):
        options.append(m)
        path = manager.get_model_path(m.model_id)
        status = "[green]✓ Installed[/green]" if path and path.exists() else "[dim]Download[/dim]"
        table.add_row(str(i+1), m.name, m.backend, status, m.description)
        
    # Process "Use Any Local" option?
    # Maybe listing all local files is too messy if user has 50.
    # Instead, we check if there are other locals not in catalog?
    # For now, simplistic approach: Just catalog + status integration.
        
    console.print(table)
    
    # Choice logic
    while True:
        choice = Prompt.ask("Select Model", choices=[str(i) for i in range(1, len(options)+1)], default="1")
        try:
            return options[int(choice)-1]
        except (ValueError, IndexError):
            console.print("[red]Invalid selection[/red]")


def resolve_slots_ui(mode: str, tier: str, hw_stats: HardwareStats, manager: ModelManager) -> Dict[str, ModelEntry]:
    selected = {}
    
    display_mode = {"single": "Single Mode", "dual": "Dual Mode", "triple": "Triple Mode"}
    console.print(f"\n[bold cyan]Configuring: {display_mode.get(mode, mode)}[/bold cyan]")
    
    if mode == "single":
        m = pick_model_smart("Select Main Model (Shared)", tier, hw_stats, manager)
        selected = {"small": m, "mid": m, "big": m}
        
    elif mode == "dual":
        mid = pick_model_smart("Select Worker Model", tier, hw_stats, manager)
        big = pick_model_smart("Select Planner Model", tier, hw_stats, manager)
        selected = {"small": mid, "mid": mid, "big": big}
        
    elif mode == "triple":
        if tier == "entry": 
            # Entry tier triple mode forces small models
            t_small, t_mid, t_big = "entry", "entry", "entry"
        else:
            t_small, t_mid, t_big = "entry", "mid", "high"
            
        small = pick_model_smart("Select Fast Model (Refiner)", t_small, hw_stats, manager)
        mid = pick_model_smart("Select Worker Model", t_mid, hw_stats, manager)
        big = pick_model_smart("Select Planner Model", t_big, hw_stats, manager)
        selected = {"small": small, "mid": mid, "big": big}

    return selected

def download_phase(selected: Dict[str, ModelEntry], manager: ModelManager):
    """Refined download phase."""
    unique = {m.model_id: m for m in selected.values()}
    
    to_download = []
    for m in unique.values():
        p = manager.get_model_path(m.model_id)
        if not p or not p.exists():
            to_download.append(m)
            
    if not to_download:
        console.print("\n[green]All models available locally. Skipping download.[/green]")
        return

    console.print(f"\n[bold yellow]Initiating Download ({len(to_download)} models)[/bold yellow]")
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[name]}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console
    )
    
    with progress:
        for m in to_download:
            tid = progress.add_task("download", name=m.name, total=None)
            
            def cb(p: DownloadProgress):
                progress.update(tid, total=p.total_bytes, completed=p.downloaded_bytes)
                
            try:
                # Map backend
                src = "auto"
                if m.backend == "transformers": src = "huggingface"
                elif m.backend == "ollama": src = "ollama"
                
                manager.download(m.model_id, source=src, progress_callback=cb)
                progress.update(tid, description=f"[green]✓ {m.name}[/green]")
            except Exception as e:
                progress.update(tid, description=f"[red]✗ {m.name} Failed[/red]")
                console.print(f"[red]Error downloading {m.name}: {e}[/red]")


def main():
    # 0. Init
    manager = ModelManager()
    console.print("[dim]Scanning local library...[/dim]")
    manager.scan_for_models()
    removed = manager.registry.verify_integrity()
    if removed > 0:
        console.print(f"[dim]Pruned {removed} invalid entries from registry.[/dim]")
    
    stats = get_hardware_stats()
    
    # 1. Dashboard
    print_dashboard(stats, manager)
    if not Confirm.ask("Begin Setup?", default=True): return

    # 2. Sabha
    console.print("\n[bold]1. Council Configuration[/bold]")
    for key, s in SABHAS.items():
        console.print(f"[cyan]{s.name}[/cyan] ({len(s.roles)} Roles): {s.description}")
        
    sabha = SABHAS[Prompt.ask("Choice", choices=["laghu", "mantri", "maha"], default="laghu")]

    # 3. Strategy
    console.print("\n[bold]2. Model Strategy[/bold]")
    console.print(Panel(
        "[bold]Single Mode:[/bold] 1 Shared Model (Simple, Less Memory)\n"
        "[bold]Dual Mode:[/bold] 1 Mid (Worker) + 1 Heavy (Planner/Judge)\n"
        "[bold]Triple Mode:[/bold] 1 Light (Refiner) + 1 Mid (Worker) + 1 Heavy (Planner)",
        title="Strategies", border_style="cyan"
    ))
    mode = Prompt.ask("Strategy", choices=["single", "dual", "triple"], default="triple" if stats.tier == "high" else "single")

    # 4. Auto-Detect / Allocation
    local_inventory = manager.list_models()
    suggestion = None
    
    if len(local_inventory) >= (3 if mode=="triple" else 1):
        suggestion = suggest_configuration_from_local(local_inventory, mode, stats.tier)
        
    selected_slots = {}
    
    if suggestion:
        console.print(Panel(
            "\n".join([f"[bold]{k.upper()}:[/bold] {v.name}" for k,v in suggestion.items()]),
            title="[bold green]Auto-Configuration Available[/bold green]",
            border_style="green"
        ))
        if Confirm.ask("Use this configuration? (Uses existing models)", default=True):
            selected_slots = suggestion
            
    if not selected_slots:
        # Manual Selection Flow
        selected_slots = resolve_slots_ui(mode, stats.tier, stats, manager)
        
    # 5. Download
    download_phase(selected_slots, manager)
    
    # 6. Generate Config
    # Reuse existing generate_config logic but inline here for simplicity/import
    cfg = {"slots": {}}
    for slot, m in selected_slots.items():
        # Backend resolution logic
        bk = "llama_cpp" if m.backend in ["llama_cpp", "lm_studio", "ollama"] else m.backend
        path = m.model_id
        
        # Resolve path
        rp = manager.get_model_path(m.model_id)
        if rp: 
            path = str(rp)
            if str(rp).endswith(".gguf"): bk = "llama_cpp"
            
        cfg["slots"][slot] = {
            "backend": bk,
            "model_id": path,
            "context_length": sabha.min_tokens_req,
            "temperature": 0.5
        }
        
    with open(DEFAULT_CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
        
    console.print(f"\n[bold green]Configuration saved to {DEFAULT_CONFIG_PATH}[/bold green]")
    
    # 7. Launch
    console.print(Panel("[bold white]Setup Complete[/bold white]", style="green"))
    if Confirm.ask("Launch Prarambh?", default=True):
        console.clear()
        run_prarambh()

if __name__ == "__main__":
    main()
