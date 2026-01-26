"""
Parishad CLI - Command line interface for the Parishad council.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..orchestrator.engine import Parishad, ParishadEngine, PipelineConfig
from ..models.runner import ModelConfig


# Setup encoding for Windows
if sys.platform == "win32":
    try:
        # Try to reconfigure stdout/stderr to use UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        # If reconfigure fails, we'll just avoid Unicode characters
        pass

# Create console with error handling for encoding issues
try:
    console = Console()
    # Simple test - just create console, don't test Unicode output
    UNICODE_SUPPORTED = sys.platform != "win32"  # Disable Unicode on Windows by default
except Exception:
    UNICODE_SUPPORTED = False
    console = Console()


def get_config_dir() -> Path:
    """Get the default configuration directory."""
    # Check for local config first
    local_config = Path("./parishad/config")
    if local_config.exists():
        return local_config
    
    # Fall back to package config
    package_dir = Path(__file__).parent.parent
    return package_dir / "config"


def get_parishad_dir() -> Path:
    """Get the .parishad directory path (cross-platform)."""
    return Path.home() / ".parishad"


def is_first_run() -> bool:
    """Check if this is the first time parishad is being run."""
    parishad_dir = get_parishad_dir()
    return not parishad_dir.exists()


def first_run() -> bool:
    """
    Handle first-time setup permissions.
    
    Asks for:
    1. Permission to read files from the system
    2. Permission to create ~/.parishad directory
    
    Returns:
        True if setup completed, False if user declined
    """
    from .code import LOGO
    console.print(LOGO)
    console.print("[dim]पारिषद् में आपका स्वागत है![/dim]")
    console.print()
    
    # Permission 1: Read access
    console.print("[yellow]⚠️  Parishad needs permission to read files from your system.[/yellow]")
    console.print("[dim]This allows the coding assistant to understand your codebase[/dim]")
    console.print("[dim]and scan for existing model files to save download time.[/dim]")
    console.print()
    
    read_permission = click.confirm(
        "Grant read permission?",
        default=True
    )
    
    if not read_permission:
        console.print("[red]Read permission denied. Parishad cannot function without this.[/red]")
        return False
    
    console.print("[green]✓ Read permission granted.[/green]")
    console.print()
    
    # Permission 2: Write access to create .parishad directory
    parishad_dir = get_parishad_dir()
    console.print(f"[yellow]⚠️  Parishad needs to create a folder at:[/yellow]")
    console.print(f"    [bold]{parishad_dir}[/bold]")
    console.print("[dim]This stores your configuration, history, and cached data.[/dim]")
    console.print()
    
    write_permission = click.confirm(
        "Grant write permission to create this folder?",
        default=True
    )
    
    if not write_permission:
        console.print("[red]Write permission denied. Parishad cannot save settings.[/red]")
        return False
    
    # Create the directory & Gather Info
    try:
        parishad_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Gather System Info (Silent)
        from ..utils.hardware import get_system_info
        sys_info = get_system_info()
        
        # 2. Gather Model Inventory (Deep Silent Scan)
        from ..utils.scanner import ModelScanner
        scanner = ModelScanner()
        # Fast scan (Ollama/HF)
        found_models = scanner.scan_all()
        # Deep scan (Home dir)
        deep_models = scanner.scan_directory(Path.home())
        if deep_models:
            found_models.extend(deep_models)
        
        # Convert models to dict for JSON
        inventory = []
        for m in found_models:
            inventory.append({
                "name": m.name,
                "source": m.source,
                "size_gb": m.size_gb,
                "path": m.path
            })
        
        # 3. Create Enhanced Config
        config_file = parishad_dir / "config.json"
        initial_config = {
            "version": "0.1.0",
            "first_run_complete": True,
            "permissions": {
                "read": True,
                "write": True
            },
            "system": sys_info.to_dict(),
            "inventory": inventory
        }
        
        config_file.write_text(json.dumps(initial_config, indent=2))
        
        console.print(f"[green]✓ Created {parishad_dir}[/green]")
        console.print()
        console.print("[bold green]Setup complete! Starting Parishad...[/bold green]")
        console.print()
        return True
        
    except PermissionError:
        console.print(f"[red]Error: Cannot create {parishad_dir}. Permission denied by OS.[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Error creating directory: {e}[/red]")
        return False


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="parishad")
@click.pass_context
def cli(ctx):
    """
    🏛️ Parishad - एक लोकसभा-शैली LLM प्रणाली विश्वसनीय तर्क के लिए।
    
    पारिषद् लोकसभा (Parishad LokSabha) - A structured council of LLMs
    for reliable reasoning with budget tracking and systematic verification.
    
    Run 'parishad' without arguments to launch the interactive TUI.
    
    Commands:
      (no args) - Launch interactive TUI (with setup wizard on first run)
      run       - Execute a single query through the Sabha
      config    - View or modify configuration
      sthapana  - स्थापना (Setup) - Configure your Parishad Sabha
    """
    if ctx.invoked_subcommand is None:
        # First run - ask for permissions
        if is_first_run():
            if not first_run():
                # User declined permissions, exit
                sys.exit(0)
        
        # Launch unified TUI
        _launch_tui()


def _launch_tui(mode: Optional[str] = None):
    """
    Launch the unified Parishad TUI with first-time setup detection.
    
    Args:
        mode: Optional mode to start with ("fast"/"balanced"/"thorough")
    """
    from pathlib import Path
    
    # Check if first-time setup needed
    config_dir = Path.home() / ".config" / "parishad"
    config_file = config_dir / "config.json"
    
    if not config_file.exists():
        # First time - show setup wizard
        console.print("[dim]First time setup detected...[/dim]")
        # For now, go straight to TUI - setup wizard will be added in Phase 2
    
    # Launch the TUI
    from .code import run_code_cli
    
    try:
        run_code_cli(mode=mode)
    except KeyboardInterrupt:
        console.print("\n[yellow]Session ended[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option(
    "--config", "-c",
    type=click.Choice(["core", "extended", "fast"]),
    default=None,
    help="Pipeline configuration to use (overrides mode-based routing)"
)
@click.option(
    "--mode",
    type=click.Choice(["auto", "fast", "balanced", "thorough"]),
    default=None,
    help="Execution mode for adaptive routing (defaults to user config, fallback: balanced)"
)
@click.option(
    "--no-retry",
    is_flag=True,
    help="Disable Worker+Checker retry regardless of Checker verdict"
)
@click.option(
    "--model-config", "-m",
    type=click.Path(exists=True),
    help="Path to models.yaml configuration (defaults to ~/.parishad/models.yaml)"
)
@click.option(
    "--profile",
    type=str,
    default=None,
    help="Model profile to use (defaults to user config, fallback: local_cpu)"
)
@click.option(
    "--pipeline-config", "-p",
    type=click.Path(exists=True),
    help="Path to pipeline configuration YAML"
)
@click.option(
    "--trace-dir", "-t",
    type=click.Path(),
    help="Directory to save execution traces"
)
@click.option(
    "--mock",
    is_flag=True,
    help="Use mock models for testing (returns empty responses)"
)
@click.option(
    "--stub",
    is_flag=True,
    help="Use stub models with realistic role-specific responses"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show verbose output including role details"
)
@click.option(
    "--json-output",
    is_flag=True,
    help="Output results as JSON"
)
def run(
    query: str,
    config: Optional[str],
    mode: Optional[str],
    no_retry: bool,
    model_config: Optional[str],
    profile: Optional[str],
    pipeline_config: Optional[str],
    trace_dir: Optional[str],
    mock: bool,
    stub: bool,
    verbose: bool,
    json_output: bool
):
    """
    Run a query through the Parishad council.
    
    QUERY is the task or question to process.
    
    Execution modes control adaptive routing:
    - auto: Let Router decide based on task characteristics
    - fast: Minimal 3-role pipeline for simple queries
    - balanced: Standard 5-role pipeline (default)
    - thorough: Extended pipeline with specialized roles
    
    Profile and mode defaults are loaded from ~/.parishad/config.yaml.
    CLI flags override user defaults.
    
    Examples:
    
        parishad run "Write a Python function to compute fibonacci"
        
        parishad run --mode fast "What is 2+2?"
        
        parishad run --mode thorough "Explain quantum entanglement"
        
        parishad run --config extended --no-retry "Complex task"
        
        parishad run --mock "Test query"
        
        parishad run --stub "Test with realistic responses"
    """
    from ..config.user_config import load_user_config
    
    # Load user config for defaults
    user_cfg = load_user_config()
    
    # Handle stub/mock overrides first
    if stub:
        profile = "stub"
    elif mock:
        profile = "mock"
    elif profile is None:
        profile = user_cfg.default_profile
    
    if mode is None:
        mode = user_cfg.default_mode
    
    # Resolve config paths
    config_dir = get_config_dir()
    
    if not model_config:
        # When using stub/mock, always use package config which has those profiles
        if stub or mock:
            default_model_config = config_dir / "models.yaml"
            if default_model_config.exists():
                model_config = str(default_model_config)
        else:
            # For real models, check user config directory first
            user_model_config = Path.home() / ".parishad" / "models.yaml"
            if user_model_config.exists():
                model_config = str(user_model_config)
            else:
                # Fall back to package config
                default_model_config = config_dir / "models.yaml"
                if default_model_config.exists():
                    model_config = str(default_model_config)
    
    # Task 4: If user explicitly set --config, use it; otherwise let Router decide
    user_forced_config = config  # None if not set, or "core"/"extended"/"fast"
    if not config:
        config = "core"  # Default starting point for Router
    
    if not pipeline_config:
        if config == "core":
            default_pipeline = config_dir / "pipeline.core.yaml"
        elif config == "fast":
            default_pipeline = config_dir / "pipeline.fast.yaml"
        else:
            default_pipeline = config_dir / "pipeline.extended.yaml"
        if default_pipeline.exists():
            pipeline_config = str(default_pipeline)
    
    # Show progress
    if not json_output:
        # Use ASCII-safe title to avoid Windows encoding issues
        title = "Parishad LokSabha" if not UNICODE_SUPPORTED else "🏛️ पारिषद् लोकसभा"
        
        # Build subtitle with current settings
        subtitle_parts = []
        if user_forced_config:
            subtitle_parts.append(f"Config: {config}")
        else:
            subtitle_parts.append(f"Mode: {mode}")
        subtitle_parts.append(f"Profile: {profile}")
        if no_retry:
            subtitle_parts.append("no-retry")
        subtitle = " | ".join(subtitle_parts)
        
        console.print(Panel(
            f"[bold blue]प्रश्न (Query):[/bold blue] {query}",
            title=title,
            subtitle=subtitle
        ))
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=json_output
        ) as progress:
            task = progress.add_task("लोकसभा विचार-विमर्श जारी... (Sabha deliberating...)", total=None)
            
            # Task 4: Initialize with mode and no_retry parameters
            parishad = Parishad(
                config=config,
                model_config_path=model_config,
                profile=profile,
                pipeline_config_path=pipeline_config,
                trace_dir=trace_dir,
                mock=mock,
                stub=stub,
                mode=mode,
                user_forced_config=user_forced_config,
                no_retry=no_retry,
            )
            
            trace = parishad.run(query)
            
            progress.update(task, description="Complete!")
        
        # Output results
        if json_output:
            print(trace.to_json())
        else:
            display_results(trace, verbose)
            
    except Exception as e:
        if json_output:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def display_results(trace, verbose: bool = False):
    """Display execution results in a nice format."""
    # Final answer panel
    if trace.final_answer:
        answer = trace.final_answer.final_answer
        answer_type = trace.final_answer.answer_type
        confidence = trace.final_answer.confidence
        
        # Use ASCII-safe icons
        answer_icon = "Answer" if not UNICODE_SUPPORTED else "📝 Answer"
        code_icon = "Code" if not UNICODE_SUPPORTED else "📝 Code"
        
        # Format code nicely
        if answer_type == "code" and trace.final_answer.code_block:
            code = trace.final_answer.code_block
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            console.print(Panel(
                syntax,
                title=f"{code_icon}",
                subtitle=f"Confidence: {confidence:.0%}"
            ))
        else:
            console.print(Panel(
                answer,
                title=f"{answer_icon}",
                subtitle=f"Confidence: {confidence:.0%} | Type: {answer_type}"
            ))
        
        # Show caveats if any
        if trace.final_answer.caveats:
            console.print("\n[yellow]Caveats:[/yellow]")
            for caveat in trace.final_answer.caveats:
                console.print(f"  • {caveat}")
    
    # Summary table
    console.print()
    table = Table(title="Execution Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Query ID", trace.query_id[:8] + "...")
    table.add_row("Config", trace.config)
    table.add_row("Total Tokens", str(trace.total_tokens))
    table.add_row("Total Latency", f"{trace.total_latency_ms}ms")
    table.add_row("Budget Used", f"{trace.budget_initial - trace.budget_remaining}/{trace.budget_initial}")
    table.add_row("Retries", str(trace.retries))
    table.add_row("Success", "✓" if trace.success else "✗")
    
    console.print(table)
    
    # Verbose role details
    if verbose:
        console.print("\n[bold]Role Execution Details:[/bold]")
        
        role_table = Table()
        role_table.add_column("Role", style="cyan")
        role_table.add_column("Slot", style="magenta")
        role_table.add_column("Tokens", style="green")
        role_table.add_column("Latency", style="yellow")
        role_table.add_column("Status", style="blue")
        
        for role_output in trace.roles:
            role_table.add_row(
                role_output.role,
                role_output.metadata.slot.value if hasattr(role_output.metadata.slot, 'value') else str(role_output.metadata.slot),
                str(role_output.metadata.tokens_used),
                f"{role_output.metadata.latency_ms}ms",
                role_output.status
            )
        
        console.print(role_table)










@cli.command()
@click.argument("trace_file", type=click.Path(exists=True))
def inspect(trace_file: str):
    """
    Inspect an execution trace file.
    
    TRACE_FILE is the path to a trace JSON file.
    """
    with open(trace_file) as f:
        trace_data = json.load(f)
    
    console.print(Panel(
        f"[bold]Query ID:[/bold] {trace_data.get('query_id', 'unknown')}\n"
        f"[bold]Query:[/bold] {trace_data.get('user_query', 'unknown')[:100]}...",
        title="📋 Trace Inspection"
    ))
    
    # Show role outputs
    roles = trace_data.get("roles", [])
    for role_output in roles:
        role_name = role_output.get("role", "unknown")
        status = role_output.get("status", "unknown")
        output = role_output.get("output", {})
        
        console.print(f"\n[bold cyan]== {role_name.upper()} ==[/bold cyan]")
        console.print(f"Status: {status}")
        
        # Pretty print output
        if output:
            output_str = json.dumps(output, indent=2)[:1000]
            console.print(Syntax(output_str, "json", theme="monokai"))
    
    # Final answer
    final = trace_data.get("final_answer")
    if final:
        console.print("\n[bold green]== FINAL ANSWER ==[/bold green]")
        console.print(final.get("final_answer", "No answer"))


@cli.command()
def init():
    """
    Initialize Parishad configuration in the current directory.
    
    Creates a config/ directory with example configuration files.
    """
    config_dir = Path("./config")
    config_dir.mkdir(exist_ok=True)
    
    # Copy example configs
    package_config = get_config_dir()
    
    files_to_copy = [
        "models.example.yaml",
        "pipeline.core.yaml",
        "pipeline.extended.yaml"
    ]
    
    for filename in files_to_copy:
        src = package_config / filename
        dst = config_dir / filename
        
        if src.exists() and not dst.exists():
            dst.write_text(src.read_text())
            console.print(f"[green]Created:[/green] {dst}")
        elif dst.exists():
            console.print(f"[yellow]Skipped (exists):[/yellow] {dst}")
    
    console.print("\n[bold]Configuration initialized![/bold]")
    console.print("Edit config/models.example.yaml and rename to models.yaml")


@cli.command()
def info():
    """
    Show information about Parishad and available configurations.
    """
    console.print(Panel(
        "[bold]Parishad[/bold] - Cost-aware Council of LLMs\n\n"
        "A local-first system that orchestrates multiple LLMs into a structured\n"
        "council for reliable reasoning, coding, and factual correctness.\n\n"
        "[bold]Configurations:[/bold]\n"
        "  • core     - 5 roles: Refiner, Planner, Worker, Checker, Judge\n"
        "  • extended - 9 roles: Specialized variants of each role\n\n"
        "[bold]Model Slots:[/bold]\n"
        "  • small - 2-4B models for Refiner, Checker\n"
        "  • mid   - 7-13B models for Worker\n"
        "  • big   - 13-34B models for Planner, Judge",
        title="🏛️ Parishad Info"
    ))


# =============================================================================
# Model Management Commands
# =============================================================================


@cli.group()
def models():
    """
    Manage LLM models (download, list, remove).
    
    Download models from HuggingFace, Ollama, or LM Studio.
    """
    pass


@models.command("list")
@click.option(
    "--source", "-s",
    type=click.Choice(["all", "huggingface", "ollama", "lmstudio"]),
    default="all",
    help="Filter by source"
)
@click.option("--json-output", is_flag=True, help="Output as JSON")
def list_models(source: str, json_output: bool):
    """List downloaded models."""
    from ..models.downloader import ModelManager
    
    manager = ModelManager()
    
    # Scan for any unregistered models
    manager.scan_for_models()
    
    source_filter = source if source != "all" else None
    models = manager.list_models(source_filter)
    
    if json_output:
        print(json.dumps([m.to_dict() for m in models], indent=2, default=str))
        return
    
    if not models:
        console.print("[yellow]No models found.[/yellow]")
        console.print("Use [bold]parishad models download[/bold] to download models.")
        return
    
    table = Table(title="Downloaded Models")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Format", style="blue")
    table.add_column("Size", style="yellow")
    table.add_column("Quantization", style="magenta")
    
    for model in models:
        table.add_row(
            model.name,
            model.source.value,
            model.format.value,
            model.size_human,
            model.quantization or "-",
        )
    
    console.print(table)
    console.print(f"\n[dim]Model directory: {manager.model_dir}[/dim]")


@models.command("download")
@click.argument("model_name")
@click.option(
    "--source", "-s",
    type=click.Choice(["auto", "huggingface", "ollama", "lmstudio"]),
    default="auto",
    help="Source to download from"
)
@click.option(
    "--quantization", "-q",
    help="Preferred quantization (e.g., q4_k_m, q8_0)"
)
def download_model(model_name: str, source: str, quantization: Optional[str]):
    """
    Download a model.
    
    MODEL_NAME can be:
    
    \b
    - A shortcut: qwen2.5:1.5b, llama3.2:1b, phi3:mini
    - HuggingFace: owner/repo/file.gguf
    - Ollama: llama3.2:1b (requires Ollama installed)
    - LM Studio: path/to/model.gguf (from LM Studio's models dir)
    
    \b
    Examples:
        parishad models download qwen2.5:1.5b
        parishad models download llama3.2:1b --source ollama
        parishad models download TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf
    """
    from ..models.downloader import ModelManager, print_progress
    
    manager = ModelManager()
    
    # Check sources
    sources = manager.get_available_sources()
    
    console.print(f"\n[bold]Downloading:[/bold] {model_name}")
    console.print(f"[dim]Source: {source}[/dim]")
    
    if source == "ollama" and not sources["ollama"]:
        console.print("[red]Error:[/red] Ollama is not installed or not running.")
        console.print("Install from: https://ollama.ai")
        sys.exit(1)
    
    if source == "lmstudio" and not sources["lmstudio"]:
        console.print("[red]Error:[/red] LM Studio models directory not found.")
        sys.exit(1)
    
    try:
        from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
        
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Downloading {model_name}", total=None)
            
            def progress_callback(p):
                """Update Rich progress bar."""
                if p.total_bytes > 0:
                    progress.update(task, total=p.total_bytes, completed=p.downloaded_bytes)
            
            model = manager.download(
                model_name,
                source=source,
                quantization=quantization,
                progress_callback=progress_callback,
            )
        
        console.print(f"\n[green]✓ Downloaded:[/green] {model.name}")
        console.print(f"  [dim]Path: {model.path}[/dim]")
        console.print(f"  [dim]Size: {model.size_human}[/dim]")
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


@models.command("remove")
@click.argument("model_name")
@click.option("--keep-files", is_flag=True, help="Keep model files, only remove from registry")
def remove_model(model_name: str, keep_files: bool):
    """Remove a downloaded model."""
    from ..models.downloader import ModelManager
    
    manager = ModelManager()
    
    model = manager.registry.get(model_name)
    if not model:
        console.print(f"[red]Error:[/red] Model not found: {model_name}")
        console.print("Use [bold]parishad models list[/bold] to see available models.")
        sys.exit(1)
    
    # Confirm
    if not keep_files:
        console.print(f"[yellow]Warning:[/yellow] This will delete: {model.path}")
        if not click.confirm("Continue?"):
            console.print("Cancelled.")
            return
    
    if manager.remove_model(model_name, delete_files=not keep_files):
        console.print(f"[green]✓ Removed:[/green] {model_name}")
    else:
        console.print(f"[red]Error:[/red] Failed to remove model")


@models.command("available")
def available_models():
    """Show available model shortcuts for download."""
    from ..models.downloader import ModelManager
    
    manager = ModelManager()
    sources = manager.get_available_sources()
    
    console.print("\n[bold]Available Model Sources:[/bold]\n")
    
    table = Table()
    table.add_column("Source", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description")
    
    table.add_row(
        "HuggingFace",
        "✓ Available" if sources["huggingface"] else "✗",
        "Download GGUF models from HuggingFace Hub"
    )
    table.add_row(
        "Ollama",
        "✓ Available" if sources["ollama"] else "✗ Not installed",
        "Pull models via Ollama CLI"
    )
    table.add_row(
        "LM Studio",
        "✓ Available" if sources["lmstudio"] else "✗ Not found",
        "Import models from LM Studio"
    )
    
    console.print(table)
    
    console.print("\n[bold]Popular Model Shortcuts (HuggingFace):[/bold]\n")
    
    hf_table = Table()
    hf_table.add_column("Shortcut", style="cyan")
    hf_table.add_column("Repository")
    hf_table.add_column("Size")
    
    shortcuts = {
        "qwen2.5:0.5b": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "~400MB"),
        "qwen2.5:1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "~1GB"),
        "qwen2.5:3b": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "~2GB"),
        "qwen2.5:7b": ("Qwen/Qwen2.5-7B-Instruct-GGUF", "~5GB"),
        "llama3.2:1b": ("bartowski/Llama-3.2-1B-Instruct-GGUF", "~1GB"),
        "llama3.2:3b": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "~2GB"),
        "phi3:mini": ("microsoft/Phi-3-mini-4k-instruct-gguf", "~2.5GB"),
        "mistral:7b": ("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "~5GB"),
        "gemma2:2b": ("bartowski/gemma-2-2b-it-GGUF", "~1.5GB"),
    }
    
    for shortcut, (repo, size) in shortcuts.items():
        hf_table.add_row(shortcut, repo, size)
    
    console.print(hf_table)
    console.print("\n[dim]Usage: parishad models download qwen2.5:1.5b[/dim]")


@models.command("wizard")
def download_wizard():
    """Interactive model download wizard."""
    from ..models.downloader import ModelManager, interactive_download
    
    manager = ModelManager()
    
    try:
        model = interactive_download(manager)
        if model:
            console.print(f"\n[green]✓ Model ready:[/green] {model.name}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


# =============================================================================
# Setup Wizard Commands (prarambh and sthapana)
# =============================================================================

@cli.command("prarambh")
def prarambh():
    """
    🚀 Start your Parishad journey - Interactive session.
    
    'Prarambh' (प्रारम्भ) means 'beginning' in Sanskrit.
    
    This command:
    
    \b
    1. Loads existing council config (or runs setup if needed)
    2. Enters interactive mode for queries
    3. Processes questions through the council
    
    Example:
    
        parishad prarambh
    """
    from .prarambh import main as run_prarambh
    
    try:
        run_prarambh()
    except KeyboardInterrupt:
        console.print("\n[yellow]Session ended[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command("code")
@click.option(
    "--backend", "-b",
    default="ollama_native",
    help="Backend to use (ollama_native, openai, ollama, etc.)"
)
@click.option(
    "--model", "-m",
    default="llama3.2:3b",
    help="Model ID to use"
)
@click.option(
    "--cwd", "-d",
    default=None,
    help="Working directory (default: current)"
)
def code(backend: str, model: str, cwd: Optional[str]):
    """
    🤖 Interactive agentic coding assistant (like Claude Code).
    
    Start an interactive chat session where you can:
    
    \\b
    - Ask questions about code
    - Read and write files
    - Run shell commands
    - Get help with programming tasks
    
    The AI will use tools (file system, shell) to help you.
    
    Examples:
    
        parishad code
        
        parishad code --model llama3.2:3b
        
        parishad code --backend openai --model gpt-4o-mini
    """
    from .code import run_code_cli
    
    try:
        run_code_cli(backend=backend, model=model, cwd=cwd)
    except KeyboardInterrupt:
        console.print("\n[yellow]Session ended[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command("sthapana")
def sthapana():
    """
    🔧 Configure your Parishad council - Setup wizard.
    
    'Sthapana' (स्थापना) means 'establishment' in Sanskrit.
    
    This wizard guides you through:
    
    \b
    1. Choose a council configuration (Full/Medium/Minimal)
    2. Select models for each tier (Heavy/Mid/Light)
    3. Download required models
    4. Run health checks
    5. Generate pipeline configuration
    
    Example:
    
    	parishad sthapana
    """
    from .sthapana import main as run_sthapana
    
    try:
        run_sthapana()
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


# =============================================================================
# Configuration Commands
# =============================================================================

@cli.group("config")
def config_cmd():
    """⚙️ Manage Parishad configuration."""
    pass


@config_cmd.command("model-dir")
@click.argument("path", required=False)
def set_model_dir_cmd(path: Optional[str]):
    """
    View or set the model storage directory.
    
    Without arguments, shows the current model directory.
    With a path argument, sets the model directory.
    
    Examples:
    
    \b
        # View current directory
        parishad config model-dir
        
        # Set custom directory (Windows example)
        parishad config model-dir D:\\AI\\models
        
        # Set custom directory (macOS/Linux)
        parishad config model-dir /data/llm-models
    
    You can also set the PARISHAD_MODELS_DIR environment variable.
    """
    from ..models.downloader import (
        get_default_model_dir, 
        get_user_configured_model_dir,
        get_platform_default_model_dir,
        set_model_dir,
        PARISHAD_MODELS_DIR_ENV,
    )
    
    if path is None:
        # Show current configuration
        current_dir = get_default_model_dir()
        user_dir = get_user_configured_model_dir()
        platform_default = get_platform_default_model_dir()
        
        console.print("\n[bold]Model Directory Configuration[/bold]\n")
        console.print(f"  [cyan]Current:[/cyan]          {current_dir}")
        console.print(f"  [dim]Platform default:[/dim] {platform_default}")
        
        if user_dir:
            console.print(f"  [green]Custom (config):[/green]  {user_dir}")
        
        env_val = os.environ.get(PARISHAD_MODELS_DIR_ENV)
        if env_val:
            console.print(f"  [yellow]From env var:[/yellow]    {env_val}")
        
        console.print(f"\n[dim]To change: parishad config model-dir /your/path[/dim]")
        console.print(f"[dim]Or set: {PARISHAD_MODELS_DIR_ENV}=/your/path[/dim]\n")
    else:
        # Set new directory
        target_path = Path(path).resolve()
        
        # Validate path
        if target_path.exists() and not target_path.is_dir():
            console.print(f"[red]Error:[/red] {path} exists but is not a directory")
            sys.exit(1)
        
        # Create directory if needed
        try:
            target_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            console.print(f"[red]Error:[/red] Permission denied creating {path}")
            console.print("[dim]Try running with administrator/sudo privileges[/dim]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] Cannot create directory: {e}")
            sys.exit(1)
        
        # Save configuration
        set_model_dir(target_path)
        
        console.print(f"\n[green]✓[/green] Model directory set to: [cyan]{target_path}[/cyan]")
        console.print("[dim]New models will be downloaded to this location.[/dim]\n")


@config_cmd.command("show")
def show_config():
    """Show all Parishad configuration."""
    from ..models.downloader import get_config_file_path, get_default_model_dir
    from ..config.user_config import get_user_config_path, load_user_config
    
    config_file = get_config_file_path()
    user_config_path = get_user_config_path()
    
    console.print("\n[bold]Parishad Configuration[/bold]\n")
    
    # User Config (new)
    if user_config_path.exists():
        console.print("[bold cyan]User Config:[/bold cyan]")
        console.print(f"  [dim]Path:[/dim] {user_config_path}")
        try:
            user_cfg = load_user_config()
            console.print(f"  [cyan]Default Profile:[/cyan]  {user_cfg.default_profile}")
            console.print(f"  [cyan]Default Mode:[/cyan]     {user_cfg.default_mode}")
            console.print(f"  [cyan]Model Directory:[/cyan] {user_cfg.model_dir}")
        except Exception as e:
            console.print(f"  [red]Error loading:[/red] {e}")
        console.print()
    else:
        console.print(f"[yellow]User Config:[/yellow] Not found ({user_config_path})")
        console.print("  [dim]Run 'parishad sthapana' to create[/dim]\n")
    
    # Council Config
    council_config_path = Path.home() / ".parishad" / "council_config.json"
    if council_config_path.exists():
        console.print("[bold cyan]Council Config:[/bold cyan]")
        console.print(f"  [dim]Path:[/dim] {council_config_path}")
        try:
            with open(council_config_path) as f:
                council = json.load(f)
            console.print(f"  [cyan]Council:[/cyan]  {council['council']['name']}")
            console.print(f"  [cyan]Roles:[/cyan]    {council['council']['role_count']}")
            console.print(f"  [cyan]Size:[/cyan]     {council['total_size_gb']:.1f} GB")
        except Exception as e:
            console.print(f"  [red]Error loading:[/red] {e}")
        console.print()
    else:
        console.print(f"[yellow]Council Config:[/yellow] Not found ({council_config_path})\n")
    
    # Models Config
    models_config_path = Path.home() / ".parishad" / "models.yaml"
    if models_config_path.exists():
        console.print("[bold cyan]Models Config:[/bold cyan]")
        console.print(f"  [dim]Path:[/dim] {models_config_path}")
        console.print(f"  [green]✓[/green] Found")
        console.print()
    else:
        console.print(f"[yellow]Models Config:[/yellow] Not found ({models_config_path})\n")
    
    # Pipeline Config
    pipeline_config_path = Path.home() / ".parishad" / "pipeline.yaml"
    if pipeline_config_path.exists():
        console.print("[bold cyan]Pipeline Config:[/bold cyan]")
        console.print(f"  [dim]Path:[/dim] {pipeline_config_path}")
        console.print(f"  [green]✓[/green] Found")
        console.print()
    else:
        console.print(f"[yellow]Pipeline Config:[/yellow] Not found ({pipeline_config_path})\n")
    
    # Model Directory
    console.print("[bold cyan]Model Storage:[/bold cyan]")
    console.print(f"  [dim]Path:[/dim] {get_default_model_dir()}")
    
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            console.print(f"\n[bold]Config contents:[/bold]")
            console.print(json.dumps(config, indent=2))
        except Exception as e:
            console.print(f"[yellow]Could not read config: {e}[/yellow]")
    else:
        console.print(f"\n[dim]No config file found (using defaults)[/dim]")
    
    console.print("")


if __name__ == "__main__":
    cli()
