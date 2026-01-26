"""
Parishad Prarambh: The Launch Command.
Starts the council chat interface using the saved configuration.
"""
import sys
from pathlib import Path
from rich.console import Console
from parishad import Parishad

console = Console()
CONFIG_PATH = Path("models.yaml")

def main():
    console.print("[bold green]Parishad Prarambh[/bold green] - Starting the Council...")
    
    if not CONFIG_PATH.exists():
        console.print("[red]Error: Configuration file 'models.yaml' not found.[/red]")
        console.print("Please run [bold cyan]parishad sthapana[/bold cyan] first to set up your council.")
        sys.exit(1)
        
    try:
        # Initialize Parishad with the config
        council = Parishad(
            config="core", # Or extended, depending on Sabha choice? 
                           # Actually engine determines extended/core based on roles used?
                           # The engine default is 'core'. We might need to store this in config too?
                           # For now, let's default to 'core' or check roles.
            model_config_path=str(CONFIG_PATH),
            trace_dir="traces"
        )
        
        console.print("[dim]Council assembled. Ready for queries.[/dim]")
        console.print("Type 'exit' or 'quit' to leave.\n")
        
        while True:
            try:
                query = console.input("[bold cyan]You > [/bold cyan]")
                if query.lower() in ("exit", "quit"):
                    break
                if not query.strip():
                    continue
                    
                console.print("\n[dim]The Council is deliberating...[/dim]")
                
                trace = council.run(query)
                
                if trace.final_answer:
                    console.print("\n[bold green]Council Verdict:[/bold green]")
                    # Parse the output if it is JSON (PRP returns JSON string)
                    raw_answer = trace.final_answer.final_answer
                    import json
                    import re
                    
                    # Clean markdown code blocks
                    cleaned = re.sub(r'^```json\s*', '', raw_answer, flags=re.MULTILINE)
                    cleaned = re.sub(r'^```\s*', '', cleaned, flags=re.MULTILINE)
                    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
                    
                    try:
                        parsed = json.loads(cleaned)
                    except json.JSONDecodeError:
                        # Fallback for trailing commas or python-style dicts
                        try:
                            import ast
                            parsed = ast.literal_eval(cleaned)
                        except (ValueError, SyntaxError):
                            console.print(raw_answer)
                            continue

                    if isinstance(parsed, dict) and ("final_answer" in parsed or "final_answer" in str(parsed)):
                        # Handle potential key mismatch or direct access
                        answer = parsed.get("final_answer")
                        if answer:
                           console.print(answer)
                           
                        # Also print code block if present
                        if parsed.get("answer_type") == "code" and "code_block" in parsed:
                            from rich.syntax import Syntax
                            console.print(Syntax(parsed["code_block"], "python", theme="monokai", line_numbers=True))
                    else:
                        console.print(raw_answer)
                else:
                    console.print("\n[red]The Council could not reach a verdict.[/red]")
                    
                console.print(f"\n[dim]Tokens: {trace.total_tokens} | Latency: {trace.total_latency_ms}ms[/dim]\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}")
                
    except Exception as e:
        console.print(f"[red]Failed to initialize council:[/red] {e}")
        sys.exit(1)
        
    console.print("\n[bold]Council Adjourned.[/bold]")

if __name__ == "__main__":
    main()
