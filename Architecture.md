# Architecture — Technical Details of Changes

---

## System Architecture After All Changes
┌─────────────────────────────────────────────────┐
│ USER QUERY │
└──────────────────────┬──────────────────────────┘
│
▼
┌─────────────────────────────────────────────────┐
│ PARISHAD ENGINE │
│ │
│ 1. Parse input (extract @mentions, context) │
│ 2. Run Darbari on SMALL model → TaskSpec │
│ 3. Call AdaptiveRouter(TaskSpec) │
│ → PipelineConfig (roles + model slots) │
│ 4. Execute selected roles in order │
│ 5. Each role uses assigned model slot │
│ 6. Compile results + traces │
│ 7. Return FinalAnswer │
└──────────┬─────────────┬────────────────────────┘
│ │
▼ ▼
┌──────────────┐ ┌──────────────────────────────┐
│ ADAPTIVE │ │ MULTI-MODEL RUNNER │
│ ROUTER │ │ │
│ │ │ ┌───────┐ ┌─────┐ ┌─────┐ │
│ TaskSpec │ │ │ SMALL │ │ MID │ │ BIG │ │
│ ↓ │ │ │ 1.5B │ │ 7B │ │ 14B │ │
│ Routing │ │ └───────┘ └─────┘ └─────┘ │
│ Rules │ │ │
│ ↓ │ │ • ensure_loaded(slot) │
│ PipelineConfig│ │ • generate(prompt, slot) │
│ │ │ • unload(slot) │
└──────────────┘ └──────────────────────────────┘

text


---

## New Components

### 1. AdaptiveRouter
Location: src/parishad/orchestrator/router.py
(or src/parishad_research/router/adaptive.py if separate repo)

Depends on: TaskSpec (from roles/darbari.py)
Used by: ParishadEngine (orchestrator/engine.py)

text


```python
class AdaptiveRouter:
    """
    Decides which roles to execute and which model slot for each.
    
    This is a stateless, rule-based router.
    It takes a TaskSpec and returns a PipelineConfig.
    
    No ML, no training, no state between queries.
    """
    
    def __init__(self, rules: dict | None = None):
        """
        Args:
            rules: Optional custom routing rules.
                   If None, uses DEFAULT_ROUTING_RULES.
        """
    
    def route(self, task_spec: TaskSpec) -> PipelineConfig:
        """
        Select roles and model slots for this task.
        
        Args:
            task_spec: Output from Darbari role.
            
        Returns:
            PipelineConfig with ordered list of (role, slot) tuples.
            
        The returned config does NOT include Darbari
        (it has already run before routing).
        """
    
    def _match_rule(
        self, difficulty: str, task_type: str
    ) -> tuple[list[str], dict[str, str]]:
        """
        Find the matching routing rule.
        
        Returns:
            Tuple of (role_list, model_assignment_dict)
        """
2. PipelineConfig
Python

class PipelineConfig(BaseModel):
    """Configuration for a single query's pipeline execution."""
    
    steps: list[PipelineStep] = Field(
        description="Ordered list of roles to execute"
    )
    routing_rule: str = Field(
        description="Which routing rule was applied"
    )
    difficulty: str = Field(
        description="Difficulty from TaskSpec"
    )
    task_type: str = Field(
        description="Task type from TaskSpec"
    )

class PipelineStep(BaseModel):
    """A single step in the pipeline."""
    
    role_name: str = Field(description="e.g. 'sainik'")
    model_slot: str = Field(description="e.g. 'mid'")
3. MultiModelRunner
text

Location: src/parishad/models/multi_runner.py
          (or src/parishad_research/multi_model/runner.py if separate repo)

Depends on: existing ModelBackend protocol (models/backends/base.py)
Used by: ParishadEngine
Python

class MultiModelRunner:
    """
    Manages 3 model slots: small, mid, big.
    
    Each slot can hold one loaded model.
    Models are loaded on demand and cached.
    VRAM is managed to prevent OOM.
    
    Thread-safety: uses a threading.Lock for model loading.
    This is important because roles may request different
    models in sequence, and we must not have race conditions.
    """
    
    def __init__(
        self,
        slot_configs: dict[str, SlotConfig],
        max_vram_gb: float | None = None,
    ):
        """
        Args:
            slot_configs: Config for each slot (small, mid, big).
                         Each config has: model_id, backend, quantization, etc.
            max_vram_gb: Maximum VRAM to use. If None, auto-detect.
        """
    
    def ensure_loaded(self, slot: str) -> None:
        """
        Ensure the model for this slot is loaded and ready.
        
        If already loaded, this is a no-op.
        If not loaded, load it, potentially unloading another model first.
        
        Raises:
            ModelSlotError: If model cannot be loaded.
        """
    
    def generate(
        self,
        prompt: str,
        slot: str,
        max_tokens: int = 1024,
        temperature: float = 0.5,
        stop: list[str] | None = None,
    ) -> BackendResult:
        """
        Generate text using the model in the specified slot.
        
        Args:
            prompt: The full prompt to send.
            slot: Which model slot to use (small/mid/big).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop: Stop sequences.
            
        Returns:
            BackendResult with generated text and metadata.
            
        Raises:
            ModelSlotError: If slot is not loaded and cannot be loaded.
            BudgetExceededError: If token budget would be exceeded.
        """
    
    def unload(self, slot: str) -> None:
        """
        Unload a model to free VRAM.
        
        Args:
            slot: Which slot to unload.
            
        Note: Refuses to unload "small" slot (it should always be loaded).
        """
    
    def get_loaded_slots(self) -> list[str]:
        """Return list of currently loaded slot names."""
    
    def get_vram_usage(self) -> dict[str, float]:
        """Return estimated VRAM usage per loaded slot."""
4. Enhanced Tracing
Python

class RoleTrace(BaseModel):
    """Trace data for a single role execution."""
    
    role_name: str
    model_slot: str
    model_name: str
    tokens_in: int
    tokens_out: int
    total_tokens: int
    latency_ms: int
    success: bool
    error: str | None = None
    output_summary: str = ""  # first 200 chars of output
    
class PipelineTrace(BaseModel):
    """Complete trace for one query through the pipeline."""
    
    query_id: str
    query: str
    timestamp: str
    routing_decision: RoutingDecision
    role_traces: list[RoleTrace]
    roles_skipped: list[str]
    total_tokens: int
    total_latency_ms: int
    final_answer: str
    final_confidence: float
    models_used: list[str]      # unique model names used
    budget_total: int
    budget_remaining: int
Data Flow Diagrams
Easy query data flow
text

User Query
  │
  ├──► Darbari (SMALL)
  │      Input:  query
  │      Output: TaskSpec {difficulty: "easy", type: "code"}
  │
  ├──► Router
  │      Input:  TaskSpec
  │      Output: [("sainik","mid"), ("raja","mid")]
  │      Skipped: majumdar, prerak
  │
  ├──► Sainik (MID)
  │      Input:  query + task_spec
  │      Missing: plan (null), no file context
  │      Output: Candidate {code, confidence}
  │
  └──► Raja (MID)
         Input:  query + task_spec + candidate
         Missing: plan (null), verdict (null)
         Output: FinalAnswer
Hard query data flow
text

User Query + @file1 + @file2
  │
  ├──► Darbari (SMALL)
  │      Input:  query + file_context
  │      Output: TaskSpec {difficulty: "hard", type: "code"}
  │
  ├──► Router
  │      Input:  TaskSpec
  │      Output: [("majumdar","mid"), ("sainik","mid"),
  │               ("prerak","small"), ("raja","big")]
  │      Skipped: none
  │
  ├──► Majumdar (MID)
  │      Input:  query + task_spec + file_context
  │      Output: Plan {steps, dependencies}
  │
  ├──► Sainik (MID)
  │      Input:  query + task_spec + plan + file_context
  │      Output: Candidate {code, reasoning_trace}
  │
  ├──► Prerak (SMALL)
  │      Input:  task_spec + plan + candidate
  │      Output: Verdict {flags, must_fix, confidence}
  │
  │  [If must_fix and budget allows: retry Sainik]
  │
  └──► Raja (BIG)  ← loaded on demand, MID may be unloaded
         Input:  query + task_spec + plan + candidate + verdict
         Output: FinalAnswer
Files Changed vs Files Created
Files CREATED (new code)
text

NEW  src/parishad/orchestrator/router.py
NEW  src/parishad/models/multi_runner.py
NEW  src/parishad/eval/__init__.py
NEW  src/parishad/eval/gsm8k.py
NEW  src/parishad/eval/humaneval.py
NEW  src/parishad/eval/metrics.py
NEW  src/parishad/eval/baselines.py
NEW  scripts/experiment_zero.py
NEW  scripts/run_pruning.py
NEW  scripts/run_ablations.py
NEW  scripts/run_multi_model.py
NEW  scripts/run_full_benchmarks.py
NEW  scripts/analyze_results.py
NEW  scripts/generate_figures.py
NEW  tests/test_router.py
NEW  tests/test_multi_runner.py
NEW  tests/test_gsm8k_eval.py
NEW  configs/models.yaml
NEW  configs/routing_rules.yaml
Files MODIFIED (minimal changes to existing code)
text

MOD  src/parishad/orchestrator/engine.py
     — add router call after Darbari
     — add model slot parameter to role execution
     — add PipelineTrace creation
     Change size: ~50-80 lines added

MOD  src/parishad/roles/sainik.py
     — handle null plan in format_input()
     Change size: ~5-10 lines added

MOD  src/parishad/roles/raja.py
     — handle null plan and null verdict in format_input()
     Change size: ~10-15 lines added

MOD  src/parishad/roles/base.py
     — add timing and token logging to execute()
     Change size: ~15-20 lines added

MOD  src/parishad/utils/tracing.py
     — add RoleTrace and PipelineTrace dataclasses
     Change size: ~30-40 lines added
Files NOT TOUCHED (leave alone)
text

UNTOUCHED  src/parishad/cli/*
UNTOUCHED  src/parishad/tools/*
UNTOUCHED  src/parishad/checker/*
UNTOUCHED  src/parishad/config/*
UNTOUCHED  src/parishad/roles/darbari.py
UNTOUCHED  src/parishad/roles/majumdar.py
UNTOUCHED  src/parishad/roles/prerak.py
UNTOUCHED  src/parishad/roles/sar_senapati.py
UNTOUCHED  src/parishad/roles/sacheev.py
UNTOUCHED  src/parishad/roles/dandadhyaksha.py
UNTOUCHED  src/parishad/roles/pantapradhan.py
UNTOUCHED  src/parishad/roles/vidushak.py
UNTOUCHED  src/parishad/models/backends/*
UNTOUCHED  src/parishad/models/profiles.py
UNTOUCHED  src/parishad/models/downloader.py