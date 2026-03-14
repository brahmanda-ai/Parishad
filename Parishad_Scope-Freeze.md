# SCOPE FREEZE — Parishad Research Extension
## This document is LAW. Do not deviate.

---

## Instructions for AI Assistants (Claude, Copilot, or any other)

You are working on the Parishad project.
You are acting as a **senior engineer** with the following traits:

### Engineering Standards

1. **You write production-level Python code.**
   - Type hints on every function signature
   - Docstrings on every public function and class
   - No magic numbers — use named constants
   - No bare except — always catch specific exceptions
   - Logging with `logging` module, not print statements
   - Functions do ONE thing
   - Files stay under 300 lines — split if longer

2. **You do not invent features.**
   - If it is not in this document, do not build it
   - If you think something should be added, ASK first
   - Do not refactor existing code unless explicitly told to
   - Do not rename files or restructure directories unless explicitly told to
   - Do not add dependencies unless explicitly approved

3. **You write debuggable code.**
   - Every function that can fail returns a clear error or raises a typed exception
   - Every experiment logs inputs, outputs, and metadata to JSON
   - Every model call logs: prompt length, output length, tokens used, time taken
   - No silent failures — if something goes wrong, it must be visible

4. **You write testable code.**
   - Business logic is separate from I/O
   - Functions take inputs and return outputs (no hidden state)
   - Side effects (file writes, model loads) are isolated in specific functions
   - Config is passed in, not read from global state

5. **You follow the existing codebase style.**
   - Parishad uses pydantic dataclasses
   - Parishad uses the `Role` base class pattern
   - Parishad uses `ModelRunner` for all model interactions
   - Do not introduce new patterns — use existing ones

---

## What This Project IS

This project is a **research extension** to Parishad that adds two features:

### Feature 1: Adaptive Pipeline Pruning

The system decides **which roles to run** for each query based on
the query's difficulty and type.

Easy queries use fewer roles.
Hard queries use all roles.

This saves tokens and reduces latency.

### Feature 2: 3-Model Heterogeneous Orchestration

The system assigns **different models to different roles** using 3 model slots:

- SMALL (1.5B–3B) for lightweight tasks
- MID (7B–8B) for main reasoning
- BIG (14B–32B) for hard final synthesis

This makes Parishad a genuine multi-model system.

### Feature 3: Benchmark Evaluation Harness

A set of scripts that run Parishad and baselines on standard benchmarks
(GSM8K, HumanEval) and produce structured results for paper writing.

---

## What This Project IS NOT

Do NOT build any of the following:

- ❌ Role distillation or fine-tuning
- ❌ Learned/ML-based routing (use rules only)
- ❌ SWE-Bench support
- ❌ BIG-Bench Hard support
- ❌ ALFWorld or WebArena support
- ❌ New TUI features or UI changes
- ❌ New slash commands
- ❌ New roles (10 roles exist, do not add more)
- ❌ New backends (4 backends exist, do not add more)
- ❌ API server or web interface
- ❌ Docker support
- ❌ CI/CD pipeline
- ❌ Model downloading (user handles this manually)
- ❌ Prompt optimization or prompt tuning
- ❌ RAG or retrieval integration for benchmarks
- ❌ Any form of training or gradient-based learning
- ❌ Changes to existing role prompts (unless fixing a clear bug)
- ❌ Changes to existing TUI code
- ❌ Changes to existing CLI code (except adding benchmark commands)

If you are about to write code for anything on this list, **STOP**.

---

## Detailed Explanation: Adaptive Pipeline Pruning

### What it is

Currently, Parishad runs a fixed pipeline for every query:
Darbari → Majumdar → Sainik → Prerak → Raja

text


This is wasteful because:
- easy queries do not need planning (Majumdar)
- simple outputs do not need checking (Prerak)
- the planner and checker burn tokens even when they add nothing

Adaptive pruning means the pipeline shape changes per query.

### How it works

#### Step 1: Darbari always runs first

Darbari analyzes the query and produces a `TaskSpec`:

```python
@dataclass
class TaskSpec:
    problem: str
    constraints: list[str]
    output_format: str        # code | text | numeric | structured
    difficulty_guess: str     # easy | medium | hard
    task_type: str            # math | code | qa | explanation
    key_concepts: list[str]
    safety_sensitivity: str   # low | medium | high
    expected_answer_length: str  # short | paragraph | long
Step 2: Router reads TaskSpec and selects roles
The router is a pure function that takes a TaskSpec and returns
an ordered list of (role_name, model_slot) tuples.

Python

def route(task_spec: TaskSpec) -> list[tuple[str, str]]:
    """
    Decide which roles to run and which model slot for each.
    
    This is a deterministic, rule-based function.
    No ML, no randomness, no side effects.
    """
Step 3: Orchestrator executes only the selected roles
The orchestrator receives the route and executes each role in order,
passing outputs from previous roles to subsequent ones.

Routing rules (THESE ARE THE EXACT RULES — do not change)
Python

ROUTING_RULES = {
    # Easy tasks: skip planner and checker
    "easy": {
        "any": ["darbari", "sainik", "raja"],
    },
    
    # Medium tasks: include planner OR checker, not both
    "medium": {
        "code": ["darbari", "majumdar", "sainik", "raja"],
        "math": ["darbari", "sainik", "prerak", "raja"],
        "qa":   ["darbari", "sainik", "prerak", "raja"],
        "explanation": ["darbari", "majumdar", "sainik", "raja"],
        "creative": ["darbari", "sainik", "raja"],
        "analysis": ["darbari", "majumdar", "sainik", "raja"],
    },
    
    # Hard tasks: full pipeline
    "hard": {
        "any": ["darbari", "majumdar", "sainik", "prerak", "raja"],
    },
}
What gets logged for every routed query
Python

@dataclass
class RoutingDecision:
    query_id: str
    difficulty: str           # from TaskSpec
    task_type: str            # from TaskSpec
    roles_selected: list[str]
    roles_skipped: list[str]
    routing_rule_used: str    # which rule matched
    timestamp: str
Data flow through pruned pipeline
The data flow works exactly as it does today, except skipped roles
produce no output. Downstream roles receive whatever is available.

text

If Majumdar is skipped:
  - Sainik receives: user_query + task_spec (no plan)
  - Sainik's prompt says "No plan was provided, work directly from task spec"

If Prerak is skipped:
  - Raja receives: user_query + task_spec + candidate (no verdict)
  - Raja's prompt says "No verification was performed"
This means role prompts need ONE small change each:
they must handle the case where upstream data is missing.

What pruning does NOT change
Role implementations (same code, same prompts with minor null handling)
Role output formats (same dataclasses)
Budget tracking (still tracks per-role tokens)
Error handling (same retry logic, but only for roles that ran)
Trace logging (still logs everything, including which roles were skipped)
Detailed Explanation: 3-Model Orchestration
What it is
Instead of using the same model for all roles, Parishad assigns
one of 3 model slots to each role:

text

SMALL → 1.5B–3B parameter model
         Fast, cheap, limited reasoning
         Good for: classification, refinement, simple checks

MID   → 7B–8B parameter model
         Balanced speed and capability
         Good for: planning, code generation, reasoning

BIG   → 14B–32B parameter model
         Slowest, most expensive, strongest reasoning
         Good for: complex synthesis, hard judgment calls
Default model-to-role mapping
Python

DEFAULT_MODEL_ASSIGNMENT = {
    "darbari":       "small",   # just classifying the query
    "majumdar":      "mid",     # planning needs moderate reasoning
    "sainik":        "mid",     # main work, needs good capability
    "prerak":        "small",   # checking can be lightweight
    "raja":          "mid",     # judge, usually mid is enough
}

# Override for hard tasks:
HARD_TASK_OVERRIDE = {
    "raja": "big",              # use big model for hard judgments
}

# Override for hard code tasks:
HARD_CODE_OVERRIDE = {
    "sainik": "big",            # use big model for hard code generation
    "raja":   "big",            # use big model for hard judgments
}
How models are managed in memory
The ModelRunner manages 3 model slots.

Python

class MultiModelRunner:
    """
    Manages up to 3 model slots.
    
    Loading strategy:
    - SMALL is always loaded (it is tiny, ~2 GB)
    - MID is loaded on first use and kept if VRAM allows
    - BIG is loaded on demand and unloaded after use if VRAM is tight
    
    The runner NEVER loads MID and BIG simultaneously on GPUs with <16 GB VRAM.
    On GPUs with >=24 GB VRAM or Apple Silicon with >=32 GB, all 3 can coexist.
    """
Model loading sequence for a hard query
text

1. SMALL is already loaded (always resident)
2. Darbari runs on SMALL
3. Router decides: this is hard → need MID and BIG
4. Load MID (if not loaded)
5. Majumdar runs on MID
6. Sainik runs on MID
7. Prerak runs on SMALL (already loaded)
8. Check VRAM: can BIG coexist with SMALL + MID?
   YES → load BIG, keep MID
   NO  → unload MID, then load BIG
9. Raja runs on BIG
10. Unload BIG (optional, depends on config)
Model loading sequence for an easy query
text

1. SMALL is already loaded
2. Darbari runs on SMALL
3. Router decides: this is easy → only need MID
4. Load MID (if not loaded)
5. Sainik runs on MID
6. Raja runs on MID
7. BIG is never loaded → saved VRAM and load time
What 3-model orchestration does NOT change
Role implementations (roles do not know which model they run on)
Role prompts (same prompts regardless of model size)
Role output formats (same dataclasses)
Pipeline structure (pruning decides structure, not model assignment)
Budget tracking (tokens counted same way regardless of model)
What 3-model orchestration DOES change
ModelRunner class (major change: manages 3 slots instead of 1)
ParishadEngine (minor change: passes slot name per role)
Config format (must specify model per slot)
Benchmark logging (must record which model was used per role)
Full System Anatomy — How Everything Works Together
Complete flow for a single query
text

USER QUERY: "Write a Python function to calculate compound interest"
    │
    ▼
┌─────────────────────────────────────────────────┐
│ STEP 1: PARSE INPUT                             │
│                                                 │
│ • Extract @mentions (file references)           │
│ • Load file contents into context               │
│ • Create initial RoleInput                      │
│ • Initialize Budget (8000 tokens default)       │
│ • Generate query_id (UUID)                      │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ STEP 2: DARBARI (always runs, always on SMALL)  │
│                                                 │
│ Model: SMALL (1.5B)                             │
│ Input: raw user query + file context            │
│ Output: TaskSpec                                │
│                                                 │
│ TaskSpec:                                       │
│   difficulty_guess: "easy"                      │
│   task_type: "code"                             │
│   output_format: "code"                         │
│   expected_answer_length: "short"               │
│                                                 │
│ Tokens used: ~400                               │
│ Time: ~1.5s                                     │
│                                                 │
│ LOG: {                                          │
│   role: "darbari",                              │
│   model_slot: "small",                          │
│   tokens_in: 89,                                │
│   tokens_out: 311,                              │
│   latency_ms: 1520,                             │
│   output: <TaskSpec object>                     │
│ }                                               │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ STEP 3: ADAPTIVE ROUTER                         │
│                                                 │
│ Input: TaskSpec                                 │
│ Logic:                                          │
│   difficulty == "easy"                          │
│   → use EASY route                              │
│   → roles: [darbari, sainik, raja]              │
│   → models: [small, mid, mid]                   │
│                                                 │
│ Output: PipelineConfig                          │
│   steps: [                                      │
│     ("sainik", "mid"),                           │
│     ("raja", "mid"),                             │
│   ]                                             │
│   # darbari already ran                         │
│                                                 │
│ LOG: {                                          │
│   routing_decision: "easy/code",                │
│   roles_selected: ["darbari","sainik","raja"],   │
│   roles_skipped: ["majumdar","prerak"],          │
│   models_assigned: {                            │
│     "darbari": "small",                         │
│     "sainik": "mid",                            │
│     "raja": "mid"                               │
│   }                                             │
│ }                                               │
│                                                 │
│ NO TOKENS USED (pure logic, no model call)      │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ STEP 4: ENSURE MODEL LOADED                     │
│                                                 │
│ Next role needs "mid" slot                      │
│ Check: is MID model loaded?                     │
│   NO → load Qwen2.5-7B into MID slot           │
│   YES → skip loading                            │
│                                                 │
│ VRAM check:                                     │
│   SMALL loaded: 2 GB                            │
│   MID loading: 5 GB                             │
│   Total: 7 GB                                   │
│   Available: 12 GB                              │
│   → OK, proceed                                 │
│                                                 │
│ LOG: {                                          │
│   model_loaded: "qwen2.5-7b",                   │
│   slot: "mid",                                  │
│   load_time_ms: 3200,                           │
│   vram_used_gb: 5.1                             │
│ }                                               │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ STEP 5: SAINIK (worker, on MID model)           │
│                                                 │
│ Model: MID (7B)                                 │
│ Input:                                          │
│   • user_query                                  │
│   • task_spec (from Darbari)                    │
│   • plan: NULL (Majumdar was skipped)           │
│   • file_context (if any @mentions)             │
│                                                 │
│ Prompt includes:                                │
│   "No plan was provided. Work directly from     │
│    the task specification."                     │
│                                                 │
│ Output: Candidate                               │
│   content: "def compound_interest(p, r, n, t):" │
│   content_type: "code"                          │
│   language: "python"                            │
│   confidence: 0.92                              │
│                                                 │
│ Tokens used: ~1400                              │
│ Time: ~4.8s                                     │
│                                                 │
│ LOG: {                                          │
│   role: "sainik",                                │
│   model_slot: "mid",                            │
│   model_name: "qwen2.5-7b",                    │
│   tokens_in: 340,                               │
│   tokens_out: 1060,                             │
│   latency_ms: 4820,                             │
│   upstream_data_available: ["task_spec"],        │
│   upstream_data_missing: ["plan"],              │
│   output_type: "code",                          │
│   self_confidence: 0.92                         │
│ }                                               │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ STEP 6: RAJA (judge, on MID model)              │
│                                                 │
│ Model: MID (7B) — same model, already loaded    │
│ Input:                                          │
│   • user_query                                  │
│   • task_spec (from Darbari)                    │
│   • candidate (from Sainik)                     │
│   • verdict: NULL (Prerak was skipped)          │
│                                                 │
│ Prompt includes:                                │
│   "No verification was performed.               │
│    Review the candidate directly."              │
│                                                 │
│ Output: FinalAnswer                             │
│   final_answer: "```python\ndef compound..."    │
│   answer_type: "code"                           │
│   confidence: 0.90                              │
│   rationale: "Code is correct..."               │
│                                                 │
│ Tokens used: ~900                               │
│ Time: ~3.1s                                     │
│                                                 │
│ LOG: {                                          │
│   role: "raja",                                 │
│   model_slot: "mid",                            │
│   model_name: "qwen2.5-7b",                    │
│   tokens_in: 580,                               │
│   tokens_out: 320,                              │
│   latency_ms: 3100,                             │
│   upstream_data_available:                      │
│     ["task_spec", "candidate"],                 │
│   upstream_data_missing:                        │
│     ["plan", "verdict"],                        │
│   final_confidence: 0.90                        │
│ }                                               │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ STEP 7: COMPILE RESULTS                         │
│                                                 │
│ Collect:                                        │
│   • final_answer from Raja                      │
│   • all role logs                               │
│   • routing decision                            │
│   • total tokens: 400 + 1400 + 900 = 2700      │
│   • total latency: 1.5 + 4.8 + 3.1 = 9.4s     │
│   • budget remaining: 8000 - 2700 = 5300       │
│   • roles run: 3 of 5                           │
│   • roles skipped: 2 of 5                       │
│   • models used: small + mid (big not loaded)   │
│                                                 │
│ Save to: results/{query_id}.json                │
│                                                 │
│ FULL TRACE:                                     │
│ {                                               │
│   "query_id": "a1b2c3",                         │
│   "query": "Write a Python function...",        │
│   "routing": { ... },                           │
│   "roles_executed": [                           │
│     { "role": "darbari", ... },                 │
│     { "role": "sainik", ... },                  │
│     { "role": "raja", ... }                     │
│   ],                                            │
│   "roles_skipped": ["majumdar", "prerak"],      │
│   "final_answer": "...",                        │
│   "total_tokens": 2700,                         │
│   "total_latency_ms": 9420,                     │
│   "models_used": ["small", "mid"],              │
│   "budget_remaining": 5300                      │
│ }                                               │
└─────────────────────────────────────────────────┘
Complete flow for a HARD query (different path)
text

USER QUERY: "This codebase has a race condition in the
             async handler. Find it and fix it.
             @src/handlers/auth.py @src/utils/db.py"
    │
    ▼
STEP 1: Parse input
  → extract 2 file references
  → read file contents
  → create context with file data

STEP 2: Darbari (SMALL)
  → difficulty: "hard"
  → task_type: "code"
  → safety_sensitivity: "high"

STEP 3: Router
  → hard + code → FULL pipeline
  → roles: [darbari, majumdar, sainik, prerak, raja]
  → models: [small, mid, mid, small, big]

STEP 4: Load MID model

STEP 5: Majumdar (MID)
  → creates plan with steps
  → "Step 1: analyze auth.py for async patterns"
  → "Step 2: identify shared state access"
  → etc.

STEP 6: Sainik (MID)
  → receives plan + file contents
  → generates fix with explanation
  → outputs code diff

STEP 7: Prerak (SMALL)
  → checks syntax of generated code
  → verifies the fix addresses the plan
  → flags any issues

STEP 8: Check VRAM for BIG
  → need to unload MID to fit BIG? depends on GPU
  → if yes: unload MID, load BIG
  → if no: load BIG alongside MID

STEP 9: Raja (BIG)
  → receives everything: task_spec, plan, candidate, verdict
  → synthesizes final answer with BIG model's stronger reasoning
  → higher quality judgment on hard task

STEP 10: Compile results
  → 5 roles ran
  → 3 models used (small, mid, big)
  → higher token cost but higher quality
Code Standards for This Project
File naming
text

All new files use snake_case.
All new files go in existing directories.
No new top-level directories without explicit approval.

New files that will be created:
  src/parishad/orchestrator/router.py          — adaptive router
  src/parishad/models/multi_runner.py          — 3-model manager
  src/parishad/eval/                           — new directory for benchmarks
  src/parishad/eval/__init__.py
  src/parishad/eval/gsm8k.py                  — GSM8K evaluation
  src/parishad/eval/humaneval.py               — HumanEval evaluation
  src/parishad/eval/runner.py                  — unified benchmark runner
  src/parishad/eval/metrics.py                 — metric calculations
  src/parishad/eval/baselines.py               — direct/CoT baselines
  scripts/experiment_zero.py                   — first experiment
  scripts/run_benchmarks.py                    — full benchmark suite
  scripts/analyze_results.py                   — result analysis + plots
Dataclass conventions
Python

# Use pydantic dataclasses for all new data structures
from pydantic import BaseModel, Field

class RoutingDecision(BaseModel):
    """Record of which roles were selected for a query."""
    
    query_id: str = Field(description="Unique query identifier")
    difficulty: str = Field(description="easy | medium | hard")
    task_type: str = Field(description="code | math | qa | ...")
    roles_selected: list[str] = Field(description="Roles that will run")
    roles_skipped: list[str] = Field(description="Roles that were pruned")
    model_assignments: dict[str, str] = Field(
        description="role_name -> model_slot mapping"
    )
    routing_rule: str = Field(description="Which rule was applied")
Logging conventions
Python

import logging

logger = logging.getLogger(__name__)

# Use structured logging for all experiment data
logger.info(
    "Role executed",
    extra={
        "role": role_name,
        "model_slot": slot,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": latency_ms,
    }
)
Error handling conventions
Python

# Define specific exceptions
class RouterError(Exception):
    """Raised when routing fails."""
    pass

class ModelSlotError(Exception):
    """Raised when a model slot cannot be loaded."""
    pass

class BudgetExceededError(Exception):
    """Raised when token budget is exceeded."""
    pass

# Always catch specific exceptions
try:
    result = model.generate(prompt)
except ModelSlotError as e:
    logger.error(f"Model slot {slot} failed: {e}")
    # fallback logic here
except BudgetExceededError:
    logger.warning("Budget exceeded, stopping pipeline")
    # return partial results
Scope Freeze Checklist
Before writing ANY code, verify:

 Is this feature in Scope-Freeze.md? If NO → do not build it
 Does this change existing role logic? If YES → get approval first
 Does this add a new dependency? If YES → get approval first
 Does this change the TUI or CLI? If YES → probably do not do it
 Does this file exceed 300 lines? If YES → split it
 Does every function have type hints? If NO → add them
 Does every public function have a docstring? If NO → add it
 Is there logging for debugging? If NO → add it
 Can this function be tested without a GPU? If NO → refactor
Models to Use for Experiments
Fixed model choices (do not change without discussion)
text

SMALL slot: Qwen2.5-1.5B-Instruct
  Format: GGUF Q4_K_M
  Source: HuggingFace or Ollama
  Size: ~1.2 GB

MID slot: Qwen2.5-7B-Instruct
  Format: GGUF Q4_K_M
  Source: HuggingFace or Ollama
  Size: ~4.7 GB

BIG slot: Qwen2.5-14B-Instruct
  Format: GGUF Q4_K_M
  Source: HuggingFace or Ollama
  Size: ~9.0 GB
Why Qwen2.5 family:

same tokenizer across sizes (fair comparison)
good performance at all sizes
well-supported in llama.cpp and Ollama
widely available
Do not switch model families mid-experiment.
If you want to test a different family, run it as a separate experiment.

For single-model experiments (Phase 1)
Use only: Qwen2.5-7B-Instruct (MID)
This is the baseline model for all pruning experiments.

For 3-model experiments (Phase 2)
Use all three: SMALL + MID + BIG as defined above.

Benchmarks (do not add more)
GSM8K
8.5K grade school math problems
Use test split (1319 problems)
For quick experiments: random 100-problem subset (fixed seed)
Metric: exact match on final numeric answer
Answer extraction: parse last number from model output
HumanEval
164 Python programming problems
Use full set (it is small enough)
For quick experiments: first 50 problems
Metric: pass@1 (does generated code pass unit tests)
Execution: run in subprocess with timeout
Do NOT add
MBPP (not enough value over HumanEval)
MATH (too hard for small models, noisy results)
SWE-Bench (requires massive infrastructure)
BIG-Bench (too diverse, hard to analyze)
Any custom benchmarks
Summary of Frozen Decisions
Decision	Frozen Value	Do Not Change
Number of models	3 (small, mid, big)	do not add 4th
Model family	Qwen2.5-Instruct	do not switch
Small model	1.5B	do not go smaller
Mid model	7B	do not change
Big model	14B	do not go to 32B
Quantization	Q4_K_M	keep consistent
Benchmarks	GSM8K + HumanEval	do not add more
Router type	rule-based	no ML router
Number of roles	5 (darbari→raja)	ignore 6–10
Routing rules	as defined above	tuning allowed
Backend for exps	Ollama or llama.cpp	pick one, stick with it
Random seed	42	use everywhere
Quick subset size	100 (GSM8K), 50 (HumanEval)	for dev only
Full eval size	1319 (GSM8K), 164 (HumanEval)	for paper
text


---