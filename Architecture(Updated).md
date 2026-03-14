Architecture.md (Updated)
Markdown

# Architecture — Technical Design Reference

---

## ⚠️ IMPORTANT: This Document Describes FUTURE STATE

This architecture describes what Parishad will look like
AFTER Experiment 0 proves the pipeline adds value.

**Do NOT implement anything from this document until:**
1. Experiment 0 is complete
2. Results show pipeline improves accuracy by 5%+ over direct prompting
3. The decision gate in Decision-Gates.md is explicitly passed

**Before Experiment 0:**
- Read this document for understanding only
- Do NOT create router.py
- Do NOT create multi_runner.py
- Do NOT modify engine.py
- Do NOT modify any role files

**After Experiment 0 passes:**
- Use this document as the implementation blueprint
- Follow Phases.md for task ordering
- Follow Scope-Freeze.md for what is allowed

---

## Current Architecture (What Exists Today)
USER QUERY
│
▼
┌─────────────────────────────────────────┐
│ PARISHAD ENGINE │
│ │
│ Loads pipeline config from YAML │
│ Runs ALL roles in fixed order │
│ Uses SAME model for every role │
│ Tracks budget (tokens used) │
│ Saves trace to output.json │
└────────────────┬────────────────────────┘
│
┌────────────┼────────────────┐
│ │ │
▼ ▼ ▼
┌────────┐ ┌──────────┐ ┌──────────┐
│Darbari │→│Majumdar │→ │ Sainik │→ ...→ Raja
│ │ │ │ │ │
│ SAME │ │ SAME │ │ SAME │
│ MODEL │ │ MODEL │ │ MODEL │
└────────┘ └──────────┘ └──────────┘

Problems with current architecture:

Every query runs ALL roles (wasteful for easy tasks)
Same model for all roles (no heterogeneity)
No way to measure if roles actually help
No benchmark evaluation capability
text


### Key files in current system
src/parishad/
├── orchestrator/
│ └── engine.py ← main orchestration loop
│ runs roles in order from YAML config
│ manages budget
│ handles retries
│
├── roles/
│ ├── base.py ← Role ABC with execute() method
│ ├── darbari.py ← refiner role
│ ├── majumdar.py ← planner role
│ ├── sainik.py ← worker role
│ ├── prerak.py ← checker role
│ └── raja.py ← judge role
│
├── models/
│ ├── runner.py ← ModelRunner (single model)
│ └── backends/
│ ├── base.py ← BackendConfig, BackendResult
│ ├── llama_cpp.py ← GGUF backend
│ ├── ollama.py ← Ollama API backend
│ ├── mlx_lm.py ← Apple Silicon backend
│ └── huggingface.py ← Transformers backend
│
├── cli/
│ ├── main.py ← CLI entry point
│ └── code.py ← TUI application (Textual)
│
└── config/
├── pipeline.fast.yaml ← Laghu Sabha config
├── pipeline.core.yaml ← Madhyam Sabha config
└── pipeline.extended.yaml ← Maha Sabha config

text


### How engine.py works today (simplified)

```python
# Simplified current flow in engine.py

class ParishadEngine:
    def __init__(self, config, model_runner):
        self.config = config          # pipeline YAML config
        self.model_runner = model_runner  # single ModelRunner
        self.budget = Budget()
    
    def run(self, query: str, context: dict = None) -> dict:
        results = {}
        
        # Always runs ALL roles from config in order
        for role_name in self.config.pipeline:
            role = self.roles[role_name]
            
            # Every role uses the same model_runner
            role_input = self._build_input(query, context, results)
            role_output = role.execute(role_input, self.model_runner)
            
            results[role_name] = role_output
            self.budget.spend(role_output.tokens_used)
        
        return results
Future Architecture — Phase 1: Adaptive Pruning
Implement ONLY after Experiment 0 passes (see Decision-Gates.md)

What changes
text

USER QUERY
    │
    ▼
┌─────────────────────────────────────────┐
│           PARISHAD ENGINE               │
│                                         │
│  1. Run Darbari (always first)          │
│  2. Call AdaptiveRouter(TaskSpec)   ← NEW│
│     → decides which roles to run        │
│  3. Run ONLY selected roles             │
│  4. Track budget per role          ← NEW│
│  5. Save enhanced trace            ← NEW│
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         ADAPTIVE ROUTER            NEW  │
│                                         │
│  Input: TaskSpec from Darbari           │
│  Output: list of roles to execute       │
│                                         │
│  Rules:                                 │
│    easy → [darbari, sainik, raja]        │
│    medium → [darbari, +planner OR       │
│              +checker, sainik, raja]     │
│    hard → [all 5 roles]                 │
└─────────────────────────────────────────┘
New file: orchestrator/router.py
Python

# This file does NOT exist yet.
# Create it ONLY after Experiment 0 passes.

class AdaptiveRouter:
    """
    Stateless, rule-based router.
    Takes TaskSpec, returns list of roles to execute.
    No ML, no training, no side effects.
    Pure function.
    """
    
    def route(self, task_spec: TaskSpec) -> list[str]:
        """Returns ordered list of role names to execute."""
        pass
Changes to engine.py (Phase 1)
Python

# BEFORE (current):
for role_name in self.config.pipeline:
    # runs ALL roles

# AFTER (Phase 1):
darbari_output = self.run_role("darbari", query)
task_spec = darbari_output.task_spec

# NEW: router decides which roles to run
selected_roles = self.router.route(task_spec)

for role_name in selected_roles:
    if role_name == "darbari":
        continue  # already ran
    # run only selected roles
Changes to role files (Phase 1)
Only two files need small changes:

sainik.py — handle missing plan:

Python

# In format_input():
if plan is None:
    prompt += "No execution plan was provided. Work directly from the task specification.\n"
raja.py — handle missing plan and verdict:

Python

# In format_input():
if plan is None:
    prompt += "No execution plan was created.\n"
if verdict is None:
    prompt += "No verification was performed. Review the candidate output directly.\n"
New file: utils/tracing.py additions
Python

# Add these dataclasses to existing tracing.py

class RoleTrace(BaseModel):
    role_name: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    was_skipped: bool
    model_slot: str  # for Phase 2

class PipelineTrace(BaseModel):
    query_id: str
    roles_executed: list[RoleTrace]
    roles_skipped: list[str]
    routing_decision: str
    total_tokens: int
    total_latency_ms: int
Future Architecture — Phase 2: 3-Model Orchestration
Implement ONLY after Phase 1 pruning experiments show positive results

What changes on top of Phase 1
text

USER QUERY
    │
    ▼
┌─────────────────────────────────────────┐
│           PARISHAD ENGINE               │
│                                         │
│  1. Run Darbari on SMALL model     ← NEW│
│  2. Call Router(TaskSpec)                │
│     → decides roles AND model slots← NEW│
│  3. Run selected roles on assigned      │
│     model slots                    ← NEW│
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│       MULTI-MODEL RUNNER           NEW  │
│                                         │
│  3 model slots:                         │
│  ┌───────┐  ┌───────┐  ┌───────┐      │
│  │ SMALL │  │  MID  │  │  BIG  │      │
│  │ 1.5B  │  │  7B   │  │  14B  │      │
│  │always │  │on-    │  │on-    │      │
│  │loaded │  │demand │  │demand │      │
│  └───────┘  └───────┘  └───────┘      │
│                                         │
│  Manages loading/unloading per VRAM     │
└─────────────────────────────────────────┘
New file: models/multi_runner.py
Python

# This file does NOT exist yet.
# Create it ONLY after Phase 1 shows pruning works.

class MultiModelRunner:
    """
    Manages 3 model slots: small, mid, big.
    
    SMALL is always loaded (~2 GB).
    MID is loaded on demand and cached.
    BIG is loaded on demand, may require unloading MID.
    
    Thread-safe model loading with threading.Lock.
    """
    
    def ensure_loaded(self, slot: str) -> None: ...
    def generate(self, prompt: str, slot: str, **kwargs) -> BackendResult: ...
    def unload(self, slot: str) -> None: ...
Router changes for Phase 2
Python

# Phase 1 router returns:
["darbari", "sainik", "raja"]

# Phase 2 router returns:
[("darbari", "small"), ("sainik", "mid"), ("raja", "mid")]

# The return type changes from list[str] to list[tuple[str, str]]
Engine changes for Phase 2
Python

# Phase 1:
for role_name in selected_roles:
    result = self.run_role(role_name, input, model_runner=self.model_runner)

# Phase 2:
for role_name, model_slot in selected_roles:
    self.multi_runner.ensure_loaded(model_slot)
    result = self.run_role(role_name, input, model_slot=model_slot)
VRAM management strategy
text

Hardware: 8-16 GB VRAM
  SMALL always loaded:     2 GB
  MID loaded on demand:    5 GB
  BIG loaded on demand:   10 GB
  
  Rule: never MID + BIG simultaneously
  Sequence: unload MID → load BIG → run → unload BIG → load MID

Hardware: 24+ GB VRAM or Apple Silicon 32+ GB
  All three can be loaded simultaneously
  No swapping needed
Files Changed vs Created — Summary
Experiment 0 (before any architecture changes)
text

CREATED (new, no impact on existing code):
  src/parishad/eval/__init__.py
  src/parishad/eval/gsm8k.py
  src/parishad/eval/humaneval.py
  src/parishad/eval/metrics.py
  src/parishad/eval/baselines.py
  scripts/experiment_zero.py

MODIFIED: NOTHING
  Zero changes to existing Parishad code
Phase 1 (after Experiment 0 passes)
text

CREATED:
  src/parishad/orchestrator/router.py
  scripts/run_pruning.py
  scripts/run_ablations.py
  scripts/analyze_results.py
  tests/test_router.py

MODIFIED (minimal, surgical changes):
  src/parishad/orchestrator/engine.py    (~50 lines added)
  src/parishad/roles/sainik.py           (~5 lines added)
  src/parishad/roles/raja.py             (~10 lines added)
  src/parishad/roles/base.py             (~15 lines added)
  src/parishad/utils/tracing.py          (~30 lines added)
Phase 2 (after Phase 1 shows pruning works)
text

CREATED:
  src/parishad/models/multi_runner.py
  scripts/run_multi_model.py
  tests/test_multi_runner.py

MODIFIED:
  src/parishad/orchestrator/engine.py    (~30 more lines)
  src/parishad/orchestrator/router.py    (~20 lines changed)
  configs/models.yaml                    (new config file)
NEVER TOUCHED (across all phases)
text

src/parishad/cli/*              ← TUI stays as-is
src/parishad/tools/*            ← tools stay as-is
src/parishad/checker/*          ← checker stays as-is
src/parishad/config/modes.py    ← modes stay as-is
src/parishad/config/pipeline.*.yaml ← existing configs stay
src/parishad/roles/darbari.py   ← no changes
src/parishad/roles/majumdar.py  ← no changes
src/parishad/roles/prerak.py    ← no changes
src/parishad/roles/sar_senapati.py
src/parishad/roles/sacheev.py
src/parishad/roles/dandadhyaksha.py
src/parishad/roles/pantapradhan.py
src/parishad/roles/vidushak.py
src/parishad/models/backends/*  ← backends stay as-is
src/parishad/models/profiles.py
src/parishad/models/downloader.py
Dependency Between Phases
text

Experiment 0
    │
    │ requires: ZERO changes to Parishad
    │ only adds: eval/ scripts
    │
    ▼
[DECISION GATE 1: Does pipeline help?]
    │
    │ YES
    ▼
Phase 1: Adaptive Pruning
    │
    │ requires: router.py (new file)
    │           small changes to engine.py, sainik.py, raja.py
    │
    ▼
[DECISION GATE 2: Does pruning preserve quality?]
    │
    │ YES
    ▼
Phase 2: 3-Model Orchestration
    │
    │ requires: multi_runner.py (new file)
    │           changes to engine.py and router.py
    │
    ▼
[DECISION GATE 3: Does multi-model improve quality-per-token?]
    │
    │ YES
    ▼
Phase 3: Full Benchmarks + Paper
Each phase is independent.
Each phase has a decision gate.
You can stop at any gate without having wasted work on later phases.