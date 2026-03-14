# Phases — Detailed Task Breakdown

---

## Overview
Phase 0: Experiment Zero (kill switch) — Week 1
Phase 1: Adaptive Pipeline Pruning — Week 2–3
Phase 2: 3-Model Heterogeneous Orchestration — Week 4–5
Phase 3: Full Benchmarking + Ablations — Week 5–6
Phase 4: Analysis + Paper — Week 7–8

text


Each phase has:
- clear deliverables
- a go/no-go decision at the end
- specific files to create or modify
- specific tests to run

---

## Phase 0 — Experiment Zero

### Duration: Week 1 (5–7 days)

### Purpose
Answer ONE question: does the Parishad pipeline improve output quality
over direct single-shot prompting with the same model?

### Go/No-Go
This phase has a KILL SWITCH.
If the pipeline does not help, the entire research direction stops.

### Tasks

#### Task 0.1: Set up evaluation environment
- Create `src/parishad/eval/` directory
- Create `src/parishad/eval/__init__.py`
- Create `src/parishad/eval/gsm8k.py`
  - function to load GSM8K dataset (from HuggingFace datasets)
  - function to extract ground truth numeric answer
  - function to extract predicted numeric answer from model output
  - function to compare (exact match)
- Create `src/parishad/eval/humaneval.py`
  - function to load HumanEval dataset
  - function to execute generated code in sandbox (subprocess + timeout)
  - function to check pass/fail
- Create `src/parishad/eval/metrics.py`
  - accuracy calculation
  - token counting
  - latency tracking
  - results aggregation

**Estimated effort:** 1–2 days

#### Task 0.2: Create baseline runner
- Create `src/parishad/eval/baselines.py`
  - `run_direct(model, prompt) -> result`
    - single model call, no pipeline
    - log tokens, latency, output
  - `run_direct_cot(model, prompt) -> result`
    - prepend "Let's think step by step" to prompt
    - single model call
    - log tokens, latency, output

**Estimated effort:** 0.5 day

#### Task 0.3: Create experiment_zero script
- Create `scripts/experiment_zero.py`
  - load 100 GSM8K problems (fixed seed=42 subset)
  - load 50 HumanEval problems
  - run each through:
    - Setup A: direct single-shot (7B model)
    - Setup B: direct with CoT (7B model)
    - Setup C: current Parishad pipeline (7B model, all roles)
  - save results to `results/experiment_zero/`
  - print summary table

**Estimated effort:** 1 day

#### Task 0.4: Run experiment and analyze
- Run the script
- Compare accuracy across setups
- Document findings

**Estimated effort:** 1–2 days (depends on hardware speed)

### Deliverables
- `src/parishad/eval/` directory with gsm8k.py, humaneval.py, metrics.py, baselines.py
- `scripts/experiment_zero.py`
- `results/experiment_zero/summary.json`
- Decision: CONTINUE or STOP

### Decision Criteria
Pipeline accuracy > direct by 8%+
→ GREEN LIGHT: proceed to Phase 1

Pipeline accuracy > direct by 3–7%
→ YELLOW: proceed cautiously, expect modest results

Pipeline accuracy within ±2% of direct
→ RED: investigate why pipeline adds no value
→ try adjusting role prompts
→ if still no improvement after 2 days debugging → STOP

Pipeline accuracy < direct
→ HARD STOP: pipeline actively hurts
→ pivot to product direction

text


---

## Phase 1 — Adaptive Pipeline Pruning

### Duration: Week 2–3 (10–14 days)

### Purpose
Implement the adaptive router and show that pruning saves tokens
while preserving most of the pipeline's quality benefit.

### Prerequisites
- Phase 0 completed with GREEN or YELLOW result
- Evaluation harness working

### Tasks

#### Task 1.1: Create Router module
- Create `src/parishad/orchestrator/router.py`
- Contains:
  - `RoutingDecision` dataclass (pydantic BaseModel)
  - `PipelineConfig` dataclass (ordered list of role+slot tuples)
  - `AdaptiveRouter` class
    - `route(task_spec: TaskSpec) -> PipelineConfig`
    - pure function, no side effects
    - implements routing rules from Scope-Freeze.md exactly
  - `ROUTING_RULES` constant (the rule table)
- Write unit tests: `tests/test_router.py`
  - test each difficulty/task_type combination
  - verify correct roles are selected
  - verify correct roles are skipped

**Estimated effort:** 1 day

#### Task 1.2: Modify orchestrator engine
- Modify `src/parishad/orchestrator/engine.py`
- Changes:
  - after Darbari runs, call `router.route(task_spec)`
  - execute only the roles returned by the router
  - pass routing decision to trace logger
  - handle missing upstream data gracefully
    - if plan is None (Majumdar skipped), Sainik still runs
    - if verdict is None (Prerak skipped), Raja still runs

**Do NOT:**
  - change the Role base class
  - change role prompt content (except adding null-handling clauses)
  - change the Budget class
  - change the retry logic

**Estimated effort:** 1–2 days

#### Task 1.3: Update role prompts for null upstream data
- Modify `src/parishad/roles/sainik.py`
  - in `format_input()`: if `plan` is None, add line:
    "No execution plan was provided. Work directly from the task specification."
- Modify `src/parishad/roles/raja.py`
  - in `format_input()`: if `verdict` is None, add line:
    "No verification was performed. Review the candidate output directly."
  - if `plan` is None, add line:
    "No execution plan was created."

**Estimated effort:** 0.5 day

#### Task 1.4: Add per-role logging
- Modify `src/parishad/roles/base.py`
  - in `execute()`: log before and after each role
  - capture: role_name, tokens_in, tokens_out, latency_ms, model_slot
- Modify `src/parishad/utils/tracing.py`
  - add `RoleTrace` dataclass
  - add `PipelineTrace` dataclass (list of RoleTraces + routing decision)
  - save full trace to JSON after each query

**Estimated effort:** 1 day

#### Task 1.5: Run pruning experiments
- Create `scripts/run_pruning_experiments.py`
  - Run 4 configurations on GSM8K-100 + HumanEval-50:
    1. Direct single-shot
    2. Direct with CoT
    3. Full pipeline (all roles)
    4. Adaptive-pruned pipeline
  - Save all results to `results/pruning/`

**Estimated effort:** 1 day coding + 1–2 days running

#### Task 1.6: Run ablation studies
- Create `scripts/run_ablations.py`
  - Run 8 configurations on GSM8K-100 + HumanEval-50:
    1. Full pipeline (all 5 roles)
    2. Without Darbari
    3. Without Majumdar
    4. Without Prerak
    5. Without Raja (Sainik output is final)
    6. Only Sainik
    7. Darbari + Sainik only
    8. Sainik + Raja only
  - Save all results to `results/ablations/`

**Estimated effort:** 0.5 day coding + 1–2 days running

#### Task 1.7: Analyze Phase 1 results
- Create `scripts/analyze_phase1.py`
  - Generate summary tables
  - Calculate statistical significance (if multiple seeds)
  - Create accuracy vs tokens plot data (save as CSV)
  - Create ablation table data

**Estimated effort:** 1 day

### Deliverables
- `src/parishad/orchestrator/router.py`
- Modified `engine.py` with router integration
- Modified role files with null-handling
- Enhanced tracing/logging
- `scripts/run_pruning_experiments.py`
- `scripts/run_ablations.py`
- `scripts/analyze_phase1.py`
- `results/pruning/` and `results/ablations/` with data
- `tests/test_router.py`

### Phase 1 Decision Point
Pruned pipeline within 3% of full pipeline accuracy
AND saves 15%+ tokens
→ proceed to Phase 2

Pruned pipeline drops >5% accuracy
→ routing rules need adjustment
→ debug and retry before proceeding

Ablations reveal 1–2 roles are useless
→ interesting finding, record it
→ simplify default pipeline

text


---

## Phase 2 — 3-Model Heterogeneous Orchestration

### Duration: Week 4–5 (10–14 days)

### Purpose
Add true multi-model support with 3 model slots.
Show that right-sizing models to roles improves quality-per-token.

### Prerequisites
- Phase 1 completed with working router
- Evidence that pruning preserves quality

### Tasks

#### Task 2.1: Create MultiModelRunner
- Create `src/parishad/models/multi_runner.py`
- Contains:
  - `SlotState` enum: UNLOADED, LOADING, LOADED, ERROR
  - `ModelSlot` dataclass:
    - slot_name: str (small/mid/big)
    - model_id: str
    - backend: str
    - state: SlotState
    - model_instance: Any (the loaded model)
    - vram_gb: float
  - `MultiModelRunner` class:
    - `__init__(self, slot_configs: dict[str, SlotConfig])`
    - `ensure_loaded(self, slot: str) -> None`
    - `generate(self, prompt: str, slot: str, **kwargs) -> BackendResult`
    - `unload(self, slot: str) -> None`
    - `get_status(self) -> dict[str, SlotState]`
    - `estimate_vram(self) -> float`

**Key implementation rules:**
  - SMALL is loaded once at startup and never unloaded
  - MID is loaded on first use
  - BIG is loaded on demand
  - Before loading, check if VRAM is sufficient
  - If not, unload lowest-priority loaded model first
  - Priority: SMALL (never unload) > MID > BIG
  - All load/unload operations are logged with timing
  - Thread-safe (use a lock for model loading)

**Estimated effort:** 2–3 days

#### Task 2.2: Integrate MultiModelRunner with engine
- Modify `src/parishad/orchestrator/engine.py`
  - Replace single ModelRunner with MultiModelRunner
  - When executing a role, pass the model slot from PipelineConfig
  - Log which model was used for each role

**Estimated effort:** 1 day

#### Task 2.3: Update router for model assignment
- Modify `src/parishad/orchestrator/router.py`
  - `PipelineConfig` now includes model slot per role
  - Routing rules assign models as per Scope-Freeze.md tables
  - Add `ModelAssignment` to `RoutingDecision`

**Estimated effort:** 0.5 day

#### Task 2.4: Test 3-model system end-to-end
- Create `scripts/test_multi_model.py`
  - Run 10 easy + 10 medium + 10 hard queries
  - Verify correct models are loaded for each role
  - Verify model loading/unloading works
  - Verify outputs are reasonable
  - Check VRAM usage at each step

**Estimated effort:** 1 day

#### Task 2.5: Run 3-model experiments
- Create `scripts/run_multi_model_experiments.py`
  - Run 6 configurations on GSM8K-100 + HumanEval-50:
    1. Direct single-shot (MID model)
    2. Direct with CoT (MID model)
    3. Full pipeline, single model (MID for all)
    4. Pruned pipeline, single model (MID for all)
    5. Full pipeline, 3 models (SMALL + MID + BIG)
    6. Pruned pipeline, 3 models (SMALL + MID + BIG)
  - Save to `results/multi_model/`

**Estimated effort:** 1 day coding + 2–3 days running

#### Task 2.6: Analyze Phase 2 results
- Extend `scripts/analyze_results.py`
  - Compare all 6 setups
  - Generate Pareto plot data (accuracy vs tokens)
  - Generate model-usage table
  - Calculate quality-per-token for each setup

**Estimated effort:** 1 day

### Deliverables
- `src/parishad/models/multi_runner.py`
- Modified engine with multi-model support
- Updated router with model assignments
- `scripts/run_multi_model_experiments.py`
- `results/multi_model/` with data
- Test scripts

---

## Phase 3 — Full Benchmarking

### Duration: Week 5–6 (overlaps with Phase 2 analysis)

### Purpose
Run final experiments on full benchmark sets for the paper.
Generate publication-quality results.

### Tasks

#### Task 3.1: Run full GSM8K evaluation
- All 6 configurations on full GSM8K test set (1319 problems)
- 2 random seeds per configuration
- Save detailed per-problem results

**Estimated effort:** 2–3 days of compute

#### Task 3.2: Run full HumanEval evaluation
- All 6 configurations on full HumanEval (164 problems)
- 2 random seeds per configuration
- Code execution with 10-second timeout per problem

**Estimated effort:** 1–2 days of compute

#### Task 3.3: Statistical analysis
- Calculate mean accuracy with 95% confidence intervals
- Paired comparisons between setups
- Effect size calculations

**Estimated effort:** 1 day

### Deliverables
- `results/final/gsm8k_full.json`
- `results/final/humaneval_full.json`
- `results/final/summary_stats.json`

---

## Phase 4 — Analysis + Paper

### Duration: Week 7–8

### Tasks

#### Task 4.1: Generate figures
- Create `scripts/generate_figures.py`
  - Figure 1: Accuracy vs Token Cost (Pareto plot)
  - Figure 2: Ablation table (formatted)
  - Figure 3: Role importance by task type (heatmap or bar chart)
  - Figure 4: Routing distribution (pie or bar chart)
  - Figure 5: Model loading timeline (optional)
  - Save as PDF/PNG

**Estimated effort:** 1–2 days

#### Task 4.2: Write paper
- Use LaTeX (NeurIPS or ACL format)
- Sections as outlined in Scope-Freeze.md
- 6–8 pages

**Estimated effort:** 3–5 days

#### Task 4.3: Submit
- arXiv preprint
- Workshop submission (if deadline aligns)

**Estimated effort:** 1 day

### Deliverables
- `paper/` directory with LaTeX source
- `paper/figures/` with all plots
- arXiv submission