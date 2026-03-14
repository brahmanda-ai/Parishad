# Branch Strategy — Git Workflow for Parishad Research

---

## Repository Decision

### Recommended: Use existing Parishad repo with branches

Why:
- eval scripts need to import from parishad internals
- avoids managing two repos and their dependency
- easier to eventually merge changes if experiments succeed
- simpler for one person

### Alternative: Separate repo

Only if you want to keep the main repo completely untouched.
But this creates import complexity that is not worth it.

---

## Branch Structure
main
│
│ ← stable, current Parishad, do not touch
│
├── research/experiment-zero
│ │
│ │ ← ONLY adds eval/ and scripts/
│ │ ← ZERO changes to existing code
│ │ ← merged to main ONLY after experiment completes
│ │
│ └── deliverables:
│ src/parishad/eval/
│ scripts/experiment_zero.py
│ results/experiment_zero/
│
├── research/phase-1-pruning (create AFTER exp-zero passes)
│ │
│ │ ← based on main (not on experiment-zero branch)
│ │ ← adds router.py
│ │ ← modifies engine.py, sainik.py, raja.py minimally
│ │ ← adds pruning + ablation scripts
│ │
│ └── deliverables:
│ src/parishad/orchestrator/router.py
│ modified engine.py
│ scripts/run_pruning.py
│ scripts/run_ablations.py
│ results/pruning/
│ results/ablations/
│
├── research/phase-2-multi-model (create AFTER phase-1 passes)
│ │
│ │ ← based on research/phase-1-pruning
│ │ ← adds multi_runner.py
│ │ ← modifies engine.py further
│ │
│ └── deliverables:
│ src/parishad/models/multi_runner.py
│ scripts/run_multi_model.py
│ results/multi_model/
│
└── research/paper (create AFTER phase-2)
│
│ ← based on research/phase-2-multi-model
│ ← adds paper/ directory
│ ← adds figure generation scripts
│
└── deliverables:
paper/main.tex
paper/figures/
scripts/generate_figures.py

text


---

## Branch Rules

### main branch
ALLOWED:
✅ bug fixes to existing code
✅ documentation updates
✅ dependency updates

NOT ALLOWED:
❌ any research code
❌ eval/ directory
❌ experiment scripts
❌ router.py
❌ multi_runner.py
❌ changes to engine.py for research purposes

text


### research/experiment-zero branch
ALLOWED:
✅ create src/parishad/eval/ directory and all files inside
✅ create scripts/experiment_zero.py
✅ create results/ directory
✅ create configs/ for experiment settings
✅ add research dependencies to pyproject.toml (optional group)

NOT ALLOWED:
❌ modify ANY existing file in src/parishad/
❌ modify orchestrator/engine.py
❌ modify any role file
❌ modify models/runner.py
❌ modify cli/ or tools/ or checker/
❌ add router.py or multi_runner.py

text


### research/phase-1-pruning branch
ALLOWED:
✅ create src/parishad/orchestrator/router.py
✅ modify engine.py (add router integration, ~50 lines)
✅ modify sainik.py (add null plan handling, ~5 lines)
✅ modify raja.py (add null plan/verdict handling, ~10 lines)
✅ modify base.py (add per-role timing, ~15 lines)
✅ modify tracing.py (add RoleTrace, ~30 lines)
✅ create pruning and ablation scripts
✅ create tests/test_router.py

NOT ALLOWED:
❌ create multi_runner.py (that is Phase 2)
❌ modify cli/ or tools/ or checker/
❌ modify darbari.py, majumdar.py, prerak.py
❌ change existing role prompts
❌ add new roles
❌ change pipeline YAML configs

text


### research/phase-2-multi-model branch
ALLOWED:
✅ create src/parishad/models/multi_runner.py
✅ modify engine.py (add multi-model support, ~30 lines)
✅ modify router.py (return model slots, ~20 lines)
✅ create multi-model scripts
✅ create tests/test_multi_runner.py
✅ create configs/models.yaml

NOT ALLOWED:
❌ modify cli/ or tools/ or checker/
❌ modify any role file (roles are model-agnostic)
❌ modify backends (use existing backend interface)
❌ add new backends

text


---

## How to Create and Work on Branches

### Starting Experiment Zero

```bash
# Make sure main is clean
git checkout main
git pull
git status  # should be clean

# Create experiment-zero branch
git checkout -b research/experiment-zero

# Do all experiment-zero work here
# ...

# Commit often with clear messages
git add src/parishad/eval/
git commit -m "feat: add GSM8K evaluation loader"

git add scripts/experiment_zero.py
git commit -m "feat: add experiment zero script"

# After experiment completes
git add results/experiment_zero/summary.json
git commit -m "data: experiment zero results - pipeline shows +12% on GSM8K"
Moving to Phase 1 (after experiment-zero passes)
Bash

# First merge eval utilities to main (they are safe)
git checkout main
git merge research/experiment-zero
git push

# Create phase-1 branch from main
git checkout -b research/phase-1-pruning

# Do all phase-1 work here
# ...
Moving to Phase 2 (after phase-1 passes)
Bash

# Merge phase-1 into main
git checkout main
git merge research/phase-1-pruning
git push

# Create phase-2 branch from main (which now has phase-1 changes)
git checkout -b research/phase-2-multi-model

# Do all phase-2 work here
# ...
If Experiment Zero FAILS
Bash

# Do NOT merge to main
# Keep the branch for reference
git checkout main

# Optionally tag it
git tag experiment-zero-failed research/experiment-zero

# Move on (product direction or different research angle)
Merge Strategy
When to merge to main
text

research/experiment-zero → main
  WHEN: experiment complete, results documented
  WHAT MERGES: eval/ directory, experiment scripts
  WHAT STAYS ON BRANCH: raw result data (too large)

research/phase-1-pruning → main
  WHEN: pruning experiments show positive results
  WHAT MERGES: router.py, engine changes, role changes, scripts
  CONDITION: all tests pass

research/phase-2-multi-model → main
  WHEN: multi-model experiments show improvement
  WHAT MERGES: multi_runner.py, engine changes, scripts
  CONDITION: all tests pass, existing TUI still works
Before any merge to main
Bash

# Verify existing functionality still works
python -m parishad --help           # CLI works
python -m pytest tests/ -v          # existing tests pass

# Verify no unintended changes
git diff main -- src/parishad/cli/     # should be empty
git diff main -- src/parishad/tools/   # should be empty
git diff main -- src/parishad/checker/ # should be empty
Commit Message Convention
text

feat:     new feature (router, eval loader, multi-runner)
fix:      bug fix
exp:      experiment run or result
data:     adding result data
docs:     documentation change
test:     adding or fixing tests
refactor: code restructure (no behavior change)
config:   configuration change

Examples:
  feat: add adaptive router with rule-based routing
  feat: add GSM8K evaluation loader with answer extraction
  exp: run experiment-zero with qwen2.5-7b (100 GSM8K + 50 HumanEval)
  data: experiment-zero results — pipeline +11% GSM8K, +8% HumanEval
  fix: correct token counting in baseline runner
  test: add unit tests for adaptive router
  docs: update Architecture.md with Phase 1 results
.gitignore additions
gitignore

# Add to existing .gitignore

# Large result files
results/*/raw/
results/*/per_problem/
results/*/traces/

# Keep only summaries
!results/*/summary.json
!results/*/config.json

# Models (never commit)
*.gguf
*.safetensors
*.bin

# Experiment scratch
scratch/
tmp/
text


---