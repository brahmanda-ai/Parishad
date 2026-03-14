# Getting Started — Repository Setup & Experiment Guide

---

## Step 1: Create New Repository

Do NOT modify the existing Parishad repo for experiments.
Create a new repo that imports Parishad as a dependency.

### Repository structure
parishad-research/
├── README.md
├── pyproject.toml
├── .gitignore
├── .python-version # 3.11 or 3.12
│
├── src/
│ └── parishad_research/
│ ├── init.py
│ ├── eval/
│ │ ├── init.py
│ │ ├── gsm8k.py # GSM8K loader + evaluator
│ │ ├── humaneval.py # HumanEval loader + evaluator
│ │ ├── metrics.py # accuracy, tokens, latency
│ │ └── baselines.py # direct, CoT baselines
│ ├── router/
│ │ ├── init.py
│ │ └── adaptive.py # AdaptiveRouter class
│ ├── multi_model/
│ │ ├── init.py
│ │ └── runner.py # MultiModelRunner class
│ └── integration/
│ ├── init.py
│ └── engine_patch.py # patches to ParishadEngine
│
├── scripts/
│ ├── experiment_zero.py
│ ├── run_pruning.py
│ ├── run_ablations.py
│ ├── run_multi_model.py
│ ├── run_full_benchmarks.py
│ ├── analyze_results.py
│ └── generate_figures.py
│
├── configs/
│ ├── models.yaml # model slot configs
│ ├── routing_rules.yaml # routing rule definitions
│ └── experiment_configs.yaml # experiment parameters
│
├── results/ # git-ignored, large files
│ ├── experiment_zero/
│ ├── pruning/
│ ├── ablations/
│ ├── multi_model/
│ └── final/
│
├── paper/
│ ├── main.tex
│ ├── references.bib
│ └── figures/
│
├── tests/
│ ├── test_router.py
│ ├── test_multi_runner.py
│ ├── test_gsm8k_eval.py
│ └── test_humaneval_eval.py
│
├── docs/
│ ├── Scope-Freeze.md
│ ├── Phases.md
│ ├── Start.md
│ └── Architecture.md
│
└── Makefile # shortcuts for common tasks

text


### Why a separate repo?

- keeps the original Parishad clean
- experiments can be messy without polluting the main project
- clear separation between system code and research code
- easier to share/publish the research repo independently

### Alternative: branch in existing repo

If you strongly prefer one repo:
- create a branch: `git checkout -b research/adaptive-pruning`
- add all new code in `src/parishad/eval/` and `scripts/`
- do NOT modify existing files on this branch until Phase 1 proves the concept
- merge to main only after experiments are done

---

## Step 2: Environment Setup

### Create virtual environment

```bash
# Using uv (recommended)
uv venv .venv --python 3.11
source .venv/bin/activate

# Or using standard venv
python3.11 -m venv .venv
source .venv/bin/activate
Install dependencies
Bash

# Install Parishad (existing package)
pip install -e /path/to/parishad  # or pip install parishad

# Install research dependencies
pip install datasets         # for loading GSM8K, HumanEval
pip install matplotlib       # for plots
pip install pandas           # for data analysis
pip install scipy            # for statistical tests
pip install tiktoken         # for token counting
pip install tabulate         # for printing tables

# Install model backend (pick ONE for consistency)
# Option A: Ollama (recommended — simplest)
# Install Ollama from https://ollama.ai
# Then pull models:
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b

# Option B: llama.cpp
pip install llama-cpp-python
# Download GGUF files manually
Verify setup
Bash

# Test that Parishad imports work
python -c "from parishad.orchestrator.engine import ParishadEngine; print('OK')"

# Test that model responds
python -c "
import httpx
r = httpx.post('http://localhost:11434/api/generate', 
    json={'model': 'qwen2.5:7b', 'prompt': 'Say hello', 'stream': False})
print(r.json()['response'][:50])
"

# Test that datasets load
python -c "
from datasets import load_dataset
ds = load_dataset('gsm8k', 'main', split='test')
print(f'GSM8K test: {len(ds)} problems')
"
Step 3: Git Workflow
Branch strategy
text

main                    — stable, only merge when phase is complete
├── phase-0/exp-zero    — experiment zero work
├── phase-1/router      — adaptive router implementation
├── phase-1/ablations   — ablation experiments
├── phase-2/multi-model — 3-model runner
└── phase-3/paper       — paper writing
Commit conventions
text

feat: add GSM8K evaluation loader
fix: correct token counting in baseline runner
exp: run experiment_zero with qwen2.5-7b
data: add experiment_zero results
docs: update Phases.md with Phase 0 results
refactor: extract metric calculation to separate function
What gets committed
text

YES — commit these:
  ✅ all source code
  ✅ all scripts
  ✅ all config files
  ✅ all test files
  ✅ summary result files (JSON, <1MB)
  ✅ paper source (LaTeX)
  ✅ generated figures (PNG/PDF)
  ✅ documentation

NO — do NOT commit these (add to .gitignore):
  ❌ model files (GGUF, safetensors)
  ❌ raw benchmark datasets (loaded via HuggingFace)
  ❌ per-problem result files (can be >100MB)
  ❌ virtual environment
  ❌ __pycache__
  ❌ .DS_Store
.gitignore
text

# Models
*.gguf
*.safetensors
*.bin
models/

# Large results (keep summaries only)
results/*/raw/

# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
Step 4: How to Run Experiment Zero
Prerequisites checklist
text

[ ] Python 3.11+ installed
[ ] Virtual environment created and activated
[ ] Parishad package installed
[ ] Ollama running with qwen2.5:7b pulled (or llama.cpp with GGUF)
[ ] datasets package installed
[ ] results/ directory created
Run the experiment
Bash

# Step 1: Quick sanity check (5 problems only)
python scripts/experiment_zero.py --quick --problems 5

# Step 2: If sanity check passes, run small subset
python scripts/experiment_zero.py --problems 20

# Step 3: If that works, run full experiment
python scripts/experiment_zero.py --problems 100

# Results will be in:
#   results/experiment_zero/summary.json
#   results/experiment_zero/gsm8k_results.json
#   results/experiment_zero/humaneval_results.json
What the script does
text

1. Load 100 GSM8K problems (seed=42 sample from test set)
2. Load 50 HumanEval problems (first 50)
3. For each problem:
   a. Run direct single-shot → log result
   b. Run direct with CoT → log result  
   c. Run through Parishad pipeline → log result
4. Calculate accuracy for each setup
5. Calculate total tokens for each setup
6. Calculate average latency for each setup
7. Print summary table
8. Save all data to results/experiment_zero/
Expected output
text

============================================
  EXPERIMENT ZERO RESULTS
============================================

GSM8K (100 problems):
┌─────────────────────┬──────────┬────────┬─────────┐
│ Setup               │ Accuracy │ Tokens │ Latency │
├─────────────────────┼──────────┼────────┼─────────┤
│ Direct (7B)         │ ??%      │ ????   │ ?.?s    │
│ Direct + CoT (7B)   │ ??%      │ ????   │ ?.?s    │
│ Parishad Full (7B)  │ ??%      │ ????   │ ??.?s   │
└─────────────────────┴──────────┴────────┴─────────┘

HumanEval (50 problems):
┌─────────────────────┬──────────┬────────┬─────────┐
│ Setup               │ Accuracy │ Tokens │ Latency │
├─────────────────────┼──────────┼────────┼─────────┤
│ Direct (7B)         │ ??%      │ ????   │ ?.?s    │
│ Direct + CoT (7B)   │ ??%      │ ????   │ ?.?s    │
│ Parishad Full (7B)  │ ??%      │ ????   │ ??.?s   │
└─────────────────────┴──────────┴────────┴─────────┘

DECISION: [calculated based on accuracy difference]
Interpreting the results
After experiment_zero finishes:

Bash

# Read the decision
cat results/experiment_zero/summary.json | python -m json.tool
Look at the decision field:

"green" → proceed to Phase 1
"yellow" → proceed with caution
"red" → investigate further
"stop" → pipeline does not help, pivot
Step 5: How to Run Phase 1 (Pruning Experiments)
After implementing the router (see Phases.md Task 1.1–1.4)
Bash

# Test router in isolation
python -m pytest tests/test_router.py -v

# Run pruning comparison
python scripts/run_pruning.py --problems 100

# Run ablation study
python scripts/run_ablations.py --problems 100

# Analyze results
python scripts/analyze_results.py --phase pruning
python scripts/analyze_results.py --phase ablations
Expected output structure
text

results/pruning/
  ├── config_direct.json
  ├── config_cot.json
  ├── config_full_pipeline.json
  ├── config_pruned_pipeline.json
  └── summary.json

results/ablations/
  ├── config_full.json
  ├── config_no_darbari.json
  ├── config_no_majumdar.json
  ├── config_no_prerak.json
  ├── config_no_raja.json
  ├── config_only_sainik.json
  ├── config_darbari_sainik.json
  ├── config_sainik_raja.json
  └── summary.json
Step 6: How to Run Phase 2 (3-Model Experiments)
After implementing MultiModelRunner (see Phases.md Task 2.1–2.3)
Bash

# Verify 3 models are available
python scripts/test_multi_model.py

# Run multi-model comparison
python scripts/run_multi_model.py --problems 100

# Analyze
python scripts/analyze_results.py --phase multi_model
Step 7: How to Run Full Benchmarks (Phase 3)
Bash

# Full GSM8K (1319 problems, 2 seeds) — takes several hours
python scripts/run_full_benchmarks.py --benchmark gsm8k --seeds 2

# Full HumanEval (164 problems, 2 seeds) — takes 1-2 hours
python scripts/run_full_benchmarks.py --benchmark humaneval --seeds 2

# Generate all figures
python scripts/generate_figures.py

# Figures saved to paper/figures/
Makefile (shortcuts)
Makefile

.PHONY: setup test exp0 phase1 phase2 full figures clean

setup:
	uv venv .venv --python 3.11
	. .venv/bin/activate && pip install -e . && pip install -r requirements-research.txt

test:
	python -m pytest tests/ -v

exp0:
	python scripts/experiment_zero.py --problems 100

exp0-quick:
	python scripts/experiment_zero.py --quick --problems 5

phase1-router-test:
	python -m pytest tests/test_router.py -v

phase1-pruning:
	python scripts/run_pruning.py --problems 100

phase1-ablations:
	python scripts/run_ablations.py --problems 100

phase1-analyze:
	python scripts/analyze_results.py --phase pruning
	python scripts/analyze_results.py --phase ablations

phase2-test:
	python scripts/test_multi_model.py

phase2-run:
	python scripts/run_multi_model.py --problems 100

phase2-analyze:
	python scripts/analyze_results.py --phase multi_model

full-gsm8k:
	python scripts/run_full_benchmarks.py --benchmark gsm8k --seeds 2

full-humaneval:
	python scripts/run_full_benchmarks.py --benchmark humaneval --seeds 2

figures:
	python scripts/generate_figures.py

clean:
	rm -rf results/
	rm -rf __pycache__
	find . -name "*.pyc" -delete
Troubleshooting
Ollama not responding
Bash

# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running
ollama serve

# If model not pulled
ollama pull qwen2.5:7b
Out of memory
Bash

# Check GPU memory
nvidia-smi  # NVIDIA
# or
system_profiler SPDisplaysDataType  # Mac

# If OOM, reduce context length in configs/models.yaml
# Or use smaller quantization
HumanEval code execution fails
Bash

# HumanEval needs to execute generated code
# Make sure subprocess timeout is set (10 seconds default)
# Check that generated code is valid Python
# Look at results/*/humaneval_errors.json for details
Results look wrong
Bash

# Check answer extraction
python -c "
from parishad_research.eval.gsm8k import extract_answer
print(extract_answer('The answer is 42.'))  # should print 42
print(extract_answer('Therefore, x = 3.14'))  # should print 3.14
"