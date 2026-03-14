# Experiment Playbook — Step-by-Step Guide

---

## Before You Start Any Experiment

### Hardware check

```bash
# Check GPU
nvidia-smi  # or system_profiler SPDisplaysDataType on Mac

# Record these numbers:
# GPU model: ________________
# VRAM total: _______ GB
# VRAM free: _______ GB
# CPU cores: _______
# RAM total: _______ GB
Model check
Bash

# If using Ollama:
ollama list
# Verify these models appear:
#   qwen2.5:1.5b
#   qwen2.5:7b
#   qwen2.5:14b

# If using llama.cpp:
ls ~/models/  # or wherever your GGUF files are
# Verify:
#   qwen2.5-1.5b-instruct-q4_k_m.gguf
#   qwen2.5-7b-instruct-q4_k_m.gguf
#   qwen2.5-14b-instruct-q4_k_m.gguf
Sanity check
Bash

# Run 3 problems to make sure everything works
python scripts/experiment_zero.py --quick --problems 3
# This should complete in < 5 minutes
# Check results/experiment_zero/quick/ for output
Experiment Zero — The Kill Switch
What
Compare direct prompting vs Parishad pipeline.
Same model (7B) for both.

Why
To determine if the pipeline adds value at all.

How
Bash

python scripts/experiment_zero.py \
    --problems 100 \
    --model qwen2.5:7b \
    --seed 42 \
    --output-dir results/experiment_zero/
Expected runtime
GSM8K 100 problems × 3 setups × ~30s each ≈ 2.5 hours
HumanEval 50 problems × 3 setups × ~45s each ≈ 1.1 hours
Total: ~3.5 hours on a decent GPU
What to look for
Open results/experiment_zero/summary.json:

JSON

{
  "gsm8k": {
    "direct":   {"accuracy": 0.55, "avg_tokens": 1200, "avg_latency_s": 3.2},
    "cot":      {"accuracy": 0.60, "avg_tokens": 1800, "avg_latency_s": 4.5},
    "pipeline": {"accuracy": 0.67, "avg_tokens": 4500, "avg_latency_s": 11.2}
  },
  "humaneval": {
    "direct":   {"accuracy": 0.48, "avg_tokens": 900,  "avg_latency_s": 2.8},
    "cot":      {"accuracy": 0.50, "avg_tokens": 1400, "avg_latency_s": 3.9},
    "pipeline": {"accuracy": 0.60, "avg_tokens": 3800, "avg_latency_s": 10.1}
  },
  "decision": "green"
}
Decision matrix
Pipeline vs Direct delta	Decision	Action
+10% or more	GREEN	Full speed ahead
+5% to +9%	YELLOW	Proceed, expect modest gains
+1% to +4%	ORANGE	Investigate, maybe adjust prompts
0% or negative	RED	Stop research direction
Pruning Experiments
What
Compare full pipeline vs adaptively pruned pipeline.

Prerequisites
Experiment zero showed GREEN or YELLOW
AdaptiveRouter implemented and tested
How
Bash

python scripts/run_pruning.py \
    --problems 100 \
    --model qwen2.5:7b \
    --seed 42 \
    --output-dir results/pruning/
What to look for
The pruned pipeline should:

save 15-30% tokens compared to full pipeline
lose less than 3% accuracy compared to full pipeline
still beat direct single-shot on accuracy
text

GOOD result:
  Full pipeline:   67% accuracy, 4500 tokens
  Pruned pipeline: 65% accuracy, 3100 tokens  ← saves 31% tokens
  Direct:          55% accuracy, 1200 tokens

BAD result:
  Full pipeline:   67% accuracy, 4500 tokens
  Pruned pipeline: 58% accuracy, 3100 tokens  ← lost too much quality
  Direct:          55% accuracy, 1200 tokens
Ablation Experiments
What
Remove one role at a time to measure each role's contribution.

How
Bash

python scripts/run_ablations.py \
    --problems 100 \
    --model qwen2.5:7b \
    --seed 42 \
    --output-dir results/ablations/
What to look for
Create this table from results:

text

| Configuration          | GSM8K | HumanEval | Tokens | Notes          |
|------------------------|-------|-----------|--------|----------------|
| Full (5 roles)         | 67%   | 60%       | 4500   | baseline       |
| No Darbari             | ??%   | ??%       | ????   |                |
| No Majumdar            | ??%   | ??%       | ????   |                |
| No Prerak              | ??%   | ??%       | ????   |                |
| No Raja                | ??%   | ??%       | ????   |                |
| Only Sainik            | ??%   | ??%       | ????   | ≈ direct       |
| Darbari + Sainik       | ??%   | ??%       | ????   |                |
| Sainik + Raja          | ??%   | ??%       | ????   |                |
What makes a good ablation finding
A good finding would be:

removing Majumdar barely hurts GSM8K but hurts HumanEval significantly
→ "planning matters more for code than math"
removing Prerak hurts both significantly
→ "verification is universally important"
Darbari + Sainik + Raja is 95% as good as full pipeline
→ "the minimal useful pipeline has 3 roles"
3-Model Experiments
What
Compare single-model pipeline vs 3-model pipeline.

Prerequisites
Pruning experiments completed
MultiModelRunner implemented and tested
How
Bash

python scripts/run_multi_model.py \
    --problems 100 \
    --small qwen2.5:1.5b \
    --mid qwen2.5:7b \
    --big qwen2.5:14b \
    --seed 42 \
    --output-dir results/multi_model/
What to look for
The 3-model pruned pipeline should:

match or exceed full single-model pipeline accuracy
use fewer total tokens (small model is cheaper)
have better quality-per-token than any other setup
text

IDEAL result:
  Direct 7B:              55% accuracy, 1200 tokens, QPT=0.046
  Full pipeline 1-model:  67% accuracy, 4500 tokens, QPT=0.015
  Pruned 1-model:         65% accuracy, 3100 tokens, QPT=0.021
  Full pipeline 3-model:  70% accuracy, 4300 tokens, QPT=0.016
  Pruned 3-model:         69% accuracy, 2900 tokens, QPT=0.024 ← best QPT
Full Benchmark Run (for paper)
What
Run winning configurations on full benchmark sets with multiple seeds.

How
Bash

# This takes many hours — run overnight or over weekend

# Full GSM8K
python scripts/run_full_benchmarks.py \
    --benchmark gsm8k \
    --seeds 42,123,456 \
    --output-dir results/final/

# Full HumanEval
python scripts/run_full_benchmarks.py \
    --benchmark humaneval \
    --seeds 42,123,456 \
    --output-dir results/final/
Configurations to run in full
Only run the most important configurations on full data:

Direct single-shot (7B)
Direct with CoT (7B)
Full pipeline, single model (7B)
Pruned pipeline, 3 models (the main contribution)
Do NOT run all 8 ablation configs on full data.
Ablations on 100-problem subsets are sufficient.

Generating Figures
Bash

python scripts/generate_figures.py \
    --results-dir results/ \
    --output-dir paper/figures/
Figures generated
pareto_plot.pdf — accuracy vs tokens, all setups
ablation_table.pdf — formatted ablation results
routing_distribution.pdf — how queries were routed
role_importance.pdf — per-role accuracy contribution
token_breakdown.pdf — tokens used per role per setup
Logging and Debugging
Every experiment saves
text

results/{experiment_name}/
├── summary.json          # aggregate metrics
├── config.json           # exact configuration used
├── per_problem/          # individual problem results
│   ├── gsm8k_001.json
│   ├── gsm8k_002.json
│   └── ...
├── traces/               # full pipeline traces
│   ├── trace_001.json
│   └── ...
└── errors.json           # any failures with stack traces
If something goes wrong
Bash

# Check errors
cat results/{experiment}/errors.json | python -m json.tool

# Check a specific problem
cat results/{experiment}/per_problem/gsm8k_042.json | python -m json.tool

# Check a specific trace
cat results/{experiment}/traces/trace_042.json | python -m json.tool

# Common issues:
# 1. Model timeout → increase timeout in configs/models.yaml
# 2. Answer extraction fails → check eval/gsm8k.py extract function
# 3. HumanEval execution fails → check sandbox timeout
# 4. OOM → reduce context length or use smaller quantization
Results Checklist Before Paper
Before writing the paper, verify you have:

 Experiment zero results (Phase 0)
 Pruning comparison: direct vs CoT vs full vs pruned (Phase 1)
 Ablation table: all 8 configs on 100-problem subsets (Phase 1)
 3-model comparison: all 6 configs on 100-problem subsets (Phase 2)
 Full benchmark: 4 key configs on full GSM8K + HumanEval (Phase 3)
 Statistical tests: confidence intervals for all full-benchmark results
 Figures: all 5 plots generated
 All configs and seeds documented
 Hardware specs recorded
 Model versions recorded