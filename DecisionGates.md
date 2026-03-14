# Decision Gates — When to Proceed, Pivot, or Stop

---

## Purpose

This document defines explicit go/no-go criteria for each phase.
It prevents wasting time on phases that depend on failed assumptions.

**Every gate must be explicitly evaluated before proceeding.**

When working with AI assistants (Claude, Copilot):
- before starting any Phase N+1 work, check if Gate N has been passed
- if the gate has NOT been passed, do NOT proceed
- if the gate result is ambiguous, discuss before proceeding

---

## Gate 0: Can We Run Experiments At All?

### When
Before writing any experiment code.

### Checklist
[ ] Python 3.11+ installed and working
[ ] Parishad package can be imported:
python -c "from parishad.orchestrator.engine import ParishadEngine"
[ ] At least one model backend works:
- Ollama running with qwen2.5:7b pulled, OR
- llama.cpp with qwen2.5-7b GGUF file available
[ ] Model responds to a test prompt:
(verify with a simple generation call)
[ ] ParishadEngine can be instantiated programmatically:
(not just through TUI, but from a Python script)
[ ] GSM8K dataset can be loaded:
python -c "from datasets import load_dataset; load_dataset('gsm8k','main')"
[ ] We have a working directory for results:
mkdir -p results/experiment_zero

text


### Decision
All checks pass → proceed to Experiment Zero
Any check fails → fix the issue first, do NOT work around it
Engine cannot be → this is a blocker, must be resolved before
called from anything else
Python script

text


---

## Gate 1: Does the Pipeline Help?

### When
After Experiment Zero completes.

### What we look at

Compare accuracy on GSM8K (100 problems):
Pipeline accuracy minus Direct accuracy = DELTA

text


### Decision Rules
DELTA >= +10%
STATUS: GREEN
ACTION: proceed to Phase 1 with full confidence
NOTES: strong signal, pipeline clearly helps

DELTA = +5% to +9%
STATUS: YELLOW
ACTION: proceed to Phase 1 cautiously
NOTES: signal exists but modest
paper will need careful framing
results may not impress reviewers

DELTA = +2% to +4%
STATUS: ORANGE
ACTION: investigate before proceeding
TASKS:
1. check if Parishad prompts are suboptimal
2. try with a different model (Llama instead of Qwen)
3. check if some roles are actively hurting (quick ablation)
4. if improvement found → re-run experiment zero
5. if no improvement after 3 days → proceed to STOP evaluation

DELTA = -1% to +1%
STATUS: RED
ACTION: pipeline adds no measurable value
TASKS:
1. analyze per-problem results
- does pipeline help on hard problems but hurt on easy ones?
- if yes, that itself is interesting (go to Phase 1 for pruning)
- if no, pipeline is truly useless
2. if pipeline helps on SOME problems → limited proceed
3. if pipeline helps on NONE → STOP

DELTA <= -2%
STATUS: HARD STOP
ACTION: pipeline actively hurts performance
TASKS:
1. document findings (this is a valid negative result)
2. do NOT proceed to Phase 1
3. pivot to product direction or different research angle
4. optionally write up as negative result

text


### Also check HumanEval

Same delta thresholds apply.
If GSM8K is GREEN but HumanEval is RED (or vice versa):
- pipeline helps for one task type but not another
- this is actually an interesting finding
- proceed to Phase 1 with focus on the task type that works

### How to evaluate

```bash
# After experiment_zero.py finishes:
cat results/experiment_zero/summary.json

# Look for:
{
  "gsm8k": {
    "direct_accuracy": 0.55,
    "pipeline_accuracy": 0.67,
    "delta": 0.12,           ← this is the key number
    "gate_status": "green"
  }
}
Gate 2: Does Pruning Preserve Quality?
When
After Phase 1 pruning experiments complete.

What we look at
Compare pruned pipeline vs full pipeline:

text

Full pipeline accuracy minus Pruned pipeline accuracy = QUALITY_LOSS
Full pipeline tokens minus Pruned pipeline tokens = TOKEN_SAVINGS
Decision Rules
text

QUALITY_LOSS <= 3% AND TOKEN_SAVINGS >= 15%
  STATUS: GREEN
  ACTION: proceed to Phase 2
  NOTES: pruning works as hoped
         clear efficiency gain with minimal quality loss

QUALITY_LOSS <= 5% AND TOKEN_SAVINGS >= 20%
  STATUS: YELLOW
  ACTION: proceed to Phase 2 but note the tradeoff
  NOTES: pruning works but quality loss is noticeable
         paper should discuss this tradeoff explicitly

QUALITY_LOSS > 5% AND TOKEN_SAVINGS >= 20%
  STATUS: ORANGE
  ACTION: investigate routing rules
  TASKS:
    1. which route causes the most quality loss?
    2. is Darbari misclassifying difficulty?
    3. adjust routing rules (be more conservative)
    4. re-run pruning experiment
    5. if still >5% loss → use conservative pruning only

QUALITY_LOSS > 5% AND TOKEN_SAVINGS < 15%
  STATUS: RED
  ACTION: pruning does not work well
  TASKS:
    1. analyze ablation results
    2. if ablation reveals interesting role-importance patterns
       → write paper around ablation findings instead
    3. do NOT proceed to Phase 2
    4. paper becomes "which roles matter" not "adaptive pruning"

TOKEN_SAVINGS < 10% regardless of quality
  STATUS: RED
  ACTION: pruning saves too little to matter
  NOTES: the roles are all doing meaningful work
         or the routing is too conservative
Also check ablation results
The ablation table should reveal:

which roles contribute most to accuracy
which roles can be safely skipped
If removing ANY single role causes <1% accuracy drop:

that role is decorative
simplify the default pipeline
this is a finding worth reporting
Gate 3: Does Multi-Model Improve Quality-Per-Token?
When
After Phase 2 multi-model experiments complete.

What we look at
Compare these setups:

text

Setup 4 (pruned, 1 model):  accuracy_4, tokens_4
Setup 6 (pruned, 3 models): accuracy_6, tokens_6

Quality-per-token improvement:
  QPT_4 = accuracy_4 / tokens_4
  QPT_6 = accuracy_6 / tokens_6
  QPT_IMPROVEMENT = (QPT_6 - QPT_4) / QPT_4
Decision Rules
text

QPT_IMPROVEMENT >= 10% (3-model is clearly better per token)
  STATUS: GREEN
  ACTION: include multi-model results prominently in paper
  NOTES: strong contribution, genuine heterogeneous orchestration works

QPT_IMPROVEMENT = 5-9%
  STATUS: YELLOW
  ACTION: include multi-model results in paper
  NOTES: improvement exists but modest
         paper should be honest about the magnitude

QPT_IMPROVEMENT = 0-4%
  STATUS: ORANGE
  ACTION: include as secondary result, not main claim
  NOTES: multi-model adds complexity for minimal gain
         paper should acknowledge this honestly
         main paper contribution stays as pruning

QPT_IMPROVEMENT < 0% (3-model is WORSE per token)
  STATUS: RED
  ACTION: do NOT include multi-model as a contribution
  NOTES: model loading overhead negates the benefit
         or small model is too weak for its assigned roles
         paper focuses on pruning only
         mention multi-model as "explored but not beneficial"
Also check raw accuracy
If 3-model achieves HIGHER raw accuracy than 1-model
(regardless of tokens):

the BIG model in Raja genuinely helps for hard tasks
this is worth reporting even if QPT improvement is small
Gate 4: Are Full Benchmark Results Consistent?
When
After Phase 3 full benchmark runs complete.

What we look at
Do the full-benchmark results match the 100-problem subset results?

text

Subset accuracy (100 problems): X%
Full accuracy (1319 problems): Y%

If |X - Y| > 5%:
  → subset was not representative
  → results may be weaker or stronger than expected
  → re-examine and adjust claims

If |X - Y| <= 5%:
  → subset was representative
  → proceed with paper writing
Statistical validity
text

Confidence interval width for each setup:
  If 95% CI width > 10%:
    → results are too noisy
    → need more seeds or more problems
  
  If 95% CI width <= 5%:
    → results are robust
    → proceed with paper
Gate 5: Is the Paper Honest?
When
Before submitting the paper.

Checklist
text

[ ] Every claim is supported by experimental evidence
[ ] No made-up numbers (e.g. "3x cost reduction" without data)
[ ] Limitations section is honest and specific
[ ] Related work includes Self-Refine, ChatDev, AdaptOrch
[ ] Comparison to baselines is fair (same model, same prompts)
[ ] Statistical tests are appropriate (CI or p-values)
[ ] All experiments are reproducible (seeds, configs documented)
[ ] The paper does NOT claim NeurIPS-level novelty unless results warrant it
[ ] The paper does NOT claim "GPT-4 level quality"
[ ] Hardware used is clearly documented
[ ] Model versions are clearly documented
Summary of Gates
text

Gate 0: Can we run experiments?
        → technical readiness check
        → MUST PASS before any code

Gate 1: Does the pipeline help?
        → experiment-zero result
        → MUST PASS before Phase 1

Gate 2: Does pruning work?
        → Phase 1 result
        → MUST PASS before Phase 2

Gate 3: Does multi-model help?
        → Phase 2 result
        → determines paper scope (not a hard blocker)

Gate 4: Are full results consistent?
        → Phase 3 result
        → determines paper confidence

Gate 5: Is the paper honest?
        → self-check before submission
        → integrity gate
Instructions for AI Assistants
When asked to work on a specific phase:

First check: has the previous gate been passed?
If gate status is unknown, ASK the user for the result
If gate was not passed, explain why you cannot proceed
If gate was passed, proceed strictly within that phase's scope
Example interaction:

text

User: "Let's build the adaptive router"

AI should ask: "Before building the router, I need to know:
has Experiment Zero been completed? What was the accuracy
delta between pipeline and direct? I need to verify Gate 1
is passed before creating router.py."
Do NOT skip gates.
Do NOT assume gates are passed.
Always verify.

text


---