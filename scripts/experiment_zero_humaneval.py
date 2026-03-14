"""Experiment Zero runner for HumanEval using Parishad pipeline execution."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import random
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure local src/ is importable when running directly from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pydantic import BaseModel, Field

from parishad.eval.baselines import run_direct, run_parishad_pipeline
from parishad.eval.humaneval import execute_code, extract_code_from_response, load_humaneval
from parishad.eval.metrics import ExperimentResult, ProblemResult, SetupResult, save_results
from parishad.orchestrator.engine import Parishad

logger = logging.getLogger(__name__)


class ProgressState(BaseModel):
    """Checkpoint payload to support resume after interruption."""

    benchmark: str = "humaneval"
    part: int
    seed: int
    total_target: int
    chunk_size: int
    selected_indices: list[int]
    completed_problem_ids: list[str] = Field(default_factory=list)
    problems: list[ProblemResult] = Field(default_factory=list)


@dataclass
class RunContext:
    """Runtime paths for experiment artifacts."""

    output_dir: Path
    progress_path: Path
    official_csv_path: Path
    metrics_csv_path: Path


def run_experiment(args: argparse.Namespace) -> int:
    """Execute one HumanEval chunk for direct one-shot and Parishad pipeline."""
    _configure_logging(args.log_level)
    context = _build_paths(args)
    context.output_dir.mkdir(parents=True, exist_ok=True)

    stop_requested = {"value": False}

    def _handle_sigint(signum: int, frame: Any) -> None:  # noqa: ARG001
        if stop_requested["value"]:
            raise KeyboardInterrupt("Second Ctrl+C detected. Exiting now.")
        stop_requested["value"] = True
        logger.warning("Ctrl+C detected. Finishing current problem, checkpointing, then stopping.")

    signal.signal(signal.SIGINT, _handle_sigint)

    parishad = Parishad(
        config=args.config,
        model_config_path=None,
        profile=args.profile,
        pipeline_config_path=None,
        trace_dir=args.trace_dir,
        mock=False,
        stub=False,
        mode=args.mode,
        user_forced_config=None,
        no_retry=args.no_retry,
    )

    dataset = load_humaneval(dataset_path=args.dataset_path)
    selected_indices = _partition_indices(
        total_items=len(dataset),
        seed=args.seed,
        total_target=args.total_problems,
        part=args.part,
        chunk_size=args.chunk_size,
    )
    logger.info(
        "HumanEval selection prepared: total=%d target=%d part=%d size=%d",
        len(dataset),
        args.total_problems,
        args.part,
        len(selected_indices),
    )

    state = _load_or_init_state(args, context, selected_indices)
    completed = set(state.completed_problem_ids)

    try:
        for global_idx in selected_indices:
            if stop_requested["value"]:
                break

            row = dataset[global_idx]
            task_id = row["task_id"]
            problem_id = f"humaneval-{task_id}"
            if problem_id in completed:
                continue

            prompt = row["prompt"]
            entry_point = row["entry_point"]
            ground_truth = row["canonical_solution"]

            _run_setup_for_problem(
                state=state,
                completed=completed,
                problem_id=problem_id,
                prompt=prompt,
                entry_point=entry_point,
                test_source=row["test"],
                ground_truth=ground_truth,
                timeout=args.timeout,
                setup="direct",
                execute_fn=lambda: run_direct(parishad.engine.model_runner, prompt),
            )
            _run_setup_for_problem(
                state=state,
                completed=completed,
                problem_id=problem_id,
                prompt=prompt,
                entry_point=entry_point,
                test_source=row["test"],
                ground_truth=ground_truth,
                timeout=args.timeout,
                setup="pipeline",
                execute_fn=lambda: run_parishad_pipeline(parishad, prompt),
            )
            _persist_state_and_reports(state, context)

            logger.info(
                "Completed %s (%d/%d)",
                problem_id,
                len([k for k in completed if k.endswith(":pipeline")]),
                len(selected_indices),
            )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user; checkpoint saved.")
    finally:
        _persist_state_and_reports(state, context)

    direct_results = [item for item in state.problems if item.setup == "direct"]
    pipeline_results = [item for item in state.problems if item.setup == "pipeline"]
    experiment = ExperimentResult(
        experiment_name=f"experiment_zero_humaneval_part_{args.part}",
        model_name=_resolve_model_name(parishad),
        seed=args.seed,
        hardware=platform.platform(),
        results={
            "humaneval": {
                "direct": _build_setup_result("direct", "humaneval", direct_results),
                "pipeline": _build_setup_result("pipeline", "humaneval", pipeline_results),
            }
        },
        decision="red",
        decision_reason="Single-benchmark pipeline-only run. Gate decision requires both benchmarks and deltas.",
    )
    save_results(experiment, str(context.output_dir))

    logger.info("HumanEval part %d finished. Artifacts: %s", args.part, context.output_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for HumanEval run script."""
    parser = argparse.ArgumentParser(description="Run Experiment Zero HumanEval chunk with Parishad pipeline")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--total-problems", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/experiment_zero")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--config", type=str, default="core")
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--no-retry", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def _partition_indices(
    total_items: int,
    seed: int,
    total_target: int,
    part: int,
    chunk_size: int,
) -> list[int]:
    """Create deterministic, non-overlapping partition indices."""
    rng = random.Random(seed)
    indices = list(range(total_items))
    rng.shuffle(indices)

    target_pool = indices[: min(total_items, total_target)]
    start = (part - 1) * chunk_size
    end = start + chunk_size
    return target_pool[start:end]


def _build_paths(args: argparse.Namespace) -> RunContext:
    """Build deterministic output file paths for this part."""
    out_dir = Path(args.output_dir)
    progress_path = out_dir / f"humaneval_part{args.part}_progress.json"
    official_csv = out_dir / f"humaneval_official_part{args.part}.csv"
    metrics_csv = out_dir / f"humaneval_metrics_part{args.part}.csv"
    return RunContext(
        output_dir=out_dir,
        progress_path=progress_path,
        official_csv_path=official_csv,
        metrics_csv_path=metrics_csv,
    )


def _load_or_init_state(
    args: argparse.Namespace,
    context: RunContext,
    selected_indices: list[int],
) -> ProgressState:
    """Load progress from disk or initialize a new checkpoint state."""
    if args.resume and context.progress_path.exists():
        payload = json.loads(context.progress_path.read_text(encoding="utf-8"))
        state = ProgressState.model_validate(payload)
        if state.selected_indices == selected_indices:
            logger.info("Resuming from %s", context.progress_path)
            return state
        logger.warning("Existing progress file selection differs; starting fresh for this run.")

    return ProgressState(
        part=args.part,
        seed=args.seed,
        total_target=args.total_problems,
        chunk_size=args.chunk_size,
        selected_indices=selected_indices,
    )


def _persist_state_and_reports(state: ProgressState, context: RunContext) -> None:
    """Persist checkpoint JSON and regenerate both CSV reports."""
    context.progress_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    with context.official_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["task_id", "setup", "completion", "passed"],
        )
        writer.writeheader()
        for item in state.problems:
            writer.writerow(
                {
                    "task_id": item.problem_id.replace("humaneval-", "", 1),
                    "setup": item.setup,
                    "completion": item.predicted_answer,
                    "passed": item.correct,
                }
            )

    with context.metrics_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "problem_id",
                "setup",
                "benchmark",
                "correct",
                "tokens_used",
                "latency_ms",
                "estimated_cost",
                "error",
            ],
        )
        writer.writeheader()
        for item in state.problems:
            writer.writerow(
                {
                    "problem_id": item.problem_id,
                    "setup": item.setup,
                    "benchmark": item.benchmark,
                    "correct": item.correct,
                    "tokens_used": item.tokens_used,
                    "latency_ms": item.latency_ms,
                    "estimated_cost": item.estimated_cost,
                    "error": item.error or "",
                }
            )


def _build_setup_result(setup: str, benchmark: str, problems: list[ProblemResult]) -> SetupResult:
    """Aggregate problem-level results into setup-level metrics."""
    total = len(problems)
    correct_count = sum(1 for p in problems if p.correct)
    total_tokens = sum(p.tokens_used for p in problems)
    total_latency = sum(p.latency_ms for p in problems)
    total_cost = sum(p.estimated_cost for p in problems)

    return SetupResult(
        setup=setup,
        benchmark=benchmark,
        accuracy=(correct_count / total) if total else 0.0,
        total_problems=total,
        correct_count=correct_count,
        avg_tokens=(total_tokens / total) if total else 0.0,
        avg_latency_ms=(total_latency / total) if total else 0.0,
        total_tokens=total_tokens,
        total_estimated_cost=total_cost,
        problems=problems,
    )


def _run_setup_for_problem(
    state: ProgressState,
    completed: set[str],
    problem_id: str,
    prompt: str,
    entry_point: str,
    test_source: str,
    ground_truth: str,
    timeout: int,
    setup: str,
    execute_fn: Any,
) -> None:
    """Execute one setup for one HumanEval problem with error-safe state updates."""
    setup_key = f"{problem_id}:{setup}"
    if setup_key in completed:
        return

    try:
        result = execute_fn()
        predicted_text = str(result.get("final_answer", "") or result.get("response", ""))
        candidate_code = extract_code_from_response(predicted_text, entry_point)
        test_program = f"{test_source}\n\ncheck({entry_point})\n"
        passed = execute_code(candidate_code, test_program, timeout=timeout)
        problem_result = ProblemResult(
            problem_id=problem_id,
            setup=setup,
            benchmark="humaneval",
            correct=passed,
            predicted_answer=candidate_code,
            ground_truth=ground_truth,
            tokens_used=int(result.get("tokens_used", 0)),
            latency_ms=int(result.get("latency_ms", 0)),
            estimated_cost=float(result.get("estimated_cost", 0.0)),
            error=None,
        )
    except Exception as exc:  # pragma: no cover - runtime/LLM/backend failures are expected
        logger.exception("Problem %s setup %s failed. Continuing.", problem_id, setup)
        problem_result = ProblemResult(
            problem_id=problem_id,
            setup=setup,
            benchmark="humaneval",
            correct=False,
            predicted_answer="",
            ground_truth=ground_truth,
            tokens_used=0,
            latency_ms=0,
            estimated_cost=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    state.problems.append(problem_result)
    state.completed_problem_ids.append(setup_key)
    completed.add(setup_key)


def _resolve_model_name(parishad: Parishad) -> str:
    """Resolve human-readable model names from active slot configuration."""
    slots = parishad.engine.model_config.slots
    values = sorted({cfg.model_id for cfg in slots.values() if getattr(cfg, "model_id", None)})
    return ", ".join(values) if values else "unknown-model"


def _configure_logging(level: str) -> None:
    """Configure script logging once."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
