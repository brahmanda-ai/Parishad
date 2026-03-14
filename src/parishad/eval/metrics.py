"""Metric calculation and result aggregation utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProblemResult(BaseModel):
    """Result for a single benchmark problem."""

    problem_id: str
    setup: str = Field(description="direct, cot, or pipeline")
    benchmark: str = Field(description="gsm8k or humaneval")
    correct: bool
    predicted_answer: str
    ground_truth: str
    tokens_used: int = 0
    latency_ms: int = 0
    estimated_cost: float = 0.0
    error: str | None = None


class SetupResult(BaseModel):
    """Aggregate result for one setup and one benchmark."""

    setup: str
    benchmark: str
    accuracy: float
    total_problems: int
    correct_count: int
    avg_tokens: float
    avg_latency_ms: float
    total_tokens: int
    total_estimated_cost: float
    problems: list[ProblemResult]


class ExperimentResult(BaseModel):
    """Complete experiment output payload."""

    experiment_name: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_name: str
    seed: int
    hardware: str
    results: dict[str, dict[str, SetupResult]]
    decision: str
    decision_reason: str


def calculate_accuracy(results: list[ProblemResult]) -> float:
    """Compute simple accuracy as ``correct / total``."""
    if not results:
        return 0.0
    correct = sum(1 for item in results if item.correct)
    return correct / len(results)


def calculate_delta(pipeline_acc: float, direct_acc: float) -> float:
    """Compute pipeline accuracy improvement over direct baseline."""
    return pipeline_acc - direct_acc


def determine_gate_decision(gsm8k_delta: float, humaneval_delta: float) -> tuple[str, str]:
    """Compute go/no-go decision using lower benchmark delta."""
    lower = min(gsm8k_delta, humaneval_delta)

    if lower >= 0.10:
        return "green", f"Lower delta {lower:.3f} >= 0.10"
    if lower >= 0.05:
        return "yellow", f"Lower delta {lower:.3f} >= 0.05"
    if lower >= 0.02:
        return "orange", f"Lower delta {lower:.3f} >= 0.02"
    if lower >= -0.01:
        return "red", f"Lower delta {lower:.3f} >= -0.01"
    return "stop", f"Lower delta {lower:.3f} < -0.01"


def save_results(result: ExperimentResult, output_dir: str) -> None:
    """Save complete and per-benchmark JSON artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_path = out_path / "summary.json"
    summary_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    for benchmark_name, setup_map in result.results.items():
        bench_payload = {
            "benchmark": benchmark_name,
            "setups": {
                setup_name: setup_result.model_dump(mode="json")
                for setup_name, setup_result in setup_map.items()
            },
        }
        bench_path = out_path / f"{benchmark_name}_results.json"
        bench_path.write_text(json.dumps(bench_payload, indent=2), encoding="utf-8")

    logger.info("Saved experiment results to %s", out_path)


def print_summary(result: ExperimentResult) -> None:
    """Log a compact experiment summary table."""
    logger.info("=" * 72)
    logger.info("Experiment: %s", result.experiment_name)
    logger.info("Model: %s", result.model_name)
    logger.info("Decision: %s (%s)", result.decision, result.decision_reason)
    logger.info("=" * 72)

    for benchmark_name, setup_map in result.results.items():
        logger.info("Benchmark: %s", benchmark_name)
        for setup_name, setup in setup_map.items():
            logger.info(
                "  %-10s acc=%6.2f%%  correct=%3d/%-3d  avg_tokens=%8.1f  avg_latency_ms=%8.1f",
                setup_name,
                setup.accuracy * 100.0,
                setup.correct_count,
                setup.total_problems,
                setup.avg_tokens,
                setup.avg_latency_ms,
            )
