"""Experiment Zero orchestrator for chunked GSM8K and HumanEval runs.

Usage examples:
    python scripts/experiment_zero.py --benchmark gsm8k --part 1
    python scripts/experiment_zero.py --benchmark humaneval --part 2
    python scripts/experiment_zero.py --benchmark both --part all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure sibling scripts and local src/ are importable.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for candidate in (SCRIPT_DIR, SRC_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import experiment_zero_gsm8k as gsm8k_runner
import experiment_zero_humaneval as humaneval_runner

logger = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> int:
    """Run selected benchmark(s) and partition(s)."""
    _configure_logging(args.log_level)

    parts = [1, 2, 3, 4] if args.part == "all" else [int(args.part)]
    benchmarks = ["gsm8k", "humaneval"] if args.benchmark == "both" else [args.benchmark]

    failures = 0
    for benchmark in benchmarks:
        for part in parts:
            logger.info("Starting %s part %d", benchmark, part)
            child_args = argparse.Namespace(
                part=part,
                chunk_size=args.chunk_size,
                total_problems=args.total_problems,
                seed=args.seed,
                output_dir=args.output_dir,
                resume=args.resume,
                config=args.config,
                mode=args.mode,
                profile=args.profile,
                trace_dir=args.trace_dir,
                no_retry=args.no_retry,
                log_level=args.log_level,
                split=args.split,
                timeout=args.timeout,
                dataset_path=args.gsm8k_dataset_path if benchmark == "gsm8k" else args.humaneval_dataset_path,
            )
            try:
                if benchmark == "gsm8k":
                    exit_code = gsm8k_runner.run_experiment(child_args)
                else:
                    exit_code = humaneval_runner.run_experiment(child_args)
                if exit_code != 0:
                    failures += 1
            except KeyboardInterrupt:
                logger.warning("Interrupted by user while running %s part %d", benchmark, part)
                return 130
            except Exception:
                failures += 1
                logger.exception("Run failed for %s part %d", benchmark, part)

    if failures:
        logger.error("Experiment Zero completed with %d failed chunk(s)", failures)
        return 1

    logger.info("Experiment Zero completed successfully")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create command-line parser."""
    parser = argparse.ArgumentParser(description="Experiment Zero orchestrator")
    parser.add_argument("--benchmark", choices=["gsm8k", "humaneval", "both"], default="both")
    parser.add_argument("--part", choices=["1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--total-problems", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/experiment_zero")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--config", type=str, default="core")
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--no-retry", action="store_true")
    parser.add_argument("--split", type=str, default="test", help="GSM8K split")
    parser.add_argument("--gsm8k-dataset-path", type=str, default=None)
    parser.add_argument("--humaneval-dataset-path", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=10, help="HumanEval subprocess timeout (seconds)")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def _configure_logging(level: str) -> None:
    """Configure script logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
