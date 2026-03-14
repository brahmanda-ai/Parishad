"""HumanEval benchmark evaluation utilities."""

from __future__ import annotations

import logging
import importlib
import random
import re
import subprocess
import sys
import tempfile
import csv
import json
from pathlib import Path

logger = logging.getLogger(__name__)

LOCAL_FILE_ENCODING = "utf-8-sig"


def load_humaneval(
    n_problems: int | None = None,
    seed: int = 42,
    dataset_path: str | None = None,
) -> list[dict[str, str]]:
    """Load HumanEval problems from Hugging Face datasets.

    Args:
        n_problems: Optional deterministic sample size.
        seed: Seed used for deterministic sampling.
        dataset_path: Optional local dataset file path (JSON/JSONL/CSV).

    Returns:
        List of dictionaries with keys:
        ``task_id``, ``prompt``, ``entry_point``, ``test``, ``canonical_solution``.

    Raises:
        ImportError: If the optional benchmark dependency is not installed.
        RuntimeError: If dataset loading fails.
    """
    if dataset_path:
        records = _load_humaneval_from_local_file(dataset_path)
    else:
        try:
            datasets_module = importlib.import_module("datasets")
            load_dataset = datasets_module.load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. Install benchmark extras: pip install -e .[benchmark]"
            ) from exc

        try:
            ds = load_dataset("openai_humaneval", split="test")
        except Exception as exc:  # pragma: no cover - depends on remote/data cache
            raise RuntimeError(f"Failed to load HumanEval: {exc}") from exc

        records = [
            {
                "task_id": str(row["task_id"]),
                "prompt": str(row["prompt"]),
                "entry_point": str(row["entry_point"]),
                "test": str(row["test"]),
                "canonical_solution": str(row["canonical_solution"]),
            }
            for row in ds
        ]

    if n_problems is None or n_problems >= len(records):
        return records

    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), n_problems)
    return [records[i] for i in indices]


def execute_code(code: str, test: str, timeout: int = 10) -> bool:
    """Execute generated code and tests in an isolated subprocess.

    Args:
        code: Generated Python code.
        test: Test program source to append.
        timeout: Maximum execution duration in seconds.

    Returns:
        ``True`` when subprocess exits with status 0.

    Safety:
        Execution is delegated to a subprocess and bounded by timeout.
        The main process never uses ``eval`` or ``exec``.
    """
    if not code.strip():
        return False

    source = f"{code}\n\n{test}\n"
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
            fh.write(source)
            tmp_path = Path(fh.name)

        proc = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            logger.debug("HumanEval execution failed: %s", proc.stderr.strip())
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning("HumanEval execution timed out after %ss", timeout)
        return False
    except OSError as exc:
        logger.exception("Failed to execute HumanEval snippet: %s", exc)
        return False
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Unable to remove temp file %s", tmp_path)


def extract_code_from_response(response: str, entry_point: str) -> str:
    """Extract candidate Python code from a model response.

    Handles fenced code blocks, plain code, and mixed prose/code outputs.

    Args:
        response: Raw model response text.
        entry_point: Required function name for the HumanEval task.

    Returns:
        Extracted code string (best-effort).
    """
    if not response.strip():
        return ""

    fenced_blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    if fenced_blocks:
        best_block = _pick_best_block(fenced_blocks, entry_point)
        return best_block.strip()

    candidate = response.strip()
    start_idx = candidate.find(f"def {entry_point}")
    if start_idx >= 0:
        return candidate[start_idx:].strip()

    return candidate


def _pick_best_block(blocks: list[str], entry_point: str) -> str:
    """Select the most likely code block for a target entry point."""
    target = f"def {entry_point}"
    for block in blocks:
        if target in block:
            return block
    return blocks[0]


def _load_humaneval_from_local_file(dataset_path: str) -> list[dict[str, str]]:
    """Load HumanEval rows from a local JSON/JSONL/CSV file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"HumanEval dataset file not found: {dataset_path}")

    suffix = path.suffix.lower()
    rows: list[dict[str, str]] = []

    if suffix == ".jsonl":
        with path.open("r", encoding=LOCAL_FILE_ENCODING) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding=LOCAL_FILE_ENCODING))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
            rows = payload["data"]
        else:
            raise ValueError("JSON dataset must be a list or a dict with a 'data' list")
    elif suffix == ".csv":
        with path.open("r", encoding=LOCAL_FILE_ENCODING, newline="") as fh:
            rows = list(csv.DictReader(fh))
    else:
        raise ValueError("Unsupported dataset file type. Use .jsonl, .json, or .csv")

    required = ["task_id", "prompt", "entry_point", "test", "canonical_solution"]
    records: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        missing = [key for key in required if key not in row or row.get(key) is None]
        if missing:
            raise ValueError(f"Row {idx} missing required keys: {', '.join(missing)}")
        records.append(
            {
                "task_id": str(row["task_id"]),
                "prompt": str(row["prompt"]),
                "entry_point": str(row["entry_point"]),
                "test": str(row["test"]),
                "canonical_solution": str(row["canonical_solution"]),
            }
        )
    return records
