"""GSM8K benchmark evaluation utilities."""

from __future__ import annotations

import logging
import importlib
import random
import re
import csv
import json
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LOCAL_FILE_ENCODING = "utf-8-sig"


def load_gsm8k(
    split: str = "test",
    n_problems: int | None = None,
    seed: int = 42,
    dataset_path: str | None = None,
) -> list[dict[str, str]]:
    """Load GSM8K problems from Hugging Face datasets.

    Args:
        split: Dataset split name. Usually ``test`` for evaluation.
        n_problems: Optional deterministic sample size.
        seed: Seed used for deterministic sampling.
        dataset_path: Optional local dataset file path (JSON/JSONL/CSV).

    Returns:
        List of problem dictionaries with keys: ``question`` and ``answer``.

    Raises:
        ImportError: If the optional benchmark dependency is not installed.
        RuntimeError: If dataset loading fails.
    """
    if dataset_path:
        records = _load_gsm8k_from_local_file(dataset_path)
    else:
        try:
            datasets_module = importlib.import_module("datasets")
            load_dataset = datasets_module.load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. Install benchmark extras: pip install -e .[benchmark]"
            ) from exc

        try:
            ds = load_dataset("gsm8k", "main", split=split)
        except Exception as exc:  # pragma: no cover - depends on remote/data cache
            raise RuntimeError(f"Failed to load GSM8K split '{split}': {exc}") from exc

        records = [{"question": str(row["question"]), "answer": str(row["answer"])} for row in ds]

    if n_problems is None or n_problems >= len(records):
        return records

    rng = random.Random(seed)
    selected_indices = rng.sample(range(len(records)), n_problems)
    return [records[i] for i in selected_indices]


def extract_numeric_answer(text: str) -> float | None:
    """Extract a numeric answer from model output.

    Strategy order:
    1. ``#### <number>``
    2. ``the answer is <number>``
    3. last numeric token in text

    Args:
        text: Model output text.

    Returns:
        Extracted float value or ``None`` when extraction fails.
    """
    if not text:
        return None

    normalized = text.replace(",", "")

    strict_match = re.search(r"####\s*([-+]?\d*\.?\d+)", normalized)
    if strict_match:
        return _safe_float(strict_match.group(1))

    answer_match = re.search(
        r"(?:the\s+answer\s+is|answer\s*:)\s*([-+]?\d*\.?\d+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if answer_match:
        return _safe_float(answer_match.group(1))

    all_numbers = re.findall(r"[-+]?\d*\.?\d+", normalized)
    if all_numbers:
        return _safe_float(all_numbers[-1])

    return None


def extract_ground_truth(answer_text: str) -> float:
    """Extract the ground-truth numeric answer from GSM8K answer text.

    Args:
        answer_text: GSM8K formatted answer, usually ending with ``#### <number>``.

    Returns:
        Ground-truth value as float.

    Raises:
        ValueError: If the ground truth cannot be parsed.
    """
    strict_match = re.search(r"####\s*([-+]?\d*\.?\d+)", answer_text.replace(",", ""))
    if strict_match:
        value = _safe_float(strict_match.group(1))
        if value is None:
            raise ValueError(f"Unable to convert GSM8K ground truth: {answer_text}")
        return value

    fallback = extract_numeric_answer(answer_text)
    if fallback is None:
        raise ValueError(f"Unable to extract GSM8K ground truth from: {answer_text}")
    return fallback


def evaluate_gsm8k(predicted: str, ground_truth: str) -> bool:
    """Evaluate GSM8K prediction against ground truth.

    Args:
        predicted: Model response text.
        ground_truth: GSM8K answer field text.

    Returns:
        ``True`` if absolute numeric difference is within ``0.01``.
    """
    pred_val = extract_numeric_answer(predicted)
    if pred_val is None:
        return False

    try:
        gt_val = extract_ground_truth(ground_truth)
    except ValueError:
        logger.warning("Ground truth parse failed for GSM8K row")
        return False

    return abs(pred_val - gt_val) <= 0.01


def _safe_float(value: Any) -> float | None:
    """Best-effort float conversion helper."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_gsm8k_from_local_file(dataset_path: str) -> list[dict[str, str]]:
    """Load GSM8K records from a local JSON/JSONL/CSV file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"GSM8K dataset file not found: {dataset_path}")

    suffix = path.suffix.lower()
    rows: list[dict[str, Any]] = []

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

    records: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        question = row.get("question") if isinstance(row, dict) else None
        answer = row.get("answer") if isinstance(row, dict) else None
        if question is None or answer is None:
            raise ValueError(f"Row {idx} is missing required keys: question, answer")
        records.append({"question": str(question), "answer": str(answer)})

    return records
