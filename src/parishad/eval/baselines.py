"""Baseline runners for direct, CoT, and Parishad pipeline execution."""

from __future__ import annotations

import logging
import time
from typing import Any

from parishad.roles.base import Slot

logger = logging.getLogger(__name__)


def run_direct(
    model_runner: Any,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """Run direct prompting through the current model runner."""
    slot = _resolve_slot(model_runner)
    started = time.perf_counter()
    response, tokens_used, model_id = model_runner.generate(
        system_prompt="You are a helpful and concise assistant.",
        user_message=prompt,
        slot=slot,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    return {
        "response": response,
        "tokens_used": int(tokens_used),
        "latency_ms": latency_ms,
        "model_id": model_id,
        "slot": slot.value,
        "estimated_cost": _estimate_generation_cost(model_runner, slot, int(tokens_used)),
    }


def run_cot(
    model_runner: Any,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """Run chain-of-thought prompting through the model runner."""
    cot_prompt = "Let's think step by step.\n\n" + prompt
    return run_direct(
        model_runner=model_runner,
        prompt=cot_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def run_parishad_pipeline(parishad: Any, query: str) -> dict[str, Any]:
    """Run a query via Parishad high-level API and extract execution metrics."""
    started = time.perf_counter()
    trace = parishad.run(query)
    wall_latency_ms = int((time.perf_counter() - started) * 1000)

    final_answer = ""
    if trace.final_answer and trace.final_answer.final_answer:
        final_answer = trace.final_answer.final_answer

    roles_run = [role.role for role in trace.roles]
    estimated_cost = _estimate_trace_cost(parishad, trace)

    return {
        "response": final_answer,
        "tokens_used": int(trace.total_tokens),
        "latency_ms": int(trace.total_latency_ms or wall_latency_ms),
        "roles_run": roles_run,
        "final_answer": final_answer,
        "trace": trace,
        "estimated_cost": estimated_cost,
    }


def _resolve_slot(model_runner: Any) -> Slot:
    """Select an available slot for direct baseline calls."""
    preferred = ["mid", "big", "small"]
    slots = getattr(getattr(model_runner, "config", object()), "slots", {})
    for name in preferred:
        if name in slots:
            return Slot(name)

    available = ", ".join(sorted(slots.keys())) if isinstance(slots, dict) else "<none>"
    raise ValueError(f"No supported slot found for baseline generation. Available: {available}")


def _estimate_generation_cost(model_runner: Any, slot: Slot, tokens_used: int) -> float:
    """Estimate token-weight cost for single generation calls."""
    weight = 1.0
    try:
        weight = float(model_runner.get_token_weight(slot))
    except Exception:
        logger.debug("Falling back to token weight=1.0 for slot %s", slot.value)
    return tokens_used * weight


def _estimate_trace_cost(parishad: Any, trace: Any) -> float:
    """Estimate weighted token cost from per-role trace metadata."""
    total = 0.0
    model_runner = getattr(getattr(parishad, "engine", object()), "model_runner", None)
    for role_out in trace.roles:
        tokens = int(getattr(role_out.metadata, "tokens_used", 0) or 0)
        slot_obj = getattr(role_out.metadata, "slot", None)
        slot_name = getattr(slot_obj, "value", str(slot_obj)) if slot_obj is not None else "mid"
        try:
            slot = Slot(slot_name)
        except ValueError:
            slot = Slot.MID

        weight = 1.0
        if model_runner is not None:
            try:
                weight = float(model_runner.get_token_weight(slot))
            except Exception:
                weight = 1.0
        total += tokens * weight
    return total
