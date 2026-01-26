"""
Ensemble checker combining multiple verification methods.

This is the main Checker interface that combines:
1. Deterministic checks (free)
2. Retrieval-based checks (cheap)
3. LLM-based verification (expensive)

Key research question: What combination minimizes cost
while maximizing error detection?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .deterministic import DeterministicChecker, DeterministicCheckResults
from .retrieval import RetrievalChecker, FactCheckResult


logger = logging.getLogger(__name__)


class CheckerVerdict(str, Enum):
    """Final verdict from ensemble checker."""
    
    PASS = "pass"           # All checks passed
    FAIL = "fail"           # Critical failure detected
    UNCERTAIN = "uncertain" # Needs LLM review
    NEEDS_RETRY = "retry"   # Fixable issue, should retry


@dataclass
class EnsembleCheckResult:
    """Result from ensemble checking."""
    
    verdict: CheckerVerdict
    deterministic: DeterministicCheckResults
    retrieval_results: list[FactCheckResult]
    llm_verdict: Optional[dict[str, Any]] = None
    confidence: float = 0.0
    feedback: str = ""
    cost_tokens: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "feedback": self.feedback,
            "cost_tokens": self.cost_tokens,
            "deterministic": self.deterministic.to_dict(),
            "retrieval_count": len(self.retrieval_results),
            "llm_used": self.llm_verdict is not None,
        }


class CheckerEnsemble:
    """
    Ensemble checker combining multiple verification methods.
    
    Implements tiered checking:
    1. Deterministic checks (always run, free)
    2. Retrieval checks (run if enabled, cheap)
    3. LLM checks (run if needed, expensive)
    
    The goal is to catch errors with minimal LLM cost.
    """
    
    def __init__(
        self,
        model_runner: Optional[Any] = None,
        enable_retrieval: bool = True,
        enable_llm: bool = True,
        llm_threshold: float = 0.7,
        retrieval_checker: Optional[RetrievalChecker] = None,
        checker_mode: str = "full",
    ):
        """
        Initialize ensemble checker.
        
        Args:
            model_runner: ModelRunner for LLM-based checks
            enable_retrieval: Whether to run retrieval checks
            enable_llm: Whether to run LLM checks
            llm_threshold: Confidence threshold for LLM checks
            retrieval_checker: Custom retrieval checker
            checker_mode: Checking mode - "none", "deterministic", or "full" (default)
        """
        self.model_runner = model_runner
        self.enable_retrieval = enable_retrieval
        self.enable_llm = enable_llm
        self.llm_threshold = llm_threshold
        self.checker_mode = checker_mode
        
        self.deterministic = DeterministicChecker()
        self.retrieval = retrieval_checker or RetrievalChecker()
    
    def check(
        self,
        text: str,
        task_type: str = "general",
        context: dict[str, Any] | None = None,
        force_llm: bool = False,
    ) -> EnsembleCheckResult:
        """
        Run tiered ensemble checking on output text.
        
        Executes checks in order of cost:
        1. Deterministic (free) - always runs
        2. Retrieval (cheap) - runs if enabled
        3. LLM (expensive) - runs only if needed
        
        Args:
            text: Output text to check (will be truncated for LLM checks)
            task_type: Type of task (code, math, general, qa)
            context: Additional context (query, schema, etc.)
            force_llm: Force LLM check regardless of other results
            
        Returns:
            EnsembleCheckResult with verdict, confidence, and details
        """
        context = context or {}
        
        # Task 2: Implement checker_mode behavior
        if self.checker_mode == "none":
            # Return empty pass verdict immediately
            logger.debug("Checker mode 'none': skipping all checks")
            return EnsembleCheckResult(
                verdict=CheckerVerdict.PASS,
                deterministic=DeterministicCheckResults(
                    checks=[],
                    all_passed=True,
                    critical_failure=False,
                ),
                retrieval_results=[],
                confidence=1.0,
                feedback="Checker mode 'none': all checks skipped",
                cost_tokens=0,
            )
        
        total_cost = 0
        text_len = len(text)
        
        logger.debug(f"Ensemble check: task_type={task_type}, text_len={text_len}, mode={self.checker_mode}")
        
        # === Stage 1: Deterministic checks (free) ===
        det_results = self.deterministic.run_all(text, task_type, context)
        
        logger.debug(
            f"Deterministic: all_passed={det_results.all_passed}, "
            f"critical={det_results.critical_failure}"
        )
        
        # Critical failure = immediate fail
        if det_results.critical_failure:
            logger.info(f"Critical failure detected: {det_results.failure_reason}")
            return EnsembleCheckResult(
                verdict=CheckerVerdict.FAIL,
                deterministic=det_results,
                retrieval_results=[],
                confidence=1.0,
                feedback=det_results.failure_reason or "Critical check failure",
                cost_tokens=0,
            )
        
        # Task 2: If checker_mode is "deterministic", stop here (no retrieval, no LLM)
        if self.checker_mode == "deterministic":
            logger.debug("Checker mode 'deterministic': skipping retrieval and LLM checks")
            verdict = CheckerVerdict.PASS if det_results.all_passed else CheckerVerdict.FAIL
            return EnsembleCheckResult(
                verdict=verdict,
                deterministic=det_results,
                retrieval_results=[],
                confidence=0.9 if det_results.all_passed else 0.5,
                feedback="Deterministic checks only" if det_results.all_passed else det_results.failure_reason or "Some checks failed",
                cost_tokens=0,
            )
        
        # === Stage 2: Retrieval checks (cheap) ===
        retrieval_results: list[FactCheckResult] = []
        retrieval_confidence = 1.0
        
        if self.enable_retrieval and task_type in ("general", "qa"):
            retrieval_results = self.retrieval.check_all_claims(text, max_claims=5)
            
            # Check for contradicted claims
            contradicted = [r for r in retrieval_results if r.supported is False]
            if contradicted:
                # Reduce confidence based on contradictions
                retrieval_confidence -= 0.2 * len(contradicted)
        
        # === Stage 3: LLM checks (expensive) ===
        llm_verdict = None
        llm_confidence = 0.0
        
        # Decide if LLM check is needed
        needs_llm = (
            force_llm
            or not det_results.all_passed
            or retrieval_confidence < 0.8
            or task_type in ("code", "math")  # High-stakes tasks
        )
        
        if needs_llm and self.enable_llm and self.model_runner is not None:
            llm_verdict, llm_cost = self._run_llm_check(text, task_type, context)
            total_cost += llm_cost
            llm_confidence = llm_verdict.get("confidence", 0.5) if llm_verdict else 0.0
        
        # === Combine results ===
        return self._make_verdict(
            det_results,
            retrieval_results,
            llm_verdict,
            retrieval_confidence,
            llm_confidence,
            total_cost,
        )
    
    def _run_llm_check(
        self,
        text: str,
        task_type: str,
        context: dict,
    ) -> tuple[Optional[dict], int]:
        """
        Run LLM-based verification.
        
        Args:
            text: Text to verify
            task_type: Type of task
            context: Additional context
            
        Returns:
            Tuple of (verdict dict, token cost)
        """
        if self.model_runner is None:
            return None, 0
        
        # Build verification prompt
        system_prompt = """You are a verification assistant. Check the following output for:
1. Correctness: Is the answer/solution correct?
2. Completeness: Does it fully address the question?
3. Consistency: Are there internal contradictions?
4. Safety: Any harmful or inappropriate content?

Respond in JSON:
{
    "correct": true/false/null,
    "complete": true/false,
    "consistent": true/false,
    "safe": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of issues found"],
    "suggestion": "how to fix if needed"
}"""
        
        user_prompt = f"""Task type: {task_type}
Original query: {context.get('query', 'N/A')}

Output to verify:
{text[:2000]}  # Truncate to limit cost

Analyze this output and provide your verification JSON."""
        
        try:
            from ..roles.base import Slot
            
            # Use SMALL model for verification (cost-efficient)
            response = self.model_runner.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                slot=Slot.SMALL,
                max_tokens=300,
            )
            
            # Parse response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                verdict = json.loads(json_match.group())
                # Estimate tokens used
                cost = len(system_prompt.split()) + len(user_prompt.split()) + 300
                return verdict, cost
            
            return None, 0
            
        except Exception:
            return None, 0
    
    def _make_verdict(
        self,
        det_results: DeterministicCheckResults,
        retrieval_results: list[FactCheckResult],
        llm_verdict: Optional[dict],
        retrieval_confidence: float,
        llm_confidence: float,
        cost: int,
    ) -> EnsembleCheckResult:
        """
        Combine all results into final verdict.
        
        Args:
            det_results: Deterministic check results
            retrieval_results: Retrieval fact check results
            llm_verdict: LLM verification result
            retrieval_confidence: Confidence from retrieval
            llm_confidence: Confidence from LLM
            cost: Total token cost
            
        Returns:
            Final ensemble result
        """
        # Start with deterministic assessment
        if det_results.critical_failure:
            verdict = CheckerVerdict.FAIL
            confidence = 1.0
            feedback = det_results.failure_reason or "Critical failure"
        
        elif det_results.all_passed and retrieval_confidence >= 0.9:
            # All deterministic passed and retrieval looks good
            if llm_verdict is None:
                verdict = CheckerVerdict.PASS
                confidence = 0.8
                feedback = "Passed deterministic and retrieval checks"
            elif llm_verdict.get("correct") is True:
                verdict = CheckerVerdict.PASS
                confidence = llm_confidence
                feedback = "Passed all verification stages"
            elif llm_verdict.get("correct") is False:
                verdict = CheckerVerdict.FAIL
                confidence = llm_confidence
                feedback = "; ".join(llm_verdict.get("issues", ["LLM detected issues"]))
            else:
                verdict = CheckerVerdict.UNCERTAIN
                confidence = 0.5
                feedback = "LLM verification inconclusive"
        
        elif not det_results.all_passed:
            # Some deterministic checks failed (non-critical)
            failed = [c for c in det_results.checks if not c.passed]
            
            # If LLM says it's fine, trust LLM
            if llm_verdict and llm_verdict.get("correct") is True:
                verdict = CheckerVerdict.PASS
                confidence = llm_confidence * 0.8  # Slight discount
                feedback = "LLM verified despite minor issues"
            else:
                verdict = CheckerVerdict.NEEDS_RETRY
                confidence = 0.6
                feedback = "; ".join(c.message for c in failed[:3])
        
        else:
            # Mixed signals
            verdict = CheckerVerdict.UNCERTAIN
            confidence = 0.5
            feedback = "Mixed verification results"
        
        return EnsembleCheckResult(
            verdict=verdict,
            deterministic=det_results,
            retrieval_results=retrieval_results,
            llm_verdict=llm_verdict,
            confidence=confidence,
            feedback=feedback,
            cost_tokens=cost,
        )
    
    def quick_check(self, text: str, task_type: str = "general") -> bool:
        """
        Quick pass/fail check using only deterministic checks.
        
        Args:
            text: Text to check
            task_type: Type of task
            
        Returns:
            True if passed, False otherwise
        """
        det_results = self.deterministic.run_all(text, task_type)
        return det_results.all_passed and not det_results.critical_failure


# === Convenience function for role integration ===

def run_checker_ensemble(
    content: str,
    check_type: str = "general",
    context: Optional[dict[str, Any]] = None,
    model_runner: Optional[Any] = None,
    enable_retrieval: bool = True,
    enable_llm: bool = True,
    checker_mode: str = "full",
) -> dict[str, Any]:
    """
    Run ensemble checker and return structured output for role integration.
    
    This is the main entry point for Checker roles to use the ensemble.
    Returns a dictionary compatible with RoleOutput.core_output format.
    
    Args:
        content: The text content to check
        check_type: Type of check (code, math, general, qa)
        context: Additional context (query, expected schema, etc.)
        model_runner: Optional ModelRunner for LLM checks
        enable_retrieval: Whether to enable retrieval-based checks
        enable_llm: Whether to enable LLM-based checks
        checker_mode: Checking mode - \"none\", \"deterministic\", or \"full\" (default)
        
    Returns:
        Dict with keys:
        - flags: List of issue identifiers
        - must_fix: Boolean, True if critical issues found
        - evidence: List of evidence strings
        - suggested_edits: List of suggested fixes
        - verdict: The overall verdict string
        - confidence: Confidence score 0-1
        - cost_tokens: Token cost of checks
    """
    context = context or {}
    
    # Create ensemble with provided configuration
    ensemble = CheckerEnsemble(
        model_runner=model_runner,
        enable_retrieval=enable_retrieval,
        enable_llm=enable_llm,
        checker_mode=checker_mode,
    )
    
    # Run the check
    result = ensemble.check(
        text=content,
        task_type=check_type,
        context=context,
    )
    
    # Extract flags from deterministic results
    flags: list[str] = []
    evidence: list[str] = []
    suggested_edits: list[str] = []
    
    # Process deterministic check results
    for check in result.deterministic.checks:
        if not check.passed:
            flags.append(check.name)
            evidence.append(f"[{check.name}] {check.message}")
            # Check details for suggestion
            suggestion = check.details.get("suggestion") if check.details else None
            if suggestion:
                suggested_edits.append(suggestion)
    
    # Process retrieval results
    for ret_result in result.retrieval_results:
        if ret_result.supported is False:
            flags.append(f"claim_contradicted:{ret_result.claim[:30]}")
            evidence.append(
                f"[Retrieval] Claim '{ret_result.claim}' contradicted by: "
                f"{ret_result.evidence[:100]}..."
            )
            suggested_edits.append(f"Review claim: {ret_result.claim}")
        elif ret_result.supported is True:
            evidence.append(f"[Retrieval] Claim '{ret_result.claim}' supported")
    
    # Process LLM verdict if present
    if result.llm_verdict:
        llm_issues = result.llm_verdict.get("issues", [])
        for issue in llm_issues:
            flags.append(f"llm_issue:{issue[:20]}")
            evidence.append(f"[LLM] {issue}")
        
        llm_suggestion = result.llm_verdict.get("suggestion")
        if llm_suggestion:
            suggested_edits.append(f"[LLM] {llm_suggestion}")
    
    # Determine must_fix
    must_fix = (
        result.verdict == CheckerVerdict.FAIL
        or result.deterministic.critical_failure
    )
    
    return {
        "flags": flags,
        "must_fix": must_fix,
        "evidence": evidence,
        "suggested_edits": suggested_edits,
        "verdict": result.verdict.value,
        "confidence": result.confidence,
        "feedback": result.feedback,
        "cost_tokens": result.cost_tokens,
    }
