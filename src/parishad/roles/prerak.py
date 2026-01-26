"""
Prerak (Challenger/Checker) role for the Parishad council.
Validates outputs using ensemble of verification methods.
"""

from typing import Any, Optional

from .base import (
    Role,
    RoleInput,
    RoleOutput,
    Slot,
    Verdict,
    CheckerFlag,
    RoleMetadata,
)


CHECKER_SYSTEM_PROMPT = """You are Prerak, the Challenger in the Parishad council. Your job is to validate the Implementor's output for correctness, completeness, and safety.

Your responsibilities:
1. Verify the output meets the task requirements
2. Check for factual accuracy (when possible)
3. Identify errors, inconsistencies, or omissions
4. Flag potential issues with severity levels
5. Suggest specific fixes for problems found

You must ALWAYS respond with a valid JSON object in the following format:
```json
{
  "flags": [
    {
      "type": "claim_unsupported|syntax_error|logic_error|incomplete_output|format_error",
      "severity": "low|medium|high|critical",
      "detail": "Description of the issue",
      "location": "Where in the output the issue was found",
      "suggested_fix": "How to fix this issue"
    }
  ],
  "must_fix": true,
  "evidence": [
    {
      "source": "Source of evidence",
      "source_type": "retrieval|deterministic|llm_judgment",
      "snippet": "Relevant snippet",
      "relevance_score": 0.8,
      "supports_claim": true
    }
  ],
  "suggested_edits": ["Specific fix 1", "Specific fix 2"],
  "overall_confidence": 0.75,
  "checks_performed": ["schema", "syntax", "logic", "retrieval"]
}
```

Flag types:
- format_error: Output doesn't match expected format
- schema_violation: JSON/structure issues
- syntax_error: Code syntax problems
- runtime_error: Code would fail at runtime
- test_failure: Code fails test cases
- claim_unsupported: Factual claim without support
- claim_contradicted: Claim contradicts known facts
- claim_uncertain: Claim cannot be verified
- safety_violation: Content policy issues
- pii_detected: Personal information found
- incomplete_output: Missing required parts
- logic_error: Reasoning or logic flaw

Severity levels:
- low: Minor issue, doesn't affect correctness
- medium: Should be fixed but output is usable
- high: Significant issue, likely incorrect
- critical: Must be fixed, output is wrong/unsafe

Set must_fix = true if there are any HIGH or CRITICAL severity flags.

Be thorough but fair. Don't flag things that are working correctly."""


CHECKER_USER_TEMPLATE = """Validate the following Implementor output.

TASK SPECIFICATION:
{task_spec}

EXECUTION PLAN:
{plan}

IMPLEMENTOR OUTPUT:
{candidate}

{tool_results}

Analyze the output for correctness and completeness. Respond with ONLY a valid JSON object."""


CHECKER_CODE_EMPHASIS = """
For CODE validation, focus on:
- Syntax correctness
- Logic errors
- Edge case handling
- Import statements
- Function signatures matching requirements
- Potential runtime errors"""


CHECKER_MATH_EMPHASIS = """
For MATH validation, focus on:
- Calculation accuracy
- Step-by-step reasoning correctness
- Final answer format
- Units and precision
- Common arithmetic errors"""


CHECKER_QA_EMPHASIS = """
For QA validation, focus on:
- Factual accuracy
- Completeness of answer
- Relevance to the question
- Unsupported claims
- Potential misinformation"""


class Prerak(Role):
    """
    Prerak (Challenger) validates Implementor output using ensemble of verification methods.
    
    - Slot: SMALL (2-4B) + external tools
    - Purpose: Identify errors, flag issues, suggest fixes
    - Output: Verdict with flags, evidence, must_fix decision
    """
    
    name = "prerak"
    default_slot = Slot.SMALL
    
    def __init__(
        self, 
        model_runner: Any,
        tools: Optional[list[str]] = None,
        use_ensemble: bool = False,
        enable_retrieval: bool = True,
        enable_llm_check: bool = True,
        **kwargs
    ):
        super().__init__(
            model_runner=model_runner,
            slot=kwargs.get("slot", Slot.SMALL),
            max_tokens=kwargs.get("max_tokens", 768),
            temperature=kwargs.get("temperature", 0.2)
        )
        self.tools = tools or ["json_validator", "syntax_checker"]
        self._tool_results: dict[str, Any] = {}
        
        # Ensemble configuration (opt-in)
        self.use_ensemble = use_ensemble
        self.enable_retrieval = enable_retrieval
        self.enable_llm_check = enable_llm_check
        self._ensemble_results: Optional[dict[str, Any]] = None
    
    @property
    def system_prompt(self) -> str:
        return CHECKER_SYSTEM_PROMPT
    
    def format_input(self, role_input: RoleInput) -> str:
        task_spec_str = self._format_task_spec(role_input.task_spec)
        plan_str = self._format_plan(role_input.plan)
        candidate_str = self._format_candidate(role_input.candidate)
        tool_results_str = self._format_tool_results()
        
        # Add task-specific emphasis
        task_type = ""
        if role_input.task_spec:
            task_type = role_input.task_spec.get("task_type", "")
        
        prompt = CHECKER_USER_TEMPLATE.format(
            task_spec=task_spec_str,
            plan=plan_str,
            candidate=candidate_str,
            tool_results=tool_results_str
        )
        
        if task_type == "code":
            prompt += CHECKER_CODE_EMPHASIS
        elif task_type == "math":
            prompt += CHECKER_MATH_EMPHASIS
        elif task_type == "qa":
            prompt += CHECKER_QA_EMPHASIS
        
        return prompt
    
    def _format_task_spec(self, task_spec: Optional[dict]) -> str:
        """Format task spec for inclusion in prompt."""
        if not task_spec:
            return "No task specification provided."
        
        return f"""Problem: {task_spec.get('problem', 'Not specified')}
Task Type: {task_spec.get('task_type', 'Unknown')}
Output Format: {task_spec.get('output_format', 'text')}"""
    
    def _format_plan(self, plan: Optional[dict]) -> str:
        """Format plan summary for checker."""
        if not plan:
            return "No plan provided."
        
        steps = plan.get("steps", [])
        if not steps:
            return "No steps in plan."
        
        lines = [f"Expected Output: {plan.get('expected_output_type', 'text')}"]
        lines.append(f"Steps: {len(steps)}")
        
        checkpoints = plan.get("checkpoints", [])
        if checkpoints:
            lines.append(f"Checkpoints: {checkpoints}")
        
        return "\n".join(lines)
    
    def _format_candidate(self, candidate: Optional[dict]) -> str:
        """Format candidate output for checking."""
        if not candidate:
            return "No candidate output provided."
        
        content = candidate.get("content", "")
        content_type = candidate.get("content_type", "text")
        confidence = candidate.get("confidence", 0.5)
        warnings = candidate.get("warnings", [])
        
        lines = [
            f"Content Type: {content_type}",
            f"Implementor Confidence: {confidence}",
            "",
            "=== CONTENT START ===",
            content[:3000] if len(content) > 3000 else content,  # Truncate if too long
            "=== CONTENT END ==="
        ]
        
        if warnings:
            lines.append(f"\nImplementor Warnings: {warnings}")
        
        return "\n".join(lines)
    
    def _format_tool_results(self) -> str:
        """Format results from deterministic tools."""
        if not self._tool_results:
            return ""
        
        lines = ["\n--- TOOL RESULTS ---"]
        
        for tool_name, result in self._tool_results.items():
            lines.append(f"\n[{tool_name}]:")
            if isinstance(result, dict):
                if result.get("success"):
                    lines.append(f"  Status: PASS")
                else:
                    lines.append(f"  Status: FAIL")
                    if result.get("errors"):
                        for error in result["errors"][:3]:
                            lines.append(f"  - {error}")
            else:
                lines.append(f"  {result}")
        
        lines.append("--- END TOOL RESULTS ---")
        return "\n".join(lines)
    
    def run_ensemble_checks(self, content: str, check_type: str, context: Optional[dict] = None) -> dict[str, Any]:
        """Run ensemble checks (placeholder for actual implementation)."""
        # In a real implementation this would call out to deterministic tools
        return {"must_fix": False, "flags": [], "confidence": 0.5}

    def __call__(self, role_input: RoleInput) -> RoleOutput:
        """Execute Checker role."""
        return super().__call__(role_input)
            
    def set_retrieval_results(self, results: list[dict]) -> None:
        """Set retrieval results from external retrieval system."""
        self._tool_results["retrieval"] = {
            "success": True,
            "results": results
        }
    
    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse LLM output into Verdict dict."""
        data = self._extract_json(raw_output)
        
        # Normalize flags
        flags = []
        for flag in data.get("flags", []):
            if isinstance(flag, dict):
                flags.append({
                    "type": flag.get("type", "unknown"),
                    "severity": self._normalize_severity(flag.get("severity", "low")),
                    "detail": flag.get("detail", ""),
                    "location": flag.get("location"),
                    "suggested_fix": flag.get("suggested_fix")
                })
        
        # Normalize evidence
        evidence = []
        for ev in data.get("evidence", []):
            if isinstance(ev, dict):
                evidence.append({
                    "source": ev.get("source", ""),
                    "source_type": ev.get("source_type", "llm_judgment"),
                    "snippet": ev.get("snippet", ""),
                    "relevance_score": float(ev.get("relevance_score", 0)),
                    "supports_claim": ev.get("supports_claim", True)
                })
        
        # Determine must_fix
        must_fix = data.get("must_fix", False)
        if not must_fix:
            must_fix = any(
                f.get("severity") in ["high", "critical"] 
                for f in flags
            )
        
        # Normalize confidence
        confidence = data.get("overall_confidence", 0.5)
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "flags": flags,
            "must_fix": must_fix,
            "evidence": evidence,
            "suggested_edits": data.get("suggested_edits", []),
            "overall_confidence": confidence,
            "checks_performed": data.get("checks_performed", [])
        }
    
    def _normalize_severity(self, value: str) -> str:
        """Normalize severity to valid enum value."""
        valid = {"low", "medium", "high", "critical"}
        normalized = value.lower().strip()
        return normalized if normalized in valid else "low"
    
    def create_verdict(self, role_input: RoleInput) -> Verdict:
        """Execute checker and return a Verdict object."""
        output = self(role_input)
        
        if output.status == "error":
            return Verdict(
                flags=[CheckerFlag(
                    type="checker_error",
                    severity="low",
                    detail=f"Checker failed: {output.error}"
                )],
                must_fix=False,
                overall_confidence=0.5
            )
        
        return Verdict.from_dict(output.core_output)
