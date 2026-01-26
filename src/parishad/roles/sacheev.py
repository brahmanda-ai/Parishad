"""
Sacheev (Advisor/CheckerFact) role for the Parishad council.
Verifies factual claims using retrieval and reasoning.
"""

from typing import Any, Optional

from .base import (
    Role,
    RoleInput,
    RoleOutput,
    Slot,
)


CHECKER_FACT_SYSTEM_PROMPT = """You are Sacheev, the Advisor in the Parishad council. Your job is to verify the factual accuracy of the Implementor's output.

Your responsibilities:
1. Identify factual claims in the output
2. Verify claims against known facts and reasoning
3. Flag unsupported or incorrect claims
4. Assess overall factual reliability
5. Suggest corrections for factual errors

You must ALWAYS respond with a valid JSON object in the following format:
```json
{
  "claims": [
    {
      "claim": "The specific claim being verified",
      "status": "verified|unverified|incorrect|partially_correct",
      "confidence": 0.9,
      "evidence": "Supporting or contradicting evidence",
      "correction": "Corrected version if incorrect"
    }
  ],
  "overall_accuracy": 0.85,
  "factual_issues": [
    {
      "type": "incorrect_fact|unsupported_claim|outdated_info|logical_error",
      "severity": "low|medium|high|critical",
      "description": "Description of the issue",
      "suggestion": "How to fix it"
    }
  ],
  "must_fix": false,
  "summary": "Brief summary of factual assessment"
}
```

Be rigorous but fair. Only flag issues you're confident about.
Distinguish between factual errors and matters of opinion."""


CHECKER_FACT_USER_TEMPLATE = """Verify the factual accuracy of the following output.

TASK SPECIFICATION:
{task_spec}

EXECUTION PLAN:
{plan}

OUTPUT TO VERIFY:
{candidate}

Analyze for factual correctness. Respond with ONLY a valid JSON object."""


class Sacheev(Role):
    """
    Sacheev (Advisor) verifies factual accuracy of outputs.
    
    - Slot: SMALL (2-4B)
    - Purpose: Verify claims and flag factual errors
    - Output: Verdict on factual correctness
    """
    
    name = "sacheev"
    default_slot = Slot.SMALL
    
    def __init__(
        self, 
        model_runner: Any,
        tools: Optional[list[str]] = None,
        **kwargs
    ):
        super().__init__(
            model_runner=model_runner,
            slot=kwargs.get("slot", Slot.SMALL),
            max_tokens=kwargs.get("max_tokens", 768),
            temperature=kwargs.get("temperature", 0.2)
        )
        self.tools = tools or ["retrieval", "claim_extractor"]
    
    @property
    def system_prompt(self) -> str:
        return CHECKER_FACT_SYSTEM_PROMPT
    
    def format_input(self, role_input: RoleInput) -> str:
        task_spec_str = self._format_task_spec(role_input.task_spec)
        plan_str = self._format_plan(role_input.plan)
        candidate_str = self._format_candidate(role_input.candidate)
        
        return CHECKER_FACT_USER_TEMPLATE.format(
            task_spec=task_spec_str,
            plan=plan_str,
            candidate=candidate_str
        )
    
    def _format_task_spec(self, task_spec: Optional[dict]) -> str:
        """Format task spec for inclusion in prompt."""
        if not task_spec:
            return "No task specification provided."
        
        return f"""Problem: {task_spec.get('problem', 'Not specified')}
Task Type: {task_spec.get('task_type', 'Unknown')}"""
    
    def _format_plan(self, plan: Optional[dict]) -> str:
        """Format plan summary."""
        if not plan:
            return "No plan provided."
        
        steps = plan.get("steps", [])
        return f"Steps: {len(steps)}, Expected: {plan.get('expected_output_type', 'text')}"
    
    def _format_candidate(self, candidate: Optional[dict]) -> str:
        """Format candidate output for checking."""
        if not candidate:
            return "No output to verify."
        
        content = candidate.get("content", "")
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"
        
        return f"""Content Type: {candidate.get('content_type', 'unknown')}
Content:
{content}"""
    
    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse LLM output into factual verdict dict."""
        import json
        import re
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}
        
        # Normalize claims
        claims = []
        for claim in data.get("claims", []):
            claims.append({
                "claim": claim.get("claim", ""),
                "status": claim.get("status", "unverified"),
                "confidence": max(0.0, min(1.0, claim.get("confidence", 0.5))),
                "evidence": claim.get("evidence", ""),
                "correction": claim.get("correction", "")
            })
        
        # Normalize factual issues
        issues = []
        for issue in data.get("factual_issues", []):
            issues.append({
                "type": issue.get("type", "unknown"),
                "severity": self._normalize_severity(issue.get("severity", "low")),
                "description": issue.get("description", ""),
                "suggestion": issue.get("suggestion", "")
            })
        
        # Determine must_fix
        must_fix = data.get("must_fix", False)
        if not must_fix:
            must_fix = any(i.get("severity") in ["high", "critical"] for i in issues)
        
        return {
            "verdict_fact": {
                "claims": claims,
                "overall_accuracy": max(0.0, min(1.0, data.get("overall_accuracy", 0.5))),
                "factual_issues": issues,
                "must_fix": must_fix,
                "summary": data.get("summary", "")
            },
            # Compatible with standard Verdict schema
            "flags": [{
                "type": i["type"], 
                "severity": i["severity"], 
                "detail": i["description"], 
                "suggested_fix": i["suggestion"]
            } for i in issues],
            "must_fix": must_fix,
            "overall_confidence": max(0.0, min(1.0, data.get("overall_accuracy", 0.5)))
        }
    
    def _normalize_severity(self, value: str) -> str:
        """Normalize severity to valid enum value."""
        valid = {"low", "medium", "high", "critical"}
        normalized = value.lower().strip()
        return normalized if normalized in valid else "low"
