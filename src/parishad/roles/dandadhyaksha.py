"""
Dandadhyaksha (Enforcer/Safety Checker) role for the Parishad council.
Checks for safety violations, PII, and policy compliance.
"""

from typing import Any, Optional

from .base import (
    Role,
    RoleInput,
    RoleOutput,
    Slot,
    Verdict,
    CheckerFlag,
)


CHECKER_SAFETY_SYSTEM_PROMPT = """You are Dandadhyaksha, the Enforcer in the Parishad council. Your job is to verify the output is safe, policy-compliant, and free of sensitive information.

Your responsibilities:
1. Check for harmful, offensive, or inappropriate content
2. Identify any personally identifiable information (PII)
3. Verify compliance with content policies
4. Flag potential safety or ethical concerns
5. Suggest redactions or modifications for safety issues

You must ALWAYS respond with a valid JSON object in the following format:
```json
{
  "flags": [
    {
      "type": "safety_violation|pii_detected|policy_violation|unsafe_code",
      "severity": "medium|high|critical",
      "detail": "Description of the safety issue",
      "location": "Where it appears",
      "suggested_fix": "Redaction or modification"
    }
  ],
  "must_fix": true,
  "safe": false,
  "summary": "Safety assessment summary"
}
```

If "safe" is false, "must_fix" MUST be true.
Be extremely vigilant about PII (names, emails, phones, keys) and harmful content."""


CHECKER_SAFETY_USER_TEMPLATE = """Perform a safety check on the following output.

OUTPUT TO CHECK:
{candidate}

Analyze for safety/PII violations. Respond with ONLY a valid JSON object."""


class Dandadhyaksha(Role):
    """
    Dandadhyaksha (Enforcer) checks for safety violations and PII.
    
    - Slot: SMALL (2-4B)
    - Purpose: Safety and policy enforcement
    - Output: Verdict on safety
    """
    
    name = "dandadhyaksha"
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
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.1)
        )
        self.tools = tools or ["toxicity_detector", "pii_scanner"]
    
    @property
    def system_prompt(self) -> str:
        return CHECKER_SAFETY_SYSTEM_PROMPT
    
    def format_input(self, role_input: RoleInput) -> str:
        candidate_str = self._format_candidate(role_input.candidate)
        
        return CHECKER_SAFETY_USER_TEMPLATE.format(
            candidate=candidate_str
        )
    
    def _format_candidate(self, candidate: Optional[dict]) -> str:
        """Format candidate output for checking."""
        if not candidate:
            return "No output to verify."
        
        content = candidate.get("content", "")
        # Don't truncate too much for safety check, but respect limit
        if len(content) > 4000:
            content = content[:4000] + "... [truncated]"
        
        return f"""Content:
{content}"""
    
    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse LLM output into safety verdict dict."""
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
        
        # Normalize flags
        flags = []
        for flag in data.get("flags", []):
            flags.append({
                "type": flag.get("type", "unknown"),
                "severity": self._normalize_severity(flag.get("severity", "medium")),
                "detail": flag.get("detail", ""),
                "location": flag.get("location"),
                "suggested_fix": flag.get("suggested_fix")
            })
        
        # Determine must_fix
        must_fix = data.get("must_fix", False)
        safe = data.get("safe", True)
        
        if not safe:
            must_fix = True
        
        # If flags exist, ensure must_fix matches severity
        if flags and any(f["severity"] in ["high", "critical"] for f in flags):
            must_fix = True
            safe = False
        
        return {
            "verdict_safety": {
                "flags": flags,
                "must_fix": must_fix,
                "safe": safe,
                "summary": data.get("summary", "")
            },
            # Compatible with standard Verdict schema
            "flags": flags,
            "must_fix": must_fix,
            "evidence": [],
            "overall_confidence": 1.0 if safe else 0.0
        }
    
    def _normalize_severity(self, value: str) -> str:
        """Normalize severity to valid enum value."""
        valid = {"low", "medium", "high", "critical"}
        normalized = value.lower().strip()
        return normalized if normalized in valid else "high"
