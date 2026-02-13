"""
Raja (Judge) role for the Parishad council.
Decider who synthesizes all information to produce the final answer.
"""

from typing import Any, Optional
import logging
import json

from .base import (
    Role,
    RoleInput,
    Slot,
    FinalAnswer,
)
from ..utils.text import truncate_with_note

logger = logging.getLogger(__name__)

# Full orchestration prompt - works for models across all sizes with temperature adjustment
JUDGE_SYSTEM_PROMPT = """You are Raja, the Judge in the Parishad council. Your job is to synthesize all information from the council and produce the final, authoritative answer.

You have access to:
1. The original user query
2. The Task Specification (from Darbari)
3. The Execution Plan (from Majumdar/Sar-Senapati)
4. The Implementor's solution (from Sainik)
5. The Challenger's verification verdict (from Prerak)

Your responsibilities:
1. Review all outputs from the council
2. Consider the Challenger's flags and evidence
3. Make the final decision on the answer
4. Ensure the answer is complete and accurate
5. Note any caveats or limitations

You must ALWAYS respond with a valid JSON object in the following format:
```json
{
  "final_answer": "The complete, polished answer to present to the user",
  "answer_type": "code|text|numeric|structured",
  "rationale": "Why this answer is correct and how you arrived at it",
  "confidence": 0.9,
  "caveats": ["Any limitations or assumptions"],
  "sources_used": ["What information you relied on"],
  "numeric_answer": 42,
  "code_block": "def solution(): ..."
}
```

Guidelines:

For CODE answers:
- Include the complete, runnable code in "code_block"
- Set "answer_type" to "code"
- Include any necessary explanations in "final_answer"
- If Challenger found issues, fix them in your final code

For MATH answers:
- Include the numeric result in "numeric_answer"
- Show key steps in "final_answer"
- Set "answer_type" to "numeric"

For QA/TEXT answers:
- Provide a clear, complete answer in "final_answer"
- Set "answer_type" to "text"
- Address the question directly

When Challenger found issues (must_fix = true):
- Carefully consider each flag
- Fix issues if possible
- If you cannot fix, explain why in caveats
- Adjust confidence accordingly

Be authoritative but honest. If something is uncertain, say so."""


JUDGE_USER_TEMPLATE = """Synthesize the council's outputs and produce the final answer.

ORIGINAL QUERY:
{user_query}

TASK SPECIFICATION:
{task_spec}

EXECUTION PLAN:
{plan}

IMPLEMENTOR OUTPUT:
{candidate}

CHALLENGER VERDICT:
{verdict}

Based on all the above, provide the final, authoritative answer. Respond with ONLY a valid JSON object."""


class Raja(Role):
    """
    Raja (Judge) integrates all council outputs into final answer.
    
    Works across all model sizes through intelligent orchestration:
    - Small models (0.5-3B): Receives simplified inputs from reduced pipeline
    - Large models (3B+): Receives full council synthesis
    
    Temperature is automatically adjusted based on model size.
    
    - Slot: BIG (13-34B)
    - Purpose: Final synthesis and decision making
    - Output: FinalAnswer with polished answer, rationale, confidence
    """
    
    name = "raja"
    default_slot = Slot.BIG
    
    def __init__(
        self, 
        model_runner: Any,
        fallback_slot: Optional[Slot] = Slot.MID,
        **kwargs
    ):
        super().__init__(
            model_runner=model_runner,
            slot=kwargs.get("slot", Slot.BIG),
            max_tokens=kwargs.get("max_tokens", 1536),
            temperature=kwargs.get("temperature", 0.4)
        )
        self.fallback_slot = fallback_slot
        # Phase-3 Task 2: Track truncation for metadata
        self._worker_truncated = False
        self._checker_truncated = False
    
    @property
    def system_prompt(self) -> str:
        return JUDGE_SYSTEM_PROMPT
    
    def format_input(self, role_input: RoleInput) -> str:
        # Orchestration mode: Use council outputs
        # For small models with reduced pipeline, some inputs will be None
        # Phase-3 Task 2: Extract truncation policy from routing metadata
        routing_meta = role_input.metadata.get("routing", {})
        truncation_policy = routing_meta.get("truncation_policy", "none")
        
        # Reset truncation tracking
        self._worker_truncated = False
        self._checker_truncated = False
        
        task_spec_str = self._format_task_spec(role_input.task_spec)
        plan_str = self._format_plan(role_input.plan)
        candidate_str = self._format_candidate(role_input.candidate, truncation_policy)
        verdict_str = self._format_verdict(role_input.verdict, truncation_policy)
        
        return JUDGE_USER_TEMPLATE.format(
            user_query=role_input.user_query,
            task_spec=task_spec_str,
            plan=plan_str,
            candidate=candidate_str,
            verdict=verdict_str
        )
    
    def __call__(self, role_input: RoleInput):
        """Override to add truncation metadata to output."""
        from .base import RoleOutput, RoleMetadata
        
        # Call base implementation
        output = super().__call__(role_input)
        
        # Phase-3 Task 2: Add truncation metadata if truncation occurred
        if self._worker_truncated or self._checker_truncated:
            # Create new RoleMetadata with truncation info
            new_metadata = RoleMetadata(
                tokens_used=output.metadata.tokens_used,
                latency_ms=output.metadata.latency_ms,
                model_id=output.metadata.model_id,
                slot=output.metadata.slot,
                timestamp=output.metadata.timestamp,
                duration_ms=output.metadata.duration_ms,
                schema_warning=output.metadata.schema_warning,
                worker_truncated=self._worker_truncated,
                checker_truncated=self._checker_truncated,
            )
            
            # Create new RoleOutput with updated metadata
            output = RoleOutput(
                role=output.role,
                status=output.status,
                core_output=output.core_output,
                error=output.error,
                metadata=new_metadata,
            )
        
        return output
    
    def _format_task_spec(self, task_spec: Optional[dict]) -> str:
        """Format task spec for judge review."""
        if not task_spec:
            return "No task specification provided."
        
        return f"""Problem: {task_spec.get('problem', 'Not specified')}
Task Type: {task_spec.get('task_type', 'Unknown')}
Output Format: {task_spec.get('output_format', 'text')}
Difficulty: {task_spec.get('difficulty_guess', 'medium')}"""
    
    def _format_plan(self, plan: Optional[dict]) -> str:
        """Format plan summary for judge."""
        if not plan:
            return "No plan provided."
        
        lines = []
        
        if plan.get("suggested_approach"):
            lines.append(f"Approach: {plan['suggested_approach']}")
        
        steps = plan.get("steps", [])
        lines.append(f"Steps planned: {len(steps)}")
        
        expected = plan.get("expected_output_type", "")
        if expected:
            lines.append(f"Expected output: {expected}")
        
        return "\n".join(lines)
    
    def _format_candidate(self, candidate: Optional[dict], truncation_policy: str = "none") -> str:
        """Format worker candidate for judge review.
        
        Args:
            candidate: Worker output dict
            truncation_policy: "none", "moderate", or "aggressive"
        """
        if not candidate:
            return "No candidate output from Implementor."
        
        content = candidate.get("content", "")
        content_type = candidate.get("content_type", "text")
        confidence = candidate.get("confidence", 0.5)
        warnings = candidate.get("warnings", [])
        
        # Phase-3 Task 2: Apply truncation based on policy
        limits = {
            "none": None,
            "moderate": 2500,
            "aggressive": 1200,
        }
        max_chars = limits.get(truncation_policy)
        
        was_truncated = False
        if max_chars and len(content) > max_chars:
            content, was_truncated = truncate_with_note(content, max_chars, "worker")
            self._worker_truncated = True  # Track for metadata
        
        lines = [
            f"Content Type: {content_type}",
            f"Implementor Confidence: {confidence}",
        ]
        
        if warnings:
            lines.append(f"Implementor Warnings: {warnings}")
        
        if was_truncated:
            lines.append(f"[Note: Worker output truncated from {len(candidate.get('content', ''))} to {max_chars} chars]")
        
        lines.extend([
            "",
            "=== IMPLEMENTOR OUTPUT ===",
            content[:4000] if len(content) > 4000 else content,
            "=== END OUTPUT ==="
        ])
        
        return "\n".join(lines)
    
    def _format_verdict(self, verdict: Optional[dict], truncation_policy: str = "none") -> str:
        """Format checker verdict for judge consideration.
        
        Args:
            verdict: Checker verdict dict
            truncation_policy: "none", "moderate", or "aggressive"
        """
        if not verdict:
            return "No challenger verdict available."
        
        lines = []
        
        must_fix = verdict.get("must_fix", False)
        confidence = verdict.get("overall_confidence", 0.5)
        
        lines.append(f"Must Fix: {'YES' if must_fix else 'No'}")
        lines.append(f"Challenger Confidence: {confidence}")
        
        flags = verdict.get("flags", [])
        if flags:
            # Phase-3 Task 2: Truncate number of flags shown based on policy
            flag_limit = {
                "none": len(flags),
                "moderate": min(5, len(flags)),
                "aggressive": min(3, len(flags)),
            }.get(truncation_policy, 5)
            
            if flag_limit < len(flags):
                self._checker_truncated = True  # Track for metadata
            
            lines.append(f"\nFlags ({len(flags)} total, showing {min(flag_limit, len(flags))}):")
            for flag in flags[:flag_limit]:
                severity = flag.get("severity", "unknown").upper()
                detail = flag.get("detail", "")
                fix = flag.get("suggested_fix", "")
                lines.append(f"  [{severity}] {detail}")
                if fix and truncation_policy != "aggressive":  # Skip fix details in aggressive mode
                    lines.append(f"    Fix: {fix}")
        else:
            lines.append("\nNo flags raised - output appears valid.")
        
        edits = verdict.get("suggested_edits", [])
        if edits and truncation_policy != "aggressive":  # Skip edits in aggressive mode
            lines.append("\nSuggested Edits:")
            for edit in edits[:3]:
                lines.append(f"  - {edit}")
        
        evidence = verdict.get("evidence", [])
        if evidence:
            lines.append(f"\nEvidence items: {len(evidence)}")
        
        return "\n".join(lines)
    
    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse LLM output into FinalAnswer dict."""
        
        # Parse JSON as usual - works for all models with appropriate temperature
        data = self._extract_json(raw_output)
        
        # Check if extraction failed (new flag from enhanced _extract_json)
        extraction_failed = data.get("extraction_failed", False)
        if extraction_failed:
            logger.warning(f"Raja: JSON extraction failed, using raw_output fallback")
        
        # Handle raw output fallback - ensure we always get some answer
        final_answer = data.get("final_answer", "")
        if not final_answer and "raw_output" in data:
            final_answer = data["raw_output"]
        
        # Fix: Sometimes models return JSON-formatted text in final_answer
        # Detect and extract the actual text content
        if final_answer and isinstance(final_answer, str):
            final_answer = self._extract_text_from_json_answer(final_answer)
        
        # CRITICAL: Detect degenerate / repetitive output from weak models
        from ..utils.output_sanitizer import detect_degenerate_output, sanitize_output
        if final_answer and detect_degenerate_output(final_answer):
            logger.warning(f"Raja: degenerate final_answer detected, sanitizing")
            final_answer = sanitize_output(final_answer)
        
        # Strip useless meta-phrases that weak models output
        # If final_answer is just "The final answer is " or similar, it's useless
        if final_answer:
            stripped = final_answer.strip()
            useless_patterns = [
                "the final answer is",
                "final answer:",
                "answer:",
                "the answer is",
                "the final answer is:",
                "final answer",
                "answer",
            ]
            if stripped.lower().rstrip(': ') in useless_patterns or stripped.lower() in useless_patterns:
                logger.warning(f"Raja output only meta-phrase '{stripped}' - treating as empty")
                final_answer = ""
        
        # CRITICAL: Ensure we never return an empty final_answer
        # This is the last line of defense against "No answer generated"
        if not final_answer or not final_answer.strip():
            logger.error(f"Raja: final_answer is empty! raw_output preview: {raw_output[:300]}")
            # Use the raw output directly, cleaned up
            final_answer = self._clean_raw_output_for_answer(raw_output)
        
        return {
            "final_answer": final_answer,
            "answer_type": data.get("answer_type", "text"),
            "rationale": data.get("rationale", ""),
            "confidence": max(0.0, min(1.0, data.get("confidence", 0.5))),
            "caveats": data.get("caveats", []),
            "sources_used": data.get("sources_used", []),
            "numeric_answer": data.get("numeric_answer"),
            "code_block": data.get("code_block")
        }
    
    def _clean_raw_output_for_answer(self, text: str) -> str:
        """
        Clean raw output to extract usable answer text when all parsing fails.
        
        This is the last-resort fallback to prevent "No answer generated".
        """
        import re
        
        if not text or not text.strip():
            return "Unable to generate a response. Please try rephrasing your question."
        
        # Remove markdown code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Remove Llama-3 format markers
        text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', text)
        text = re.sub(r'<\|eot_id\|>', '', text)
        text = re.sub(r'<\|begin_of_text\|>', '', text)
        
        # Remove JSON-like prefixes
        text = re.sub(r'^\s*\{?\s*"?final_answer"?\s*:\s*"?', '', text)
        
        # Clean up and limit
        text = ' '.join(text.split())
        if len(text) > 1500:
            text = text[:1500] + "..."
        
        return text.strip() if text.strip() else "Unable to generate a response. Please try rephrasing your question."
    
    def _extract_text_from_json_answer(self, text: str) -> str:
        """
        Extract plain text from JSON-formatted answers.
        
        Sometimes models return answers like:
        - {"answer": "actual text"}
        - {"response": "actual text"}
        - {"final_answer": "actual text"}
        
        This method detects these patterns and extracts the actual text.
        """
        import re
        from ..utils.output_sanitizer import detect_degenerate_output, sanitize_output
        
        # Use centralized sanitizer for degenerate/repetitive patterns
        if detect_degenerate_output(text):
            text = sanitize_output(text)
            if not text:
                return ""
        else:
            # For non-degenerate text, still strip common meta-prefixes
            for _ in range(10):
                before = text.strip()
                
                # Strip surrounding quotes
                if len(before) >= 2:
                    if (before[0] == '"' and before[-1] == '"') or \
                       (before[0] == "'" and before[-1] == "'"):
                        before = before[1:-1]
                
                # Strip known prefixes
                before = re.sub(r'^The\s+final\s+answer\s+is:?\s*', '', before, flags=re.IGNORECASE)
                before = re.sub(r'^Final\s+answer:?\s*', '', before, flags=re.IGNORECASE)
                before = re.sub(r'^Answer:?\s*', '', before, flags=re.IGNORECASE)
                
                if before == text.strip():
                    break
                text = before
        
        # If we're left with just meta-text, return empty (fallback will trigger)
        if text.strip().lower().rstrip(': ') in [
            'the final answer is', 'final answer', 'answer',
            'the answer is', 'response', 'result', 'output'
        ]:
            return ""
        
        # Skip JSON parsing if text doesn't look like JSON
        text_stripped = text.strip()
        if not (text_stripped.startswith('{') or text_stripped.startswith('[')):
            return text
        
        try:
            # Try to parse as JSON
            parsed = json.loads(text_stripped)
            
            # If it's a dict, look for common answer keys
            if isinstance(parsed, dict):
                # Priority order of keys to check
                answer_keys = [
                    "final_answer", "answer", "response", "text", "content", 
                    "result", "output", "message", "reply"
                ]
                
                for key in answer_keys:
                    if key in parsed and isinstance(parsed[key], str):
                        return parsed[key]
                
                # If no standard key found, return the first string value
                for value in parsed.values():
                    if isinstance(value, str) and len(value) > 10:  # Ignore short metadata
                        return value
            
            # If it's a list or other type, return original
            return text
            
        except (json.JSONDecodeError, TypeError):
            # Not valid JSON, return as-is
            return text
    
    def create_final_answer(self, role_input: RoleInput) -> FinalAnswer:
        """
        Execute Raja and return a FinalAnswer object.
        """
        output = self(role_input)
        
        if output.status == "error":
            return FinalAnswer(
                final_answer="Desh sevak encountered an error and could not produce a result.",
                answer_type="text",
                confidence=0.0,
                rationale=f"Error: {output.error}"
            )
        
        return FinalAnswer.from_dict(output.core_output)
