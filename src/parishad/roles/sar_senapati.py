"""
SarSenapati (Executor/PlannerExec) role for the Parishad council.
Converts high-level strategic plans into concrete, executable steps.
"""

from typing import Any, Optional

from .base import (
    Role,
    RoleInput,
    RoleOutput,
    Slot,
)


PLANNER_EXEC_SYSTEM_PROMPT = """You are Sar-Senapati, the Executor in the Parishad council. Your job is to convert high-level strategic plans into concrete, executable steps.

Your responsibilities:
1. Take the high-level plan and make it actionable
2. Create detailed, step-by-step instructions
3. Specify exact operations for each step
4. Identify dependencies between steps
5. Add verification checkpoints

You must ALWAYS respond with a valid JSON object in the following format:
```json
{
  "steps": [
    {
      "id": 1,
      "description": "Concrete action to take",
      "rationale": "Why this step is needed",
      "expected_output": "What this step produces",
      "depends_on": [],
      "verification": "How to verify this step succeeded"
    }
  ],
  "checkpoints": [1, 3],
  "expected_output_type": "python_function|explanation|numeric_answer|structured_data",
  "worker_instructions": "Specific guidance for the Implementor"
}
```

Be specific and concrete. Every step should be directly actionable by the Implementor."""


PLANNER_EXEC_USER_TEMPLATE = """Convert the high-level plan into executable steps.

ORIGINAL QUERY:
{user_query}

TASK SPECIFICATION:
{task_spec}

HIGH-LEVEL PLAN:
{plan_high}

Create detailed, actionable steps. Respond with ONLY a valid JSON object."""


class SarSenapati(Role):
    """
    SarSenapati (Executor) converts high-level plans into executable steps.
    
    - Slot: MID (7-13B)
    - Purpose: Create detailed execution plan from strategy
    - Output: Concrete steps with dependencies and checkpoints
    """
    
    name = "sar_senapati"
    default_slot = Slot.MID
    
    def __init__(self, model_runner: Any, **kwargs):
        super().__init__(
            model_runner=model_runner,
            slot=kwargs.get("slot", Slot.MID),
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.4)
        )
    
    @property
    def system_prompt(self) -> str:
        return PLANNER_EXEC_SYSTEM_PROMPT
    
    def format_input(self, role_input: RoleInput) -> str:
        task_spec_str = self._format_task_spec(role_input.task_spec)
        plan_high_str = self._format_plan_high(role_input.context.get("plan_high"))
        
        return PLANNER_EXEC_USER_TEMPLATE.format(
            user_query=role_input.user_query,
            task_spec=task_spec_str,
            plan_high=plan_high_str
        )
    
    def _format_task_spec(self, task_spec: Optional[dict]) -> str:
        """Format task spec for inclusion in prompt."""
        if not task_spec:
            return "No task specification provided."
        
        lines = [
            f"Problem: {task_spec.get('problem', 'Not specified')}",
            f"Constraints: {', '.join(task_spec.get('constraints', []))}",
            f"Output Format: {task_spec.get('output_format', 'text')}",
        ]
        return "\n".join(lines)
    
    def _format_plan_high(self, plan_high: Optional[dict]) -> str:
        """Format high-level plan for inclusion in prompt."""
        if not plan_high:
            return "No high-level plan provided."
        
        lines = [
            f"Goal: {plan_high.get('goal', 'Not specified')}",
            f"Approach: {plan_high.get('approach', 'Not specified')}",
            f"Complexity: {plan_high.get('complexity', 'unknown')}",
        ]
        
        phases = plan_high.get("phases", [])
        if phases:
            lines.append("\nPhases:")
            for phase in phases:
                lines.append(f"  {phase.get('id', '?')}. {phase.get('name', 'Unnamed')}: {phase.get('description', '')}")
        
        return "\n".join(lines)
    
    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse LLM output into execution plan dict."""
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
        
        # Normalize steps
        steps = []
        for step in data.get("steps", []):
            steps.append({
                "id": step.get("id", len(steps) + 1),
                "description": step.get("description", ""),
                "rationale": step.get("rationale", ""),
                "expected_output": step.get("expected_output", ""),
                "depends_on": step.get("depends_on", []),
                "verification": step.get("verification", "")
            })
        
        return {
            "plan_exec": {
                "steps": steps,
                "checkpoints": data.get("checkpoints", []),
                "expected_output_type": data.get("expected_output_type", "text"),
                "worker_instructions": data.get("worker_instructions", "")
            },
            # Compatible return
            "steps": steps,
            "checkpoints": data.get("checkpoints", []),
            "expected_output_type": data.get("expected_output_type", "text")
        }
