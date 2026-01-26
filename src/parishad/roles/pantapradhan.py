"""
Pantapradhan (Manager/PlannerHigh) role for the Parishad council.
Creates high-level strategic plans and identifies phases.
"""

from typing import Any, Optional

from .base import (
    Role,
    RoleInput,
    Slot,
    RoleOutput,
)


PLANNER_HIGH_SYSTEM_PROMPT = """You are Pantapradhan, the Manager in the Parishad council. Your job is to create strategic plans and identify the major components of a task.

Your responsibilities:
1. Understand the overall goal and scope
2. Identify major sub-tasks or phases
3. Determine the strategic approach
4. Identify key decision points and risks
5. Estimate overall complexity

You must ALWAYS respond with a valid JSON object in the following format:
```json
{
  "goal": "Clear statement of what needs to be achieved",
  "approach": "High-level strategy description",
  "phases": [
    {
      "id": 1,
      "name": "Phase name",
      "description": "What this phase accomplishes",
      "success_criteria": "How to know this phase is complete"
    }
  ],
  "key_decisions": ["Critical choices that affect the solution"],
  "risks": ["Potential issues or challenges"],
  "complexity": "trivial|simple|moderate|complex|very_complex",
  "task_category": "code|math|qa|explanation|creative|analysis"
}
```

Focus on the big picture. Don't worry about implementation details."""


PLANNER_HIGH_USER_TEMPLATE = """Create a high-level strategic plan for the following task.

ORIGINAL QUERY:
{user_query}

TASK SPECIFICATION:
{task_spec}

Provide a strategic overview and decomposition. Respond with ONLY a valid JSON object."""


class Pantapradhan(Role):
    """
    Pantapradhan (Manager) creates high-level strategic plans.
    
    - Slot: BIG (13-34B)
    - Purpose: Strategic decomposition and approach selection
    - Output: High-level plan with phases, risks, decisions
    """
    
    name = "pantapradhan"
    default_slot = Slot.BIG
    
    def __init__(self, model_runner: Any, **kwargs):
        super().__init__(
            model_runner=model_runner,
            slot=kwargs.get("slot", Slot.BIG),
            max_tokens=kwargs.get("max_tokens", 768),
            temperature=kwargs.get("temperature", 0.5)
        )
    
    @property
    def system_prompt(self) -> str:
        return PLANNER_HIGH_SYSTEM_PROMPT
    
    def format_input(self, role_input: RoleInput) -> str:
        task_spec_str = self._format_task_spec(role_input.task_spec)
        
        return PLANNER_HIGH_USER_TEMPLATE.format(
            user_query=role_input.user_query,
            task_spec=task_spec_str
        )
    
    def _format_task_spec(self, task_spec: Optional[dict]) -> str:
        """Format task spec for inclusion in prompt."""
        if not task_spec:
            return "No task specification provided."
        
        lines = [
            f"Problem: {task_spec.get('problem', 'Not specified')}",
            f"Constraints: {', '.join(task_spec.get('constraints', []))}",
            f"Output Format: {task_spec.get('output_format', 'text')}",
            f"Difficulty: {task_spec.get('difficulty_guess', 'medium')}",
            f"Task Type: {task_spec.get('task_type', 'unknown')}",
        ]
        return "\n".join(lines)
    
    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse LLM output into high-level plan dict."""
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
        
        # Normalize phases
        phases = []
        for phase in data.get("phases", []):
            phases.append({
                "id": phase.get("id", len(phases) + 1),
                "name": phase.get("name", "Unnamed phase"),
                "description": phase.get("description", ""),
                "success_criteria": phase.get("success_criteria", "")
            })
        
        return {
            "plan_high": {
                "goal": data.get("goal", ""),
                "approach": data.get("approach", ""),
                "phases": phases,
                "key_decisions": data.get("key_decisions", []),
                "risks": data.get("risks", []),
                "complexity": self._normalize_complexity(data.get("complexity", "moderate")),
                "task_category": data.get("task_category", "unknown")
            },
            # Compatible return
            "goal": data.get("goal", ""),
            "approach": data.get("approach", ""),
            "phases": phases
        }
    
    def _normalize_complexity(self, value: str) -> str:
        """Normalize complexity to valid enum value."""
        valid = {"trivial", "simple", "moderate", "complex", "very_complex"}
        normalized = value.lower().strip().replace(" ", "_")
        return normalized if normalized in valid else "moderate"
