"""
Majumdar (Planner) role for the Parishad council.
Decomposes complex tasks into clear, executable steps.
"""

from typing import Any, Optional

from .base import (
    Role,
    RoleInput,
    RoleOutput,
    Slot,
    Plan,
    PlanStep,
)


PLANNER_SYSTEM_PROMPT = """You are Majumdar, the Planner in the Parishad council. Your job is to decompose complex tasks into clear, executable steps that the Implementor can follow.

Your responsibilities:
1. Analyze the task specification
2. Break down the problem into logical, sequential steps
3. Provide rationale for each step
4. Identify critical checkpoints where verification is needed
5. Estimate the complexity of execution

You must ALWAYS respond with a valid JSON object in the following format:
```json
{
  "steps": [
    {
      "id": 1,
      "description": "What to do in this step",
      "rationale": "Why this step is needed",
      "expected_output": "What this step should produce",
      "depends_on": []
    }
  ],
  "checkpoints": [1, 3],
  "expected_output_type": "python_function|explanation|numeric_answer|structured_data",
  "complexity_estimate": "trivial|simple|moderate|complex|very_complex",
  "suggested_approach": "High-level strategy for the Implementor"
}
```

Guidelines:
- Keep steps atomic and actionable
- Steps should be independent where possible
- Identify dependencies between steps accurately
- Mark steps that need verification as checkpoints

For code tasks:
- Include steps for understanding requirements, implementing, and testing
- Consider edge cases and error handling

For math tasks:
- Break down into clear mathematical operations
- Include verification steps for intermediate results

For QA tasks:
- Identify what information needs to be retrieved
- Include steps for synthesizing information into an answer"""


PLANNER_USER_TEMPLATE = """Create an execution plan for the following task.

ORIGINAL QUERY:
{user_query}

TASK SPECIFICATION:
{task_spec}

Create a detailed step-by-step plan. Respond with ONLY a valid JSON object."""


class Majumdar(Role):
    """
    Majumdar (Planner) decomposes tasks into executable steps.
    
    - Slot: BIG (13-34B), with MID fallback for easy tasks
    - Purpose: Create structured plan for Implementor to execute
    - Output: Plan with steps, checkpoints, complexity estimate
    """
    
    name = "majumdar"
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
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.5)
        )
        self.fallback_slot = fallback_slot
    
    @property
    def system_prompt(self) -> str:
        return PLANNER_SYSTEM_PROMPT
    
    def format_input(self, role_input: RoleInput) -> str:
        task_spec_str = self._format_task_spec(role_input.task_spec)
        
        return PLANNER_USER_TEMPLATE.format(
            user_query=role_input.user_query,
            task_spec=task_spec_str
        )
    
    def _format_task_spec(self, task_spec: Optional[dict]) -> str:
        """Format task spec for inclusion in prompt."""
        if not task_spec:
            return "No task specification provided."
        
        lines = [
            f"Problem: {task_spec.get('problem', 'Not specified')}",
            f"Task Type: {task_spec.get('task_type', 'Unknown')}",
            f"Output Format: {task_spec.get('output_format', 'text')}",
            f"Difficulty: {task_spec.get('difficulty_guess', 'medium')}",
        ]
        
        constraints = task_spec.get('constraints', [])
        if constraints:
            lines.append(f"Constraints: {', '.join(constraints)}")
        
        concepts = task_spec.get('key_concepts', [])
        if concepts:
            lines.append(f"Key Concepts: {', '.join(concepts)}")
        
        return "\n".join(lines)
    
    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse LLM output into Plan dict with robust fallback."""
        raw = raw_output.strip()
        
        # Handle empty output
        if not raw:
            logger.warning("Majumdar received empty output from model")
            return {
                "steps": [{
                    "id": 1,
                    "description": "Complete the task",
                    "rationale": "Default step",
                    "expected_output": "Result",
                    "depends_on": []
                }],
                "checkpoints": [],
                "expected_output_type": "text",
                "complexity_estimate": "moderate",
                "suggested_approach": "",
                "parse_status": "empty"
            }
        
        try:
            data = self._extract_json(raw)
            
            # Normalize steps
            steps = []
            raw_steps = data.get("steps", [])
            
            for i, step in enumerate(raw_steps):
                if isinstance(step, dict):
                    steps.append({
                        "id": step.get("id", i + 1),
                        "description": step.get("description", ""),
                        "rationale": step.get("rationale", ""),
                        "expected_output": step.get("expected_output", ""),
                        "depends_on": step.get("depends_on", [])
                    })
                elif isinstance(step, str):
                    steps.append({
                        "id": i + 1,
                        "description": step,
                        "rationale": "",
                        "expected_output": "",
                        "depends_on": []
                    })
            
            # Fallback: if no steps found, create default step
            if not steps:
                has_structure = "steps" in data or "checkpoints" in data
                if not has_structure and "raw_output" in data:
                    # Model didn't follow format - create simple plan from text
                    steps = [{
                        "id": 1,
                        "description": data["raw_output"][:200],
                        "rationale": "Extracted from model output",
                        "expected_output": "Completion",
                        "depends_on": []
                    }]
                    parse_status = "fallback_text"
                else:
                    steps = [{
                        "id": 1,
                        "description": "Complete the task as specified",
                        "rationale": "Single-step execution",
                        "expected_output": "Final result",
                        "depends_on": []
                    }]
                    parse_status = "default_plan"
            else:
                parse_status = "json_ok"
            
            return {
                "steps": steps,
                "checkpoints": data.get("checkpoints", []),
                "expected_output_type": data.get("expected_output_type", "text"),
                "complexity_estimate": self._normalize_complexity(
                    data.get("complexity_estimate", "moderate")
                ),
                "suggested_approach": data.get("suggested_approach", ""),
                "parse_status": parse_status,
                "raw_output": raw
            }
            
        except Exception as e:
            logger.exception("Majumdar.parse_output: unexpected error during parsing")
            # Return minimal valid plan
            return {
                "steps": [{
                    "id": 1,
                    "description": raw[:200] if raw else "Execute task",
                    "rationale": "Fallback plan",
                    "expected_output": "Result",
                    "depends_on": []
                }],
                "checkpoints": [],
                "expected_output_type": "text",
                "complexity_estimate": "moderate",
                "suggested_approach": "",
                "parse_status": "error_fallback",
                "raw_output": raw
            }
    
    def _normalize_complexity(self, value: str) -> str:
        """Normalize complexity to valid enum value."""
        valid = {"trivial", "simple", "moderate", "complex", "very_complex"}
        normalized = value.lower().strip().replace(" ", "_")
        return normalized if normalized in valid else "moderate"
    
    def should_use_fallback(self, task_spec: Optional[dict]) -> bool:
        """Determine if we should use fallback slot based on difficulty."""
        if not task_spec:
            return False
        difficulty = task_spec.get("difficulty_guess", "medium")
        return difficulty == "easy"
    
    def __call__(self, role_input: RoleInput) -> Any:
        """Execute planner with potential slot fallback."""
        if self.fallback_slot and self.should_use_fallback(role_input.task_spec):
            original_slot = self.slot
            self.slot = self.fallback_slot
            try:
                return super().__call__(role_input)
            finally:
                self.slot = original_slot
        
        return super().__call__(role_input)
    
    def create_plan(self, role_input: RoleInput) -> Plan:
        """Execute planner and return a Plan object."""
        output = self(role_input)
        
        if output.status == "error":
            return Plan(
                steps=[PlanStep(id=1, description="Complete the task")],
                expected_output_type="text"
            )
        
        return Plan.from_dict(output.core_output)
