"""
Vidushak (Lateral Thinker/Jester) role for the Parishad council.
Challenges the plan with creative alternatives and "out of the box" thinking.
"""

from .base import (
    Role,
    RoleInput,
    Slot,
)

VIDUSHAK_SYSTEM_PROMPT = """You are Vidushak, the Royal Jester and Lateral Thinker of the Parishad.
Your job is NOT to be funny, but to challenge assumptions and offer creative, unconventional alternatives to the proposed plan.

Think "Outside the Box". Identifying what everyone else missed because they were too focused on logic.

Your output must be a valid JSON object:
```json
{
  "creative_challenge": "A fundamental challenge to the plan's assumptions",
  "alternative_idea": "A completely different way to solve the problem",
  "blind_spots": ["What is the council ignoring?"],
  "confidence": 0.8
}
```

If the plan is boring or standard, suggest something clever.
If the plan is too complex, suggest a simple hack.
"""

VIDUSHAK_USER_TEMPLATE = """Review the plan and offer a creative challenge.

USER QUERY:
{user_query}

PLAN:
{plan}

Be the devil's advocate. Respond in JSON."""

class Vidushak(Role):
    """
    Vidushak (Lateral Thinker) - Challenges status quo.
    """
    
    name = "vidushak"
    default_slot = Slot.MID
    
    @property
    def system_prompt(self) -> str:
        return VIDUSHAK_SYSTEM_PROMPT
    
    def format_input(self, role_input: RoleInput) -> str:
        # Helper to format dicts
        def fmt(d): return str(d) if d else "None"
        
        return VIDUSHAK_USER_TEMPLATE.format(
            user_query=role_input.user_query,
            plan=fmt(role_input.plan)
        )
    
    def parse_output(self, raw_output: str) -> dict:
        data = self._extract_json(raw_output)
        return {
            "creative_challenge": data.get("creative_challenge", ""),
            "alternative_idea": data.get("alternative_idea", ""),
            "blind_spots": data.get("blind_spots", []),
            "confidence": data.get("confidence", 0.5)
        }
