"""
Darbari (Communicator/Refiner) role for the Parishad council.
Normalizes user input into structured task specifications.
"""

from typing import Any, Optional

from .base import (
    Role,
    RoleInput,
    RoleOutput,
    Slot,
    TaskSpec,
    Difficulty,
    TaskType,
    OutputFormat,
)


REFINER_SYSTEM_PROMPT = """You are Darbari, the Communicator in the Parishad council. Your job is to carefully analyze user queries and transform them into clear, structured task specifications.

Your responsibilities:
1. Understand the user's true intent and goal
2. Identify constraints (explicit or implicit)
3. Determine the expected output format
4. Estimate task difficulty for routing
5. Classify the task type
6. Assess safety sensitivity and expected answer length

You must ALWAYS respond with a valid JSON object in the following format:
```json
{
  "problem": "A clear, normalized restatement of what the user wants",
  "constraints": ["List of constraints or requirements"],
  "output_format": "code|text|numeric|structured|mixed",
  "difficulty_guess": "easy|medium|hard",
  "task_type": "math|code|qa|explanation|creative|analysis|chat",
  "key_concepts": ["Key concepts or topics involved"],
  "safety_sensitivity": "low|medium|high",
  "expected_answer_length": "short|paragraph|long"
}
```

Guidelines for difficulty estimation:
- EASY: Simple, single-step tasks; short answers; basic operations
- MEDIUM: Multi-step but straightforward; moderate reasoning needed
- HARD: Complex reasoning; multiple interconnected parts; edge cases; ambiguity

Guidelines for task type:
- CODE: Writing, debugging, or explaining code
- MATH: Numerical computation, word problems, proofs
- QA: Factual questions requiring knowledge retrieval
- EXPLANATION: Explaining concepts, processes, or ideas
- CREATIVE: Open-ended generation, writing, brainstorming
- ANALYSIS: Analyzing data, text, or situations
- CHAT: Conversational, no specific goal

Guidelines for safety sensitivity:
- LOW: Neutral topics, no harmful content
- MEDIUM: Potentially controversial or sensitive topics
- HIGH: Medical/legal advice, harmful content, misinformation risk

Guidelines for expected answer length:
- SHORT: 1-3 sentences, concise answer
- PARAGRAPH: 1-2 paragraphs, moderate detail
- LONG: Multi-paragraph, comprehensive explanation

Be precise and concise. Focus on extracting the essential requirements."""


REFINER_USER_TEMPLATE = """Analyze the following user query and create a structured task specification.

USER QUERY:
{user_query}

Respond with ONLY a valid JSON object following the required schema."""


class Darbari(Role):
    """
    Darbari (Communicator) normalizes user input into structured task specifications.
    
    - Slot: SMALL (2-4B)
    - Purpose: First role in pipeline, sets up context for all subsequent roles
    - Output: TaskSpec with problem, constraints, format, difficulty, type
    """
    
    name = "darbari"
    default_slot = Slot.SMALL
    
    def __init__(self, model_runner: Any, **kwargs):
        super().__init__(
            model_runner=model_runner,
            slot=kwargs.get("slot", Slot.SMALL),
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.3)
        )
    
    @property
    def system_prompt(self) -> str:
        return REFINER_SYSTEM_PROMPT
    
    def format_input(self, role_input: RoleInput) -> str:
        return REFINER_USER_TEMPLATE.format(
            user_query=role_input.user_query
        )
    
    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse LLM output into TaskSpec dict with routing metadata and robust fallback."""
        raw = raw_output.strip()
        
        # Handle empty output
        if not raw:
            logger.warning("Darbari received empty output from model")
            return {
                "problem": "",
                "constraints": [],
                "output_format": "text",
                "difficulty_guess": "medium",
                "task_type": "qa",
                "key_concepts": [],
                "difficulty": "medium",
                "safety_sensitivity": "low",
                "expected_answer_length": "paragraph",
                "parse_status": "empty"
            }
        
        try:
            data = self._extract_json(raw)
            
            # Check if JSON parsing succeeded or fell back to raw_output
            has_structure = "problem" in data or "task_type" in data
            if not has_structure and "raw_output" in data:
                # Fallback: extract problem from raw text
                problem_text = data["raw_output"][:500]
                parse_status = "fallback_text"
            else:
                problem_text = data.get("problem", "")
                parse_status = "json_ok"
            
            # Validate and normalize fields
            output = {
                "problem": problem_text,
                "constraints": data.get("constraints", []),
                "output_format": self._normalize_output_format(data.get("output_format", "text")),
                "difficulty_guess": self._normalize_difficulty(data.get("difficulty_guess", "medium")),
                "task_type": self._normalize_task_type(data.get("task_type", "qa")),
                "key_concepts": data.get("key_concepts", []),
                # Routing metadata for adaptive pipeline
                "difficulty": self._normalize_difficulty(data.get("difficulty_guess", "medium")),
                "safety_sensitivity": self._normalize_safety(data.get("safety_sensitivity", "low")),
                "expected_answer_length": self._normalize_length(data.get("expected_answer_length", "paragraph")),
                "parse_status": parse_status,
                "raw_output": raw
            }
            
            # If problem is still empty, use raw output
            if not output["problem"]:
                output["problem"] = raw[:500]
            
            return output
            
        except Exception as e:
            logger.exception("Darbari.parse_output: unexpected error during parsing")
            # Return minimal valid structure
            return {
                "problem": raw[:500] if raw else "",
                "constraints": [],
                "output_format": "text",
                "difficulty_guess": "medium",
                "task_type": "qa",
                "key_concepts": [],
                "difficulty": "medium",
                "safety_sensitivity": "low",
                "expected_answer_length": "paragraph",
                "parse_status": "error_fallback",
                "raw_output": raw
            }
    
    def _normalize_output_format(self, value: str) -> str:
        """Normalize output format to valid enum value."""
        valid = {"code", "text", "numeric", "structured", "mixed"}
        normalized = value.lower().strip()
        return normalized if normalized in valid else "text"
    
    def _normalize_difficulty(self, value: str) -> str:
        """Normalize difficulty to valid enum value."""
        valid = {"easy", "medium", "hard"}
        normalized = value.lower().strip()
        return normalized if normalized in valid else "medium"
    
    def _normalize_task_type(self, value: str) -> str:
        """Normalize task type to valid enum value."""
        valid = {"math", "code", "qa", "explanation", "creative", "analysis", "chat"}
        normalized = value.lower().strip()
        
        # Handle common aliases
        aliases = {
            "coding": "code",
            "programming": "code",
            "mathematics": "math",
            "calculation": "math",
            "question": "qa",
            "factual": "qa",
            "explain": "explanation",
            "describe": "explanation",
            "write": "creative",
            "generate": "creative",
            "analyze": "analysis",
            "evaluate": "analysis",
            "conversation": "chat",
            "conversational": "chat"
        }
        
        if normalized in valid:
            return normalized
        return aliases.get(normalized, "qa")
    
    def _normalize_safety(self, value: str) -> str:
        """Normalize safety sensitivity to valid value."""
        valid = {"low", "medium", "high"}
        normalized = value.lower().strip()
        return normalized if normalized in valid else "low"
    
    def _normalize_length(self, value: str) -> str:
        """Normalize expected answer length to valid value."""
        valid = {"short", "paragraph", "long"}
        normalized = value.lower().strip()
        return normalized if normalized in valid else "paragraph"
    
    def create_task_spec(self, role_input: RoleInput) -> TaskSpec:
        """
        Execute Refiner and return a TaskSpec object.
        """
        output = self(role_input)
        
        if output.status == "error":
            # Return default task spec on error
            return TaskSpec(
                problem=role_input.user_query,
                difficulty_guess=Difficulty.MEDIUM,
                task_type=TaskType.QA,
                output_format=OutputFormat.TEXT
            )
        
        return TaskSpec.from_dict(output.core_output)
