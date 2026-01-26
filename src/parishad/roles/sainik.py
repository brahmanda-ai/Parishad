"""
Sainik (Implementor) role for the Parishad council.
Executes plans and generates solutions (Code/Text).
Combines functionality of Worker, WorkerCode, and WorkerText.
"""

from typing import Any, Optional

from .base import (
    Role,
    RoleInput,
    RoleOutput,
    Slot,
    Candidate,
)


WORKER_SYSTEM_PROMPT = """You are Sainik, the Implementor in the Parishad council. Your job is to execute the plan created by Majumdar/Sar-Senapati and generate high-quality solutions.

Your responsibilities:
1. Follow the plan step by step
2. Generate accurate, complete solutions
3. Track your reasoning process
4. Identify potential issues or uncertainties
5. Produce output in the expected format

You must ALWAYS respond with a valid JSON object in the following format:
```json
{
  "content": "Your complete solution/answer/code here",
  "content_type": "code|text|numeric|mixed",
  "language": "python",
  "reasoning_trace": [
    "Step 1: I analyzed...",
    "Step 2: I implemented..."
  ],
  "confidence": 0.85,
  "warnings": ["Potential issue: ..."],
  "imports": ["required_imports"],
  "key_points": ["key takeaway 1"],
  "target_file": "path/to/output_file.ext",
  "tool_calls": [
    {
      "tool": "tool_name",
      "action": "action_name", 
      "args": { "arg1": "value" }
    }
  ]
}
```

Guidelines:
- If writing code, put the COMPLETE runnable code in "content".
- If writing text, put the clear explanation in "content".
- If the user asked to change/create a file, you MUST specify "target_file".
- IMPORTANT: Do NOT create files unless explicitly asked! For general questions/explanations, keep "target_file": null.
- "target_file" should be relative to the current directory (e.g., "src/main.py").
- If "target_file" is a text/markdown/json file (not executable code), put the RAW content in "content". DO NOT write a Python script to create it.
- If you need to Use a tool, add it to `tool_calls`. Available tools will be listed in the prompt.
- Be honest about confidence."""


WORKER_USER_TEMPLATE = """Execute the following plan and generate a solution.

ORIGINAL QUERY:
{user_query}

TASK SPECIFICATION:
{task_spec}

EXECUTION PLAN:
{plan}

{retry_context}

Follow the plan and generate a complete solution. Respond with ONLY a valid JSON object."""


WORKER_CODE_EMPHASIS = """
IMPORTANT: This is a CODE task. Your "content" field must contain complete, runnable code.
- Include all necessary imports at the top
- Write a complete solution that can be executed
- Add brief comments for complex logic
- The code should be ready to run without modifications
- Set "content_type" to "code"
- Set "language" to the programming language used"""


WORKER_MATH_EMPHASIS = """
IMPORTANT: This is a MATH task. Show your work clearly.
- Include step-by-step calculations in reasoning_trace
- Double-check your arithmetic
- State your final answer clearly in the content field
- If the answer is numeric, ensure it is exact"""


WORKER_TEXT_EMPHASIS = """
IMPORTANT: This is a TEXT/EXPLANATION task.
- Write clear, well-structured explanations
- Organize content logically
- Be concise but complete
- Set "content_type" to "text"
- If editing a file, provide the NEW FULL CONTENT of the file, not a description of changes."""


class Sainik(Role):
    """
    Sainik (Implementor) executes the plan and generates candidate solutions.
    Handles both Text and Code generation based on task type.
    
    - Slot: MID (7-13B)
    - Purpose: Main content generation following Planner's steps
    - Output: Candidate with content, reasoning trace, confidence
    """
    
    name = "sainik"
    default_slot = Slot.MID
    
    
    def __init__(self, model_runner: Any, tools: Optional[list[Any]] = None, **kwargs):
        super().__init__(
            model_runner=model_runner,
            slot=kwargs.get("slot", Slot.MID),
            max_tokens=kwargs.get("max_tokens", 2048),
            temperature=kwargs.get("temperature", 0.6)
        )
        self.tools = tools or []
    
    @property
    def system_prompt(self) -> str:
        return WORKER_SYSTEM_PROMPT
    
    def format_input(self, role_input: RoleInput) -> str:
        task_spec_str = self._format_task_spec(role_input.task_spec)
        plan_str = self._format_plan(role_input.plan)
        retry_context = self._format_retry_context(role_input.context)
        
        # Phase 13: Tool Integration - Inject File Context & Tool Descriptions
        file_context = ""
        tool_descriptions = ""
        
        if self.tools:
            tool_descriptions = "\n\nAVAILABLE TOOLS:\n"
            for tool in self.tools:
                tool_descriptions += f"- {tool.name}: {tool.description or 'No description'}\n"
                
                if tool.name == "file_system":
                    try:
                        # List files in current directory to give context
                        result = tool.run("list", path=".")
                        if result.success:
                            file_context += f"\n\nCURRENT DIRECTORY CONTEXT:\n{result.data}\n"
                    except Exception as e:
                        file_context += f"\n\nError accessing file system: {str(e)}\n"
        

        # Add task-specific emphasis
        task_type = ""
        if role_input.task_spec:
            task_type = role_input.task_spec.get("task_type", "")
        
        prompt = WORKER_USER_TEMPLATE.format(
            user_query=role_input.user_query,
            task_spec=task_spec_str,
            plan=plan_str,
            retry_context=retry_context + file_context + tool_descriptions # Append file context and tool details
        )
        
        if task_type == "code":
            prompt += WORKER_CODE_EMPHASIS
        elif task_type == "math":
            prompt += WORKER_MATH_EMPHASIS
        else:
            prompt += WORKER_TEXT_EMPHASIS
        
        return prompt
    
    def _format_task_spec(self, task_spec: Optional[dict]) -> str:
        """Format task spec for inclusion in prompt."""
        if not task_spec:
            return "No task specification provided."
        
        lines = [
            f"Problem: {task_spec.get('problem', 'Not specified')}",
            f"Task Type: {task_spec.get('task_type', 'Unknown')}",
            f"Output Format: {task_spec.get('output_format', 'text')}",
        ]
        
        constraints = task_spec.get('constraints', [])
        if constraints:
            lines.append(f"Constraints: {', '.join(constraints)}")
        
        return "\n".join(lines)
    
    def _format_plan(self, plan: Optional[dict]) -> str:
        """Format plan for inclusion in prompt."""
        if not plan:
            return "No plan provided. Complete the task directly."
        
        lines = []
        
        if plan.get("suggested_approach"):
            lines.append(f"Approach: {plan['suggested_approach']}")
        
        steps = plan.get("steps", [])
        for step in steps:
            step_id = step.get("id", "?")
            desc = step.get("description", "")
            lines.append(f"Step {step_id}: {desc}")
            
            if step.get("rationale"):
                lines.append(f"  Rationale: {step['rationale']}")
        
        expected_output = plan.get("expected_output_type", "")
        if expected_output:
            lines.append(f"\nExpected Output Type: {expected_output}")
            
        instructions = plan.get("worker_instructions", "")
        if instructions:
            lines.append(f"\nInstructions: {instructions}")
        
        return "\n".join(lines)
    
    def _format_retry_context(self, context: dict) -> str:
        """Format retry context if this is a retry attempt."""
        if not context.get("is_retry"):
            return ""
        
        lines = ["\n--- RETRY CONTEXT ---"]
        
        previous_output = context.get("previous_output", "")
        if previous_output:
            lines.append(f"Your previous output:\n{previous_output[:500]}...")
        
        checker_feedback = context.get("checker_feedback", {})
        if checker_feedback:
            flags = checker_feedback.get("flags", [])
            if flags:
                lines.append("\nIssues identified by Challenger:")
                for flag in flags[:5]:  # Limit to 5 flags
                    lines.append(f"- [{flag.get('severity', 'unknown')}] {flag.get('detail', '')}")
            
            edits = checker_feedback.get("suggested_edits", [])
            if edits:
                lines.append("\nSuggested fixes:")
                for edit in edits[:3]:
                    lines.append(f"- {edit}")
        
        lines.append("\nPlease fix the issues and regenerate your solution.")
        lines.append("--- END RETRY CONTEXT ---\n")
        
        return "\n".join(lines)
    
    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """
        Parse LLM output into Candidate dict with robust fallback.
        
        Strategy:
        1. If empty output, return empty but valid dict
        2. Try to parse as JSON with expected schema
        3. Fall back to extracting text answer from raw output
        4. NEVER fail completely - always return valid dict
        """
        raw = raw_output.strip()
        
        # Handle empty output
        if not raw:
            logger.warning("Sainik received empty output from model")
            return {
                "content": "",
                "content_type": "text",
                "confidence": 0.0,
                "reasoning_trace": [],
                "warnings": ["Model returned empty output"],
                "parse_status": "empty"
            }
        
        # Try strict JSON parsing first
        try:
            data = self._extract_json(raw)
            
            # Check if we got valid structured output
            content = data.get("content", "")
            if not content and "raw_output" in data:
                # _extract_json returned fallback, try text extraction
                content = self._extract_text_answer(data["raw_output"])
                parse_status = "fallback_text"
                warnings = data.get("warnings", [])
                warnings.append("Model output did not follow JSON format; extracted text answer")
            else:
                # Valid JSON with content field
                parse_status = "json_ok"
                warnings = data.get("warnings", [])
            
            # Infer content type if not provided
            content_type = data.get("content_type", "text")
            if not data.get("content_type"):
                content_type = self._infer_content_type(content)
            
            # Normalize confidence
            confidence = data.get("confidence", 0.5)
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))
            
            # Lower confidence for fallback parsing
            if parse_status == "fallback_text":
                confidence = min(confidence, 0.6)
            
            return {
                "content": content,
                "content_type": content_type,
                "language": data.get("language"),
                "reasoning_trace": data.get("reasoning_trace", []),
                "confidence": confidence,
                "warnings": warnings,
                "imports": data.get("imports", []),
                "key_points": data.get("key_points", []),
                "target_file": data.get("target_file"),
                "tool_calls": data.get("tool_calls", []),
                "parse_status": parse_status,
                "raw_output": raw
            }
            
        except Exception as e:
            # Catastrophic parse error - use pure text extraction
            logger.exception("Sainik.parse_output: unexpected error during parsing")
            text_answer = self._extract_text_answer(raw)
            
            return {
                "content": text_answer,
                "content_type": "text",
                "confidence": 0.5,
                "reasoning_trace": [],
                "warnings": [f"Parse error: {str(e)}; using raw text"],
                "parse_status": "error_fallback",
                "raw_output": raw
            }

    def _extract_text_answer(self, raw: str) -> str:
        """
        Extract answer from raw text using heuristics when JSON parsing fails.
        
        Heuristics (in order):
        1. Look for "Answer:" or "Result:" prefix patterns
        2. Look for last code block if present
        3. Take last non-empty line
        4. Return full text if nothing else works
        """
        import re
        
        if not raw:
            return ""
        
        # Try to find explicit answer markers
        answer_patterns = [
            r'(?i)(?:final\s*)?answer\s*[:=]\s*(.+)',
            r'(?i)result\s*[:=]\s*(.+)',
            r'(?i)solution\s*[:=]\s*(.+)',
            r'(?i)output\s*[:=]\s*(.+)'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, raw, re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                if answer:
                    return answer
        
        # Try to extract code block if present (might be the answer)
        code_block_pattern = r'```(?:\w+)?\s*\n([\s\S]*?)\n```'
        code_blocks = re.findall(code_block_pattern, raw)
        if code_blocks:
            # Use last code block
            last_block = code_blocks[-1].strip()
            if last_block:
                return last_block
        
        # Try last non-empty line (common pattern: model explains, then gives answer)
        lines = [line.strip() for line in raw.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            # Prefer last line if it's short (likely a direct answer)
            if len(last_line) < 200:
                return last_line
        
        # Give up and return first 500 chars of raw text
        return raw[:500] if len(raw) > 500 else raw
    
    def _infer_content_type(self, content: str) -> str:
        """Infer content type from content."""
        # Check for code indicators
        code_indicators = [
            "def ", "class ", "import ", "from ", "return ",  # Python
            "function ", "const ", "let ", "var ",  # JavaScript
            "public ", "private ", "void ", "int ",  # Java/C++
        ]
        
        for indicator in code_indicators:
            if indicator in content:
                return "code"
        
        # Check for numeric answer
        content_stripped = content.strip()
        try:
            float(content_stripped)
            return "numeric"
        except ValueError:
            pass
        
        return "text"
    
    def create_candidate(self, role_input: RoleInput) -> Candidate:
        """Execute Sainik and return a Candidate object."""
        output = self(role_input)
        
        if output.status == "error":
            return Candidate(
                content="Error generating solution",
                content_type="text",
                confidence=0.0,
                warnings=[f"Implementor error: {output.error}"]
            )
        
        return Candidate.from_dict(output.core_output)
