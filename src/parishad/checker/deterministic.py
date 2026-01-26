"""
Deterministic checking tools for Parishad.

These are "free" checks that don't require LLM calls:
- JSON schema validation
- Math expression evaluation
- Code syntax checking
- Code execution with tests
- Format validation
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jsonschema


# ==============================================================================
# Standalone helper functions (stateless, for direct use)
# ==============================================================================

def validate_schema(role_output: dict, schema: dict) -> dict:
    """
    Validate a role output dict against a JSON schema.
    
    Args:
        role_output: The output dict to validate
        schema: JSON schema to validate against
        
    Returns:
        Compact dict: {"ok": bool, "error": Optional[str]}
    """
    try:
        jsonschema.validate(role_output, schema)
        return {"ok": True, "error": None}
    except jsonschema.ValidationError as e:
        return {"ok": False, "error": f"{e.message} at {'.'.join(str(p) for p in e.path)}"}
    except jsonschema.SchemaError as e:
        return {"ok": False, "error": f"Invalid schema: {e.message}"}


def check_math(expression: str) -> dict:
    """
    Safely evaluate a simple math expression.
    
    Only allows basic operators (+, -, *, /, parentheses) and numbers.
    Uses AST parsing with node type whitelisting for safety.
    
    Args:
        expression: Math expression string (e.g., "2 + 3 * 4")
        
    Returns:
        Compact dict: {"ok": bool, "result": Optional[float], "error": Optional[str]}
    """
    # Whitelist of allowed AST node types (ast.Constant is the modern replacement for ast.Num)
    ALLOWED_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,  # Python 3.8+ for numbers, strings, etc.
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,      # Unary minus
        ast.UAdd,      # Unary plus
    )
    
    def _validate_node(node: ast.AST) -> bool:
        """Recursively check all nodes are in whitelist."""
        if not isinstance(node, ALLOWED_NODES):
            return False
        # For Constant nodes, only allow numeric types
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, complex)):
                return False
        for child in ast.iter_child_nodes(node):
            if not _validate_node(child):
                return False
        return True
    
    # Clean the expression
    expression = expression.strip()
    if not expression:
        return {"ok": False, "result": None, "error": "Empty expression"}
    
    try:
        # Parse to AST
        tree = ast.parse(expression, mode="eval")
        
        # Validate all nodes are safe
        if not _validate_node(tree):
            return {"ok": False, "result": None, "error": "Expression contains disallowed operations"}
        
        # Compile and evaluate
        code = compile(tree, "<math>", "eval")
        result = eval(code, {"__builtins__": {}}, {})
        
        # Handle division by zero
        if isinstance(result, float) and (result != result or abs(result) == float('inf')):
            return {"ok": False, "result": None, "error": "Division error (inf or nan)"}
        
        return {"ok": True, "result": float(result), "error": None}
        
    except SyntaxError as e:
        return {"ok": False, "result": None, "error": f"Syntax error: {e}"}
    except ZeroDivisionError:
        return {"ok": False, "result": None, "error": "Division by zero"}
    except Exception as e:
        return {"ok": False, "result": None, "error": f"Evaluation error: {e}"}


def run_code_tests(
    code: str,
    test_code: str,
    timeout: int = 10,
    language: str = "python"
) -> dict:
    """
    Run code with tests in isolated environment.
    
    Executes in a temporary directory with a timeout.
    
    Args:
        code: The code to test
        test_code: Test code to run against the solution
        timeout: Maximum execution time in seconds
        language: Programming language (currently only "python" supported)
        
    Returns:
        Compact dict: {"ok": bool, "stdout": str, "stderr": str, "returncode": int}
    """
    if language != "python":
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Unsupported language: {language}",
            "returncode": -1
        }
    
    # Create combined test file
    combined_code = f'''{code}

# ===== TEST CODE =====
{test_code}
'''
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file
            code_file = os.path.join(tmpdir, "solution.py")
            with open(code_file, "w") as f:
                f.write(combined_code)
            
            # Run with timeout
            result = subprocess.run(
                ["python", code_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            )
            
            # Truncate output to avoid large strings
            stdout = result.stdout[:2000] if result.stdout else ""
            stderr = result.stderr[:2000] if result.stderr else ""
            
            return {
                "ok": result.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds",
            "returncode": -1
        }
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Execution error: {e}",
            "returncode": -1
        }


# ==============================================================================
# Dataclasses for structured results
# ==============================================================================


@dataclass
class CheckResult:
    """Result from a single check."""
    
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeterministicCheckResults:
    """Aggregated results from deterministic checks."""
    
    checks: list[CheckResult]
    all_passed: bool
    critical_failure: bool = False
    failure_reason: Optional[str] = None
    
    @classmethod
    def from_checks(cls, checks: list[CheckResult]) -> "DeterministicCheckResults":
        """Create from list of check results."""
        all_passed = all(c.passed for c in checks)
        # Find critical failures (e.g., JSON parse failure when JSON expected)
        critical = [c for c in checks if not c.passed and c.details.get("critical", False)]
        critical_failure = len(critical) > 0
        failure_reason = critical[0].message if critical else None
        return cls(
            checks=checks,
            all_passed=all_passed,
            critical_failure=critical_failure,
            failure_reason=failure_reason,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "all_passed": self.all_passed,
            "critical_failure": self.critical_failure,
            "failure_reason": self.failure_reason,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }


class DeterministicChecker:
    """
    Collection of deterministic (free) checks.
    
    These checks don't require LLM inference and should be run
    before any LLM-based verification to catch obvious errors.
    """
    
    def __init__(self):
        """Initialize with default checks enabled."""
        self._custom_checks: list[Callable] = []
    
    def register_check(self, check_fn: Callable[[str, dict], CheckResult]) -> None:
        """Register a custom check function."""
        self._custom_checks.append(check_fn)
    
    def check_json_parseable(
        self,
        text: str,
        schema: Optional[dict] = None,
        critical: bool = True,
    ) -> CheckResult:
        """
        Check if text is valid JSON and optionally validate against schema.
        
        Args:
            text: Text to parse as JSON
            schema: Optional JSON schema to validate against
            critical: Whether parse failure is critical
            
        Returns:
            CheckResult with parsing/validation status
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return CheckResult(
                name="json_parse",
                passed=False,
                message=f"JSON parse error: {e}",
                details={"critical": critical, "position": e.pos},
            )
        
        if schema is not None:
            try:
                jsonschema.validate(data, schema)
            except jsonschema.ValidationError as e:
                return CheckResult(
                    name="json_schema",
                    passed=False,
                    message=f"Schema validation error: {e.message}",
                    details={"path": list(e.path), "critical": False},
                )
        
        return CheckResult(
            name="json_parse",
            passed=True,
            message="Valid JSON",
            details={"parsed": data},
        )
    
    def check_python_syntax(
        self,
        code: str,
        critical: bool = True,
    ) -> CheckResult:
        """
        Check if Python code has valid syntax.
        
        Args:
            code: Python code to check
            critical: Whether syntax error is critical
            
        Returns:
            CheckResult with syntax status
        """
        try:
            ast.parse(code)
            return CheckResult(
                name="python_syntax",
                passed=True,
                message="Valid Python syntax",
            )
        except SyntaxError as e:
            return CheckResult(
                name="python_syntax",
                passed=False,
                message=f"Syntax error: {e.msg} at line {e.lineno}",
                details={
                    "critical": critical,
                    "line": e.lineno,
                    "offset": e.offset,
                },
            )
    
    def check_math_expression(
        self,
        expression: str,
        expected_result: Optional[float] = None,
        tolerance: float = 1e-6,
    ) -> CheckResult:
        """
        Safely evaluate a math expression.
        
        Args:
            expression: Math expression to evaluate
            expected_result: Optional expected value
            tolerance: Tolerance for floating point comparison
            
        Returns:
            CheckResult with evaluation status
        """
        # Safe subset of allowed operations
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "int": int,
            "float": float,
        }
        
        try:
            # Parse the expression
            tree = ast.parse(expression, mode="eval")
            
            # Compile with restricted builtins
            code = compile(tree, "<math>", "eval")
            
            # Evaluate in restricted namespace
            result = eval(code, {"__builtins__": {}}, allowed_names)
            
            if expected_result is not None:
                if abs(result - expected_result) <= tolerance:
                    return CheckResult(
                        name="math_eval",
                        passed=True,
                        message=f"Correct: {result}",
                        details={"result": result, "expected": expected_result},
                    )
                else:
                    return CheckResult(
                        name="math_eval",
                        passed=False,
                        message=f"Wrong answer: got {result}, expected {expected_result}",
                        details={"result": result, "expected": expected_result},
                    )
            
            return CheckResult(
                name="math_eval",
                passed=True,
                message=f"Evaluated to: {result}",
                details={"result": result},
            )
            
        except Exception as e:
            return CheckResult(
                name="math_eval",
                passed=False,
                message=f"Evaluation error: {e}",
                details={"critical": False},
            )
    
    def check_format(
        self,
        text: str,
        pattern: str,
        description: str = "format",
    ) -> CheckResult:
        """
        Check if text matches a regex pattern.
        
        Args:
            text: Text to check
            pattern: Regex pattern to match
            description: Human-readable description of expected format
            
        Returns:
            CheckResult with match status
        """
        try:
            if re.search(pattern, text, re.MULTILINE | re.DOTALL):
                return CheckResult(
                    name="format_check",
                    passed=True,
                    message=f"Matches {description} format",
                )
            else:
                return CheckResult(
                    name="format_check",
                    passed=False,
                    message=f"Does not match {description} format",
                    details={"pattern": pattern},
                )
        except re.error as e:
            return CheckResult(
                name="format_check",
                passed=False,
                message=f"Invalid regex pattern: {e}",
            )
    
    def check_contains_answer(
        self,
        text: str,
        answer_patterns: Optional[list[str]] = None,
    ) -> CheckResult:
        """
        Check if text contains a properly formatted answer.
        
        Args:
            text: Text to check for answer
            answer_patterns: List of patterns that indicate an answer
            
        Returns:
            CheckResult indicating if answer format is present
        """
        if answer_patterns is None:
            answer_patterns = [
                r"(?:answer|result|solution).*?[:=]\s*\S+",
                r"\\boxed\{.+?\}",
                r"####\s*\S+",
                r"```[\w]*\n.+?\n```",
            ]
        
        for pattern in answer_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return CheckResult(
                    name="answer_present",
                    passed=True,
                    message="Answer format detected",
                    details={"pattern": pattern},
                )
        
        return CheckResult(
            name="answer_present",
            passed=False,
            message="No answer format detected",
            details={"checked_patterns": len(answer_patterns)},
        )
    
    def check_length(
        self,
        text: str,
        min_length: int = 0,
        max_length: int = 100000,
    ) -> CheckResult:
        """
        Check if text length is within bounds.
        
        Args:
            text: Text to check
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            
        Returns:
            CheckResult with length status
        """
        length = len(text)
        
        if length < min_length:
            return CheckResult(
                name="length_check",
                passed=False,
                message=f"Too short: {length} < {min_length}",
                details={"length": length, "min": min_length},
            )
        
        if length > max_length:
            return CheckResult(
                name="length_check",
                passed=False,
                message=f"Too long: {length} > {max_length}",
                details={"length": length, "max": max_length},
            )
        
        return CheckResult(
            name="length_check",
            passed=True,
            message=f"Length OK: {length}",
            details={"length": length},
        )
    
    def check_no_placeholders(
        self,
        text: str,
    ) -> CheckResult:
        """
        Check that output doesn't contain placeholder text.
        
        Args:
            text: Text to check
            
        Returns:
            CheckResult indicating if placeholders were found
        """
        placeholder_patterns = [
            r"\[insert\s+.*?\]",
            r"\[TODO\]",
            r"\[PLACEHOLDER\]",
            r"<your.*?here>",
            r"\.{3,}",  # Multiple dots as placeholder
            r"\[\.{3}\]",
        ]
        
        for pattern in placeholder_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return CheckResult(
                    name="no_placeholders",
                    passed=False,
                    message=f"Placeholder detected: '{match.group()}'",
                    details={"match": match.group()},
                )
        
        return CheckResult(
            name="no_placeholders",
            passed=True,
            message="No placeholders detected",
        )
    
    def run_all(
        self,
        text: str,
        task_type: str = "general",
        context: Optional[dict] = None,
    ) -> DeterministicCheckResults:
        """
        Run all applicable checks based on task type.
        
        Args:
            text: Output text to check
            task_type: Type of task (code, math, general)
            context: Additional context for checks
            
        Returns:
            Aggregated check results
        """
        context = context or {}
        checks: list[CheckResult] = []
        
        # Universal checks
        checks.append(self.check_length(text, min_length=1))
        checks.append(self.check_no_placeholders(text))
        checks.append(self.check_contains_answer(text))
        
        # Task-specific checks
        if task_type == "code":
            # Extract code blocks and check syntax
            code_pattern = r"```(?:python)?\n?(.*?)```"
            code_matches = re.findall(code_pattern, text, re.DOTALL)
            if code_matches:
                for i, code in enumerate(code_matches):
                    result = self.check_python_syntax(code.strip())
                    result.name = f"python_syntax_{i}"
                    checks.append(result)
        
        elif task_type == "math":
            # Look for math expressions
            math_pattern = r"####\s*([0-9+\-*/().\s]+)"
            math_matches = re.findall(math_pattern, text)
            if math_matches:
                for expr in math_matches[:3]:  # Limit checks
                    checks.append(self.check_math_expression(expr.strip()))
        
        elif task_type == "json":
            # Check JSON validity
            json_schema = context.get("json_schema")
            checks.append(self.check_json_parseable(text, json_schema))
        
        # Run custom checks
        for check_fn in self._custom_checks:
            try:
                result = check_fn(text, context)
                if result is not None:
                    checks.append(result)
            except Exception as e:
                checks.append(CheckResult(
                    name="custom_check",
                    passed=False,
                    message=f"Custom check error: {e}",
                ))
        
        return DeterministicCheckResults.from_checks(checks)
