"""
Base classes and types for Parishad roles.

All functional roles (Refiner, Planner, Worker, Checker, Judge) inherit from
the abstract Role class defined here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Dict
import json
import uuid
import logging

# Schema validation disabled
ROLE_SCHEMA = None
SCHEMA_VALIDATION_AVAILABLE = False


logger = logging.getLogger(__name__)


def validate_role_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate role output against JSON schema.
    
    Args:
        data: Role output dict to validate
        
    Returns:
        Dict with 'ok' (bool) and optional 'error' (str) keys
    """
    if not SCHEMA_VALIDATION_AVAILABLE or not ROLE_SCHEMA:
        return {"ok": True, "warning": "Schema validation not available"}
    
    try:
        # Create a schema that includes both envelope and definitions for $ref resolution
        envelope_schema = ROLE_SCHEMA.get("definitions", {}).get("envelope", {})
        if envelope_schema:
            # Build a complete schema with definitions for $ref resolution
            full_schema = {
                **envelope_schema,
                "definitions": ROLE_SCHEMA.get("definitions", {})
            }
            jsonschema.validate(instance=data, schema=full_schema)
        return {"ok": True}
    except jsonschema.ValidationError as e:
        error_msg = f"Schema validation failed: {e.message}"
        if e.path:
            error_msg += f" at path: {'.'.join(str(p) for p in e.path)}"
        return {"ok": False, "error": error_msg}
    except Exception as e:
        return {"ok": False, "error": f"Validation error: {str(e)}"}


class Slot(Enum):
    """Model slot sizes for heterogeneous council."""
    SMALL = "small"  # 2-4B: Refiner, Checker
    MID = "mid"      # 7-13B: Worker
    BIG = "big"      # 13-34B: Planner, Judge


class TaskType(Enum):
    """Types of tasks Parishad can handle."""
    CODE = "code"
    MATH = "math"
    QA = "qa"
    EXPLANATION = "explanation"
    CREATIVE = "creative"
    ANALYSIS = "analysis"


class Difficulty(Enum):
    """Task difficulty for routing decisions."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class OutputFormat(Enum):
    """Expected output format types."""
    CODE = "code"
    TEXT = "text"
    NUMERIC = "numeric"
    STRUCTURED = "structured"
    MIXED = "mixed"


@dataclass
class RoleMetadata:
    """Metadata about a role execution."""
    tokens_used: int = 0
    latency_ms: int = 0
    model_id: str = ""
    slot: Slot = Slot.MID
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: int = 0  # Added for Task 5
    schema_warning: Optional[str] = None  # Added for Task 2
    # Phase-3 Task 2: Truncation tracking for Judge
    worker_truncated: bool = False
    checker_truncated: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        result = {
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "model_id": self.model_id,
            "slot": self.slot.value,
            "timestamp": self.timestamp.isoformat()
        }
        if self.duration_ms > 0:
            result["duration_ms"] = self.duration_ms
        if self.schema_warning:
            result["schema_warning"] = self.schema_warning
        if self.worker_truncated:
            result["worker_truncated"] = self.worker_truncated
        if self.checker_truncated:
            result["checker_truncated"] = self.checker_truncated
        return result


@dataclass
class RoleInput:
    """Standard input structure for roles."""
    user_query: str
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)  # Phase-3: routing metadata
    
    # Previous role outputs (populated by orchestrator)
    task_spec: Optional[dict] = None
    plan: Optional[dict] = None
    candidate: Optional[dict] = None
    verdict: Optional[dict] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "user_query": self.user_query,
            "context": self.context,
            "metadata": self.metadata,
            "task_spec": self.task_spec,
            "plan": self.plan,
            "candidate": self.candidate,
            "verdict": self.verdict
        }


@dataclass
class RoleOutput:
    """Standard output structure from roles."""
    role: str
    status: str  # "success", "error", "partial"
    core_output: dict[str, Any]
    metadata: RoleMetadata = field(default_factory=RoleMetadata)
    error: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "status": self.status,
            "output": self.core_output,
            "metadata": self.metadata.to_dict(),
            "error": self.error
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class TaskSpec:
    """Structured task specification from Refiner."""
    problem: str
    constraints: list[str] = field(default_factory=list)
    output_format: OutputFormat = OutputFormat.TEXT
    difficulty_guess: Difficulty = Difficulty.MEDIUM
    task_type: TaskType = TaskType.QA
    key_concepts: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "constraints": self.constraints,
            "output_format": self.output_format.value,
            "difficulty_guess": self.difficulty_guess.value,
            "task_type": self.task_type.value,
            "key_concepts": self.key_concepts
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        return cls(
            problem=data.get("problem", ""),
            constraints=data.get("constraints", []),
            output_format=OutputFormat(data.get("output_format", "text")),
            difficulty_guess=Difficulty(data.get("difficulty_guess", "medium")),
            task_type=TaskType(data.get("task_type", "qa")),
            key_concepts=data.get("key_concepts", [])
        )


@dataclass
class PlanStep:
    """A single step in a plan."""
    id: int
    description: str
    rationale: str = ""
    expected_output: str = ""
    depends_on: list[int] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "rationale": self.rationale,
            "expected_output": self.expected_output,
            "depends_on": self.depends_on
        }


@dataclass
class Plan:
    """Structured plan from Planner."""
    steps: list[PlanStep]
    checkpoints: list[int] = field(default_factory=list)
    expected_output_type: str = "text"
    complexity_estimate: str = "moderate"
    suggested_approach: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "checkpoints": self.checkpoints,
            "expected_output_type": self.expected_output_type,
            "complexity_estimate": self.complexity_estimate,
            "suggested_approach": self.suggested_approach
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        steps = [
            PlanStep(
                id=s.get("id", i+1),
                description=s.get("description", ""),
                rationale=s.get("rationale", ""),
                expected_output=s.get("expected_output", ""),
                depends_on=s.get("depends_on", [])
            )
            for i, s in enumerate(data.get("steps", []))
        ]
        return cls(
            steps=steps,
            checkpoints=data.get("checkpoints", []),
            expected_output_type=data.get("expected_output_type", "text"),
            complexity_estimate=data.get("complexity_estimate", "moderate"),
            suggested_approach=data.get("suggested_approach", "")
        )


@dataclass
class Candidate:
    """Candidate output from Worker."""
    content: str
    content_type: str = "text"
    language: Optional[str] = None  # For code
    target_file: Optional[str] = None  # Target file for write operations
    reasoning_trace: list[str] = field(default_factory=list)
    confidence: float = 0.5
    warnings: list[str] = field(default_factory=list)
    
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "content_type": self.content_type,
            "language": self.language,
            "target_file": self.target_file,
            "reasoning_trace": self.reasoning_trace,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "tool_calls": self.tool_calls
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        return cls(
            content=data.get("content", ""),
            content_type=data.get("content_type", "text"),
            language=data.get("language"),
            target_file=data.get("target_file"),
            reasoning_trace=data.get("reasoning_trace", []),
            confidence=data.get("confidence", 0.5),
            warnings=data.get("warnings", []),
            tool_calls=data.get("tool_calls", [])
        )


@dataclass
class CheckerFlag:
    """A flag raised by the Checker."""
    type: str
    severity: str  # low, medium, high, critical
    detail: str
    location: Optional[str] = None
    suggested_fix: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "detail": self.detail,
            "location": self.location,
            "suggested_fix": self.suggested_fix
        }


@dataclass
class Evidence:
    """Evidence item from Checker verification."""
    source: str
    source_type: str  # retrieval, deterministic, llm_judgment
    snippet: str = ""
    relevance_score: float = 0.0
    supports_claim: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_type": self.source_type,
            "snippet": self.snippet,
            "relevance_score": self.relevance_score,
            "supports_claim": self.supports_claim
        }


@dataclass
class Verdict:
    """Checker verdict on Worker output."""
    flags: list[CheckerFlag] = field(default_factory=list)
    must_fix: bool = False
    evidence: list[Evidence] = field(default_factory=list)
    suggested_edits: list[str] = field(default_factory=list)
    overall_confidence: float = 0.5
    checks_performed: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "flags": [f.to_dict() for f in self.flags],
            "must_fix": self.must_fix,
            "evidence": [e.to_dict() for e in self.evidence],
            "suggested_edits": self.suggested_edits,
            "overall_confidence": self.overall_confidence,
            "checks_performed": self.checks_performed
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Verdict":
        flags = [
            CheckerFlag(
                type=f.get("type", "unknown"),
                severity=f.get("severity", "low"),
                detail=f.get("detail", ""),
                location=f.get("location"),
                suggested_fix=f.get("suggested_fix")
            )
            for f in data.get("flags", [])
        ]
        evidence = [
            Evidence(
                source=e.get("source", ""),
                source_type=e.get("source_type", "unknown"),
                snippet=e.get("snippet", ""),
                relevance_score=e.get("relevance_score", 0.0),
                supports_claim=e.get("supports_claim", True)
            )
            for e in data.get("evidence", [])
        ]
        return cls(
            flags=flags,
            must_fix=data.get("must_fix", False),
            evidence=evidence,
            suggested_edits=data.get("suggested_edits", []),
            overall_confidence=data.get("overall_confidence", 0.5),
            checks_performed=data.get("checks_performed", [])
        )


@dataclass
class FinalAnswer:
    """Final answer from Judge."""
    final_answer: str
    answer_type: str = "text"
    rationale: str = ""
    confidence: float = 0.5
    caveats: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    numeric_answer: Optional[float] = None  # For math problems
    code_block: Optional[str] = None  # For code problems
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "answer_type": self.answer_type,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "caveats": self.caveats,
            "sources_used": self.sources_used,
            "numeric_answer": self.numeric_answer,
            "code_block": self.code_block
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinalAnswer":
        return cls(
            final_answer=data.get("final_answer", ""),
            answer_type=data.get("answer_type", "text"),
            rationale=data.get("rationale", ""),
            confidence=data.get("confidence", 0.5),
            caveats=data.get("caveats", []),
            sources_used=data.get("sources_used", []),
            numeric_answer=data.get("numeric_answer"),
            code_block=data.get("code_block")
        )


@dataclass
class Trace:
    """Complete execution trace for a Parishad run."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: str = "core"
    timestamp: datetime = field(default_factory=datetime.now)
    user_query: str = ""
    total_tokens: int = 0
    total_latency_ms: int = 0
    budget_initial: int = 8000
    budget_remaining: int = 8000
    roles: list[RoleOutput] = field(default_factory=list)
    retries: int = 0
    final_answer: Optional[FinalAnswer] = None
    success: bool = True
    error: Optional[str] = None
    budget_exceeded: bool = False  # Flag indicating budget was exceeded during execution
    budget_enforcement_triggered: bool = False  # True if roles were skipped due to budget
    skipped_roles: list[dict] = field(default_factory=list)  # Roles skipped (with reason)
    validation_errors: list[str] = field(default_factory=list)  # Roles with validation errors
    
    def to_dict(self) -> dict[str, Any]:
        result = {
            "query_id": self.query_id,
            "config": self.config,
            "timestamp": self.timestamp.isoformat(),
            "user_query": self.user_query,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "budget_initial": self.budget_initial,
            "budget_remaining": self.budget_remaining,
            "roles": [r.to_dict() for r in self.roles],
            "retries": self.retries,
            "final_answer": self.final_answer.to_dict() if self.final_answer else None,
            "success": self.success,
            "error": self.error,
            "budget_exceeded": self.budget_exceeded,
        }
        # Only include if triggered/non-empty
        if self.budget_enforcement_triggered:
            result["budget_enforcement_triggered"] = True
        if self.skipped_roles:
            result["skipped_roles"] = self.skipped_roles
        if self.validation_errors:
            result["validation_errors"] = self.validation_errors
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def add_role_output(self, output: RoleOutput) -> None:
        """Add a role output and update totals."""
        self.roles.append(output)
        self.total_tokens += output.metadata.tokens_used
        self.total_latency_ms += output.metadata.latency_ms
        self.budget_remaining -= output.metadata.tokens_used


class Role(ABC):
    """
    Abstract base class for all Parishad roles.
    
    Each role:
    - Has a default slot (SMALL, MID, BIG)
    - Has a system prompt template
    - Produces structured JSON output
    - Can be invoked with a RoleInput and returns a RoleOutput
    """
    
    name: str = "base"
    default_slot: Slot = Slot.MID
    
    def __init__(
        self,
        model_runner: Any,  # ModelRunner instance
        slot: Optional[Slot] = None,
        max_tokens: int = 1024,
        temperature: float = 0.5
    ):
        self.model_runner = model_runner
        self.slot = slot or self.default_slot
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for this role."""
        pass
    
    @abstractmethod
    def format_input(self, role_input: RoleInput) -> str:
        """Format the role input into a user message."""
        pass
    
    @abstractmethod
    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse the raw LLM output into structured data."""
        pass
    
    def __call__(self, role_input: RoleInput) -> RoleOutput:
        """Execute this role."""
        import time
        from ..models.runner import ModelRunnerError, UnknownSlotError, ModelBackendError
        
        start_time = time.perf_counter()
        tokens_from_backend = 0  # Track tokens separately to preserve them even on parse error
        raw_output_for_debug = ""  # Save for error reporting
        model_id_from_backend = None
        
        try:
            # Format input
            user_message = self.format_input(role_input)
            
            # Call model - wrap backend errors
            try:
                raw_output, tokens_used, model_id = self.model_runner.generate(
                    system_prompt=self.system_prompt,
                    user_message=user_message,
                    slot=self.slot,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                # Save these in case parse_output fails
                tokens_from_backend = tokens_used
                raw_output_for_debug = raw_output[:500]  # First 500 chars for debugging
                model_id_from_backend = model_id
            except (UnknownSlotError, ModelBackendError) as e:
                # Normalize backend errors into RoleOutput
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                logger.error(f"Backend error in {self.name}: {e}")
                
                error_type = "unknown_slot" if isinstance(e, UnknownSlotError) else "backend_error"
                
                return RoleOutput(
                    role=self.name,
                    status="error",
                    core_output={
                        "error_type": error_type,
                        "error_message": str(e),
                        "backend_error": True
                    },
                    metadata=RoleMetadata(
                        latency_ms=latency_ms,
                        slot=self.slot,
                        tokens_used=0
                    ),
                    error=str(e)
                )
            
            # Parse output - wrap to preserve tokens even if this fails
            try:
                core_output = self.parse_output(raw_output)
            except Exception as parse_error:
                # Parse failed, but backend DID generate - preserve tokens!
                logger.error(
                    f"Parse error in {self.name}: {parse_error}. "
                    f"Backend generated {tokens_from_backend} tokens. "
                    f"Raw output preview: {raw_output_for_debug}"
                )
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                
                return RoleOutput(
                    role=self.name,
                    status="error",
                    core_output={
                        "error_type": "parse_error",
                        "error_message": str(parse_error),
                        "raw_output_preview": raw_output_for_debug
                    },
                    metadata=RoleMetadata(
                        latency_ms=latency_ms,
                        slot=self.slot,
                        tokens_used=tokens_from_backend,  # PRESERVE TOKENS!
                        model_id=model_id_from_backend
                    ),
                    error=f"Parse error: {parse_error}"
                )
            
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            
            # Build metadata
            metadata = RoleMetadata(
                tokens_used=tokens_from_backend,
                latency_ms=latency_ms,
                model_id=model_id_from_backend,
                slot=self.slot
            )
            
            # Create output
            output = RoleOutput(
                role=self.name,
                status="success",
                core_output=core_output,
                metadata=metadata
            )
            
            # Soft schema validation - don't fail, just warn
            validation_result = validate_role_output({
                "role": output.role,
                "status": output.status,
                "output": output.core_output,
                "metadata": metadata.to_dict()
            })
            
            if not validation_result.get("ok", True):
                # Add schema warning to metadata but keep status as success
                logger.warning(
                    f"Schema validation failed for {self.name}: "
                    f"{validation_result.get('error', 'Unknown error')}"
                )
                # Store warning in metadata by converting to dict, updating, and creating new metadata
                metadata_dict = metadata.to_dict()
                metadata_dict["schema_warning"] = validation_result.get("error", "Validation failed")
                # Reconstruct with warning
                output.metadata = RoleMetadata(
                    tokens_used=tokens_from_backend,
                    latency_ms=latency_ms,
                    model_id=model_id_from_backend,
                    slot=self.slot,
                    timestamp=metadata.timestamp
                )
            
            return output
            
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Role {self.name} failed: {e}")
            return RoleOutput(
                role=self.name,
                status="error",
                core_output={},
                metadata=RoleMetadata(
                    latency_ms=latency_ms,
                    slot=self.slot,
                    tokens_used=tokens_from_backend  # Preserve tokens even on total failure
                ),
                error=str(e)
            )
    
    def _extract_json(self, text: str) -> dict[str, Any]:
        """
        Extract JSON from LLM output.
        
        Handles cases where JSON is wrapped in markdown code blocks.
        """
        import re
        
        # Try to find JSON in code blocks first
        json_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # Try to parse the entire text as JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON-like structure
        brace_pattern = r'\{[\s\S]*\}'
        matches = re.findall(brace_pattern, text)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                # Fallback: Try ast.literal_eval for Python-style dicts (common in weaker models)
                try:
                    import ast
                    # Only safe evaluation
                    return ast.literal_eval(match)
                except (ValueError, SyntaxError):
                    continue
        
        # Return raw text as fallback
        return {"raw_output": text}
