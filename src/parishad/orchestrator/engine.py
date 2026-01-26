"""
Orchestrator engine for Parishad council pipeline.

Executes role graphs with budget tracking and retry logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json
import logging
import uuid

import yaml

from ..models.runner import ModelRunner, ModelConfig
from ..roles.base import (
    Role,
    RoleInput,
    RoleOutput,
    Trace,
    FinalAnswer,
    Slot,
)
from ..roles import (
    Darbari, Majumdar, Sainik, Prerak, Raja,
    Pantapradhan, SarSenapati, Sacheev, Dandadhyaksha,
    Vidushak,
)
from .config_loader import load_pipeline_config, RoleSpec
from .exceptions import InvalidPipelineConfigError


logger = logging.getLogger(__name__)


# Registry mapping role class names to Role classes
# Used by config-driven pipeline execution
ROLE_REGISTRY: dict[str, type[Role]] = {
    # Core roles
    "Darbari": Darbari,
    "Majumdar": Majumdar,
    "Sainik": Sainik,
    "Prerak": Prerak,
    "Raja": Raja,
    # Extended roles
    "Pantapradhan": Pantapradhan,
    "SarSenapati": SarSenapati,
    "Sacheev": Sacheev,
    "Dandadhyaksha": Dandadhyaksha,
    "Vidushak": Vidushak,
}


@dataclass
class Budget:
    """Runtime budget tracker for token and cost management."""
    max_tokens: int = 8000
    max_cost: float = 1.0
    used_tokens: int = 0
    used_cost: float = 0.0
    
    def spend(self, tokens: int = 0, cost: float = 0.0) -> None:
        """Record spending of tokens and cost."""
        self.used_tokens += tokens
        self.used_cost += cost
    
    @property
    def remaining_tokens(self) -> int:
        """Get remaining token budget."""
        return max(0, self.max_tokens - self.used_tokens)
    
    @property
    def remaining_cost(self) -> float:
        """Get remaining cost budget."""
        return max(0.0, self.max_cost - self.used_cost)
    
    @property
    def is_exceeded(self) -> bool:
        """Check if budget is exceeded."""
        return self.used_tokens > self.max_tokens or self.used_cost > self.max_cost
    
    @property
    def token_percent_used(self) -> float:
        """Get percentage of token budget used."""
        if self.max_tokens == 0:
            return 100.0
        return (self.used_tokens / self.max_tokens) * 100
    
    @property
    def cost_percent_used(self) -> float:
        """Get percentage of cost budget used."""
        if self.max_cost == 0:
            return 100.0
        return (self.used_cost / self.max_cost) * 100


@dataclass
class BudgetConfig:
    """Configuration for token budget management."""
    max_tokens_per_query: int = 8000
    min_budget_for_retry: int = 1500
    
    role_budgets: dict[str, int] = field(default_factory=lambda: {
        "darbari": 600,
        "majumdar": 1200,
        "sainik": 2500,
        "prerak": 1000,
        "raja": 1800,
        "reserve": 900
    })


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    enabled: bool = True
    max_retries: int = 1
    retry_roles: list[str] = field(default_factory=lambda: ["sainik"])


@dataclass
class DifficultyRouting:
    """Configuration for difficulty-based model routing."""
    enabled: bool = True
    easy_planner_slot: str = "mid"
    easy_judge_slot: str = "mid"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    name: str = "parishad-core"
    version: str = "0.1.0"
    
    roles: dict[str, dict] = field(default_factory=dict)
    pipeline: list[str] = field(default_factory=lambda: [
        "darbari", "majumdar", "sainik", "prerak", "raja"
    ])
    
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    difficulty_routing: DifficultyRouting = field(default_factory=DifficultyRouting)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        budget_data = data.get("budget", {})
        budget = BudgetConfig(
            max_tokens_per_query=budget_data.get("max_tokens_per_query", 8000),
            min_budget_for_retry=budget_data.get("min_budget_for_retry", 1500),
            role_budgets=budget_data.get("role_budgets", {})
        )
        
        retry_data = data.get("retry", {})
        retry = RetryConfig(
            enabled=retry_data.get("enabled", True),
            max_retries=retry_data.get("max_retries", 1),
            retry_roles=retry_data.get("retry_roles", ["sainik"])
        )
        
        routing_data = data.get("difficulty_routing", {})
        routing = DifficultyRouting(
            enabled=routing_data.get("enabled", True),
            easy_planner_slot=routing_data.get("rules", {}).get("easy", {}).get("planner_slot", "mid"),
            easy_judge_slot=routing_data.get("rules", {}).get("easy", {}).get("judge_slot", "mid")
        )
        
        return cls(
            name=data.get("name", "parishad-core"),
            version=data.get("version", "0.1.0"),
            roles=data.get("roles", {}),
            pipeline=data.get("pipeline", []),
            budget=budget,
            retry=retry,
            difficulty_routing=routing
        )


@dataclass
class ExecutionContext:
    """Context maintained during pipeline execution."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_query: str = ""
    
    # Budget tracking
    budget_initial: int = 8000
    budget_remaining: int = 8000
    tokens_used: int = 0
    
    # Core role outputs
    task_spec: Optional[dict] = None
    plan: Optional[dict] = None
    candidate: Optional[dict] = None
    verdict: Optional[dict] = None
    final_answer: Optional[dict] = None
    
    # Extended role outputs
    plan_high: Optional[dict] = None      # From Pantapradhan
    plan_exec: Optional[dict] = None      # From SarSenapati
    verdict_fact: Optional[dict] = None    # From Sacheev
    verdict_safety: Optional[dict] = None  # From Dandadhyaksha
    
    # Retry tracking
    retry_count: int = 0
    
    # Trace
    role_outputs: list[RoleOutput] = field(default_factory=list)
    
    # Budget enforcement tracking
    skipped_roles: list[dict] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    budget_enforcement_triggered: bool = False
    budget_exceeded: bool = False
    
    # Routing decision (Task 1 - Router integration)
    routing_decision: Optional[RoutingDecision] = None
    
    def use_tokens(self, tokens: int) -> None:
        """Record token usage."""
        self.tokens_used += tokens
        self.budget_remaining -= tokens
    
    def has_budget(self, min_required: int = 0) -> bool:
        """Check if we have enough budget."""
        return self.budget_remaining >= min_required
    
    def to_role_input(self) -> RoleInput:
        """Create RoleInput from current context."""
        # Build extended context dict for Extended pipeline roles
        extended_context = {}
        if self.plan_high is not None:
            extended_context["plan_high"] = self.plan_high
        if self.plan_exec is not None:
            extended_context["plan_exec"] = self.plan_exec
        if self.verdict_fact is not None:
            extended_context["verdict_fact"] = self.verdict_fact
        if self.verdict_safety is not None:
            extended_context["verdict_safety"] = self.verdict_safety
        
        # Phase-3 Task 1: Add routing decision to metadata (not context) for roles to access
        metadata = {}
        if self.routing_decision is not None:
            metadata["routing"] = {
                "config_name": self.routing_decision.config_name,
                "mode": self.routing_decision.mode,
                "allow_retry": self.routing_decision.allow_retry,
                "checker_mode": self.routing_decision.checker_mode,
                "truncation_policy": self.routing_decision.truncation_policy,
                "max_tokens": self.routing_decision.max_tokens,
                "per_role_max_tokens": self.routing_decision.per_role_max_tokens,
            }
        
        return RoleInput(
            user_query=self.user_query,
            task_spec=self.task_spec,
            plan=self.plan or self.plan_exec,  # Extended uses plan_exec as plan
            candidate=self.candidate,
            verdict=self.verdict,
            context=extended_context,
            metadata=metadata,
        )


class ParishadEngine:
    """
    Main orchestrator engine for Parishad council.
    
    Executes the role pipeline with budget tracking, difficulty routing,
    and retry logic.
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        model_runner: Optional[ModelRunner] = None,
        trace_dir: Optional[str | Path] = None,
        strict_validation: bool = False,
        enforce_budget: bool = False,
        mode: str = "balanced",
        user_forced_config: Optional[str] = None,
        **kwargs  # Ignore legacy mock/stub args
    ):
        """
        Initialize the Parishad engine.
        
        Args:
            model_config: Configuration for model slots
            pipeline_config: Configuration for pipeline execution
            model_runner: Pre-configured ModelRunner (optional)
            trace_dir: Directory to save execution traces
            strict_validation: If True, set status="error" when schema validation
                fails instead of soft warning (default: False)
            enforce_budget: If True, skip optional roles when budget is low
                (default: False)
            mode: Execution mode ("auto"|"fast"|"balanced"|"thorough")
        :param user_forced_config: Dict with slot overrides (model_id, backend_type)
        """
        # 1. Resolve ModelConfig
        # Check for profile in kwargs (passed by CLI)
        profile = kwargs.get("profile")
        if not model_config and profile:
            try:
                # Assuming standard config path or None (defaults)
                model_config = ModelConfig.from_profile(profile)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to load profile '{profile}': {e}")
        
        self.model_config = model_config or ModelConfig()
        
        # 2. Apply user_forced_config overrides
        # Ensure it's a dict (handling potential type hint mismatch)
        if user_forced_config and isinstance(user_forced_config, dict):
            import copy
            for slot_name, overrides in user_forced_config.items():
                if slot_name in self.model_config.slots:
                    # Deepcopy to prevent shared references (e.g. from YAML anchors)
                    # causing overrides to one slot affecting others.
                    slot_cfg = copy.deepcopy(self.model_config.slots[slot_name])
                    
                    if "model_id" in overrides:
                        slot_cfg.model_id = overrides["model_id"]
                        # Default to None, override if provided
                        slot_cfg.model_file = overrides.get("model_file")
                    if "backend_type" in overrides:
                        slot_cfg.backend = overrides["backend_type"]  # Correct field name is 'backend'
                    if "model_file" in overrides:
                         slot_cfg.model_file = overrides["model_file"]
                    
                    # Update config with new object
                    self.model_config.slots[slot_name] = slot_cfg
                else:
                    # Create new slot if it doesn't exist
                    from ..models.runner import SlotConfig
                    backend_type = overrides.get("backend_type", "ollama")
                    model_id = overrides.get("model_id", "")
                    model_file = overrides.get("model_file")
                    
                    # Use defaults for other fields
                    slot_cfg = SlotConfig(
                        model_id=model_id,
                        backend=backend_type,
                        model_file=model_file,  # Pass explicit model file
                        context_length=32768,   # Force high context defaults for local models
                        default_max_tokens=2048, 
                        default_temperature=0.7
                    )
                    self.model_config.slots[slot_name] = slot_cfg

        self.pipeline_config = pipeline_config or PipelineConfig()
        
        # Validation and budget enforcement flags
        self.strict_validation = strict_validation
        self.enforce_budget = enforce_budget
        
        # Router integration (Task 1)
        self.mode = mode  # "auto" | "fast" | "balanced" | "thorough"
        self.user_forced_config = user_forced_config  # Config explicitly set by user
        
        # Use provided runner or create new one
        self.model_runner = model_runner or ModelRunner(
            config=self.model_config
        )
        
        self.trace_dir = Path(trace_dir) if trace_dir else None
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            
        # Store cwd for tools
        self.cwd = kwargs.get("cwd") or Path.cwd()
        
        # Initialize roles
        self._init_roles()
    
    def _init_roles(self) -> None:
        """Initialize role instances."""
        role_configs = self.pipeline_config.roles
        
        
        # Tools initialization (Phase 13)
        from ..tools.fs import FileSystemTool
        from ..tools.shell import ShellTool
        from ..tools.perception import PerceptionTool
        
        # Use stored cwd or default
        cwd = self.cwd 
        self.fs_tool = FileSystemTool(working_directory=str(cwd))
        self.shell_tool = ShellTool(safe_mode=True) # Safe mode by default
        
        # Configure Perception with Vision Model (Phase 13)
        vision_slot_name = "vision" if "vision" in self.model_config.slots else "small"
        vision_slot = self.model_config.slots.get(vision_slot_name)
        
        perception_config = None
        if vision_slot:
            # Construct config for MarkItDown (uses OpenAI client)
            # Ollama provides OpenAI compatible API at /v1
            base_url = vision_slot.extra.get("base_url", "http://localhost:11434/v1")
            # Ensure base_url ends with /v1 for OpenAI client compatibility if using Ollama
            if "localhost" in base_url and "/v1" not in base_url:
                 base_url = f"{base_url.rstrip('/')}/v1"
                
            perception_config = {
                "base_url": base_url,
                "api_key": "ollama",
                "model": vision_slot.model_file or vision_slot.model_id or "llava"
            }
            
        self.perception_tool = PerceptionTool(llm_config=perception_config) 
        
        self.darbari = Darbari(
            model_runner=self.model_runner,
            **self._get_role_kwargs("darbari", role_configs.get("darbari", {}))
        )
        
        self.majumdar = Majumdar(
            model_runner=self.model_runner,
            **self._get_role_kwargs("majumdar", role_configs.get("majumdar", {}))
        )
        
        self.sainik = Sainik(
            model_runner=self.model_runner,
            tools=[self.fs_tool, self.shell_tool, self.perception_tool],
            **self._get_role_kwargs("sainik", role_configs.get("sainik", {}))
        )
        
        self.prerak = Prerak(
            model_runner=self.model_runner,
            tools=role_configs.get("prerak", {}).get("tools", []),
            **self._get_role_kwargs("prerak", role_configs.get("prerak", {}))
        )
        
        self.raja = Raja(
            model_runner=self.model_runner,
            **self._get_role_kwargs("raja", role_configs.get("raja", {}))
        )
    
    def _get_role_kwargs(self, role_name: str, config: dict) -> dict:
        """Extract role initialization kwargs from config."""
        kwargs = {}
        
        if "slot" in config:
            slot_name = config["slot"]
            kwargs["slot"] = Slot(slot_name)
        
        if "max_tokens" in config:
            kwargs["max_tokens"] = config["max_tokens"]
        
        if "temperature" in config:
            kwargs["temperature"] = config["temperature"]
        
        return kwargs
    
    def _load_pipeline(self, config_name: str = "core") -> list[RoleSpec]:
        """Load pipeline configuration from YAML."""
        return load_pipeline_config(config_name)
    
    def _get_context_updates_for_role(self, role_name: str) -> dict[str, str]:
        """
        Return context field mapping for a role.
        """
        mappings = {
            # Core
            "darbari": {"task_spec": "core_output"},
            "majumdar": {"plan": "core_output"},
            "sainik": {"candidate": "core_output"},
            "prerak": {"verdict": "core_output"},
            "raja": {"final_answer": "core_output"},
            
            # Extended
            "pantapradhan": {"plan_high": "core_output"},
            "sar_senapati": {"plan_exec": "core_output", "plan": "core_output"},
            "sacheev": {"verdict_fact": "core_output"},
            "dandadhyaksha": {"verdict_safety": "core_output"},
        }
        return mappings.get(role_name, {})
    
    def _is_optional_role(self, role_name: str) -> bool:
        """Determine if a role is optional."""
        optional_roles = {
            "sacheev",
            "dandadhyaksha",
        }
        return role_name in optional_roles
    
    def _estimate_role_tokens(self, role_name: str, role_spec: Optional[RoleSpec] = None) -> int:
        """Estimate token usage for a role based on configuration."""
        if role_spec and role_spec.budget_tokens > 0:
            return role_spec.budget_tokens

        role_budgets = self.pipeline_config.budget.role_budgets
        
        # Use configured budget if available
        if role_name in role_budgets:
            return role_budgets[role_name]
        
        # Default estimates
        default_estimates = {
            "pantapradhan": 1000,
            "sar_senapati": 800,
            "sacheev": 600,
            "dandadhyaksha": 400,
        }
        
        return default_estimates.get(role_name, 500)
    
    def _get_role_instance(self, role_name: str, role_spec: Optional[RoleSpec] = None) -> Role:
        """Get the role instance by name."""
        # Core roles (pre-initialized)
        core_roles = {
            "darbari": self.darbari,
            "majumdar": self.majumdar,
            "sainik": self.sainik,
            "prerak": self.prerak,
            "raja": self.raja,
        }
        
        if role_name in core_roles:
            return core_roles[role_name]
        
        # Extended roles (dynamically instantiated)
        if not hasattr(self, "_extended_roles"):
            self._extended_roles: dict[str, Role] = {}
        
        if role_name in self._extended_roles:
            return self._extended_roles[role_name]
        
        # Instantiate extended role
        extended_class_map = {
            "pantapradhan": Pantapradhan,
            "sar_senapati": SarSenapati,
            "sacheev": Sacheev,
            "dandadhyaksha": Dandadhyaksha,
            "vidushak": Vidushak,
            # Aliases for compatibility if needed
            "sainik_code": Sainik,
            "sainik_text": Sainik,
        }
        
        if role_name not in extended_class_map:
            raise KeyError(f"Unknown role: {role_name}")
        
        role_class = extended_class_map[role_name]
        
        # Get base config
        role_config = self.pipeline_config.roles.get(role_name, {}).copy()
        
        # Override with spec if provided
        if role_spec:
            if role_spec.slot:
                role_config["slot"] = role_spec.slot
            if role_spec.max_tokens:
                role_config["max_tokens"] = role_spec.max_tokens
            if role_spec.temperature:
                role_config["temperature"] = role_spec.temperature
            # Merge extra config (tools, etc)
            if role_spec.extra_config:
                role_config.update(role_spec.extra_config)
        
        # Build kwargs for role initialization
        kwargs = self._get_role_kwargs(role_name, role_config)
        
        # Handle tools for checker roles
        if role_name in ("sacheev", "dandadhyaksha") and "tools" in role_config:
            kwargs["tools"] = role_config["tools"]
        
        role_instance = role_class(model_runner=self.model_runner, **kwargs)
        self._extended_roles[role_name] = role_instance
        
        return role_instance
    
    def _run_role(
        self,
        role: Role,
        role_name: str,
        ctx: ExecutionContext,
        context_updates: dict[str, str] | None = None
    ) -> RoleOutput:
        """Execute a role with consistent logging and context management."""
        import time
        from ..roles.base import validate_role_output
        
        logger.debug(f"Starting role: {role_name} (budget: {ctx.budget_remaining})")
        
        start_time = time.perf_counter()
        
        # Build role input from context
        role_input = ctx.to_role_input()
        
        # Add any special context for this role (e.g., retry context)
        # Note: 'sainik' handles retry
        if role_name == "sainik" and ctx.retry_count > 0 and ctx.verdict:
            role_input.context["is_retry"] = True
            # Truncate previous output to avoid large memory copies
            prev_content = ctx.candidate.get("content", "") if ctx.candidate else ""
            role_input.context["previous_output"] = prev_content[:1024] if len(prev_content) > 1024 else prev_content
            role_input.context["checker_feedback"] = ctx.verdict
        
        # Execute role
        output = role(role_input)
        
        # Add duration to metadata
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        output.metadata.duration_ms = duration_ms
        
        # Log completion at appropriate level
        logger.debug(
            f"Role {role_name} completed: status={output.status}, "
            f"tokens={output.metadata.tokens_used}, duration={duration_ms}ms"
        )
        
        # Strict validation check
        if self.strict_validation and output.status == "success":
            validation_result = validate_role_output({
                "role": output.role,
                "status": output.status,
                "output": output.core_output,
                "metadata": output.metadata.to_dict()
            })
            
            if not validation_result.get("ok", True):
                error_msg = validation_result.get("error", "Schema validation failed")
                logger.error(f"Strict validation failed for {role_name}: {error_msg}")
                
                # Convert to error status
                output.status = "error"
                output.error = f"Schema validation failed: {error_msg}"
                ctx.validation_errors.append(role_name)
        
        # Update context
        ctx.role_outputs.append(output)
        ctx.use_tokens(output.metadata.tokens_used)
        
        # Update context fields based on role output
        if output.status == "success" and context_updates:
            for ctx_field, output_field in context_updates.items():
                if output_field == "core_output":
                    setattr(ctx, ctx_field, output.core_output)
                else:
                    setattr(ctx, ctx_field, output.core_output.get(output_field))
                logger.debug(f"Updated context.{ctx_field} from {role_name}")
        
        # Phase 13: File Writing Capability
        # Check if Sainik wants to write a file
        if role_name == "sainik" and output.status == "success":
             # Handle dictionary (raw output)
             target_file = output.core_output.get("target_file")
             content = output.core_output.get("content")
             
             if target_file and content:
                 try:
                     # Use FS tool to write
                     logger.info(f"Writing file {target_file} via Sainik")
                     
                     # Simple content write
                     result = self.fs_tool.run("write", path=target_file, content=content)
                     
                     if not result.success:
                         logger.error(f"Failed to write file {target_file}: {result.error}")
                         output.error = f"File write failed: {result.error}"
                         # Optionally mark partial success?
                     else:
                         logger.info(f"Successfully wrote {target_file}")
                         
                 except Exception as e:
                     logger.error(f"Error handling file write for {target_file}: {e}")
                     output.error = f"File write exception: {str(e)}"

        # Phase 13: General Tool Execution (Agentic)
        if role_name == "sainik" and output.status == "success":
            tool_calls = output.core_output.get("tool_calls", [])
            for call in tool_calls:
                tool_name = call.get("tool")
                action = call.get("action")
                args = call.get("args", {})
                
                if not tool_name or not action:
                    continue
                    
                logger.info(f"Executing tool {tool_name}.{action} with args {args}")
                
                # Resolve tool instance
                tool_instance = None
                if tool_name == "file_system":
                    tool_instance = self.fs_tool
                elif tool_name == "shell":
                    tool_instance = self.shell_tool
                elif tool_name == "perception":
                    tool_instance = self.perception_tool
                
                if tool_instance:
                    try:
                        result = tool_instance.run(action, **args)
                        logger.info(f"Tool {tool_name} result: {result.success}")
                        if not result.success:
                            logger.warning(f"Tool failure: {result.message if hasattr(result, 'message') else result.error}")
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
        
        return output
    
    def run(self, query: str, config: str = "core", max_tokens: int | None = None) -> Trace:
        """Execute the Parishad council pipeline on a user query."""
        # Initialize context
        budget = max_tokens or self.pipeline_config.budget.max_tokens_per_query
        ctx = ExecutionContext(
            user_query=query,
            budget_initial=budget,
            budget_remaining=budget
        )
        
        # Add soft budget tracking
        budget_exceeded = False
        
        # Log at info level with truncated query for privacy/memory
        query_preview = query[:100] + "..." if len(query) > 100 else query
        logger.info(f"Parishad run started: id={ctx.query_id}, config={config}, budget={budget}")
        
        try:
            # Load pipeline configuration
            try:
                role_specs = self._load_pipeline(config)
            except InvalidPipelineConfigError as e:
                logger.error(f"Invalid pipeline configuration: {e}")
                raise RuntimeError(f"Pipeline configuration error: {e}") from e
            
            # Execute pipeline: config-driven loop over all roles
            for idx, role_spec in enumerate(role_specs):
                role_name = role_spec.name.lower()  # Ensure lowercase for lookups
                
                # Budget enforcement check
                if self.enforce_budget and self._is_optional_role(role_name):
                    estimated_tokens = self._estimate_role_tokens(role_name, role_spec)
                    if not ctx.has_budget(estimated_tokens):
                        logger.info(
                            f"Budget enforcement: skipping optional role {role_name} "
                            f"(need ~{estimated_tokens} tokens, have {ctx.budget_remaining})"
                        )
                        ctx.skipped_roles.append({
                            "role": role_name,
                            "reason": "budget_exceeded",
                            "tokens_needed": estimated_tokens,
                            "tokens_available": ctx.budget_remaining
                        })
                        ctx.budget_enforcement_triggered = True
                        ctx.budget_exceeded = True
                        continue                
                
                role_instance = self._get_role_instance(role_name, role_spec)
                context_updates = self._get_context_updates_for_role(role_name)
                
                output = self._run_role(role_instance, role_name, ctx, context_updates)
                
                # Router integration: DISABLED due to missing route_policy function
                # if idx == 0 and output.status == "success":
                #     # Build global_config for routing
                #     effective_forced_config = self.user_forced_config
                #     
                #     global_config = {
                #         "mode": self.mode,
                #         "config": effective_forced_config,  # None if user didn't force --config
                #         "no_retry": not self.pipeline_config.retry.enabled,
                #         "profile": getattr(self.model_runner, "profile", None),
                #     }
                #     
                #     # Call Router to get adaptive decision
                #     # decision = route_policy(
                #     #     output.core_output,
                #     #     query,
                #     #     global_config
                #     # )
                #     
                #     # Store decision in context for roles to access
                #     # ctx.routing_decision = decision
                #     
                #     # Apply routing decision if user hasn't forced a config
                #     # should_apply_routing = (
                #     #     not effective_forced_config and 
                #     #     self.mode != "balanced"  # Balanced mode with no CLI override = keep run() param
                #     # )
                #     
                #     # if should_apply_routing:
                #     #     new_config = decision.config_name
                #     #     if new_config and new_config != config:
                #     #         logger.info(
                #     #             f"Router selected pipeline: {new_config} "
                #     #             f"(mode={self.mode}, task={output.core_output.get('task_type', 'unknown')})"
                #     #         )
                #     #         # Reload pipeline with new config and skip re-running first role
                            # config = new_config
                            # new_role_specs = self._load_pipeline(config)
                            # # Continue with remaining roles from new pipeline
                            # if len(new_role_specs) > 1:
                            #     role_specs = role_specs[:idx+1] + new_role_specs[1:]
                    
                    # # Apply retry setting from Router
                    # original_retry = self.pipeline_config.retry.enabled
                    # self.pipeline_config.retry.enabled = decision.allow_retry
                    # if not decision.allow_retry and original_retry:
                    #     logger.info(f"Router disabled retry for this query")
                    
                    # # Apply budget from Router (soft limit)
                    # if decision.max_tokens and decision.max_tokens < ctx.budget_initial:
                    #     ctx.budget_initial = decision.max_tokens
                    #     logger.debug(f"Router adjusted budget to {decision.max_tokens} tokens")
                    
                    # logger.debug(
                    #     f"Routing decision: pipeline={decision.config_name}, "
                    #     f"checker_mode={decision.checker_mode}, "
                    #     f"truncation={decision.truncation_policy}"
                    # )
                
                # Strict validation: stop pipeline if role failed validation
                if self.strict_validation and output.status == "error":
                    logger.error(f"Strict validation: stopping pipeline after {role_name} error")
                    raise RuntimeError(f"Role {role_name} failed validation: {output.error}")
                
                # Check soft budget after each role
                if max_tokens and ctx.tokens_used > max_tokens:
                    logger.warning(f"Soft token budget exceeded after {role_name}: {ctx.tokens_used}/{max_tokens}")
                    ctx.budget_exceeded = True
                    budget_exceeded = True
                
                # Check for retry after checker (Prerak)
                if role_name == "prerak" and self._should_retry(ctx):
                    # Budget enforcement: skip retry if budget is low
                    if self.enforce_budget:
                        min_retry_budget = self.pipeline_config.budget.min_budget_for_retry
                        if not ctx.has_budget(min_retry_budget):
                            logger.info(
                                f"Budget enforcement: skipping retry "
                                f"(need {min_retry_budget} tokens, have {ctx.budget_remaining})"
                            )
                            ctx.skipped_roles.append({
                                "role": "retry",
                                "reason": "budget_exceeded",
                                "tokens_needed": min_retry_budget,
                                "tokens_available": ctx.budget_remaining
                            })
                            ctx.budget_enforcement_triggered = True
                            ctx.budget_exceeded = True
                            continue
                    
                    logger.info(f"Retrying Sainik (attempt {ctx.retry_count + 1})")
                    ctx.retry_count += 1
                    
                    # Re-run sainik and prerak
                    sainik_instance = self._get_role_instance("sainik")
                    sainik_updates = self._get_context_updates_for_role("sainik")
                    self._run_role(sainik_instance, "sainik", ctx, sainik_updates)
                    
                    prerak_instance = self._get_role_instance("prerak")
                    prerak_updates = self._get_context_updates_for_role("prerak")
                    self._run_role(prerak_instance, "prerak", ctx, prerak_updates)
                    
                    if max_tokens and ctx.tokens_used > max_tokens:
                        budget_exceeded = True
            
            success = True
            error = None
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            success = False
            error = str(e)
            budget_exceeded = False  # Error takes precedence
        
        # Build trace
        trace = self._build_trace(ctx, success, error)
        
        # Add budget exceeded flag to trace if applicable
        if budget_exceeded and success:
            logger.warning(f"Pipeline completed but exceeded token budget: {ctx.tokens_used}/{max_tokens}")
        
        # Save trace if configured
        if self.trace_dir:
            self._save_trace(trace)
        
        logger.info(
            f"Parishad run complete: {ctx.query_id} "
            f"(tokens: {ctx.tokens_used}/{budget}, success: {success})"
        )
        
        return trace
    
    def _should_retry(self, ctx: ExecutionContext) -> bool:
        """Determine if we should retry the Sainik."""
        if not self.pipeline_config.retry.enabled:
            return False
        
        if ctx.retry_count >= self.pipeline_config.retry.max_retries:
            return False
        
        if not ctx.verdict:
            return False
        
        if not ctx.verdict.get("must_fix", False):
            return False
        
        min_budget = self.pipeline_config.budget.min_budget_for_retry
        if not ctx.has_budget(min_budget):
            logger.info("Insufficient budget for retry")
            return False
        
        return True
    
    def _build_trace(
        self, 
        ctx: ExecutionContext, 
        success: bool, 
        error: Optional[str]
    ) -> Trace:
        """Build execution trace from context."""
        final_answer = None
        if ctx.final_answer:
            final_answer = FinalAnswer.from_dict(ctx.final_answer)
        elif ctx.candidate:
            # If no Raja in pipeline, use Sainik's output as final answer
            final_answer = FinalAnswer(
                final_answer=ctx.candidate.get("content", ""),
                answer_type=ctx.candidate.get("content_type", "text"),
                confidence=ctx.candidate.get("confidence", 0.8),
                rationale="\n".join(ctx.candidate.get("reasoning_trace", [])) if isinstance(ctx.candidate.get("reasoning_trace"), list) else str(ctx.candidate.get("reasoning_trace", "")),
                caveats=ctx.candidate.get("warnings", []),
                code_block=ctx.candidate.get("content", "") if ctx.candidate.get("content_type") == "code" else None,
            )
        
        return Trace(
            query_id=ctx.query_id,
            config=self.pipeline_config.name,
            timestamp=datetime.now(),
            user_query=ctx.user_query,
            total_tokens=ctx.tokens_used,
            total_latency_ms=sum(o.metadata.latency_ms for o in ctx.role_outputs),
            budget_initial=ctx.budget_initial,
            budget_remaining=ctx.budget_remaining,
            roles=ctx.role_outputs,
            retries=ctx.retry_count,
            final_answer=final_answer,
            success=success,
            error=error,
            budget_exceeded=ctx.budget_exceeded,
            budget_enforcement_triggered=ctx.budget_enforcement_triggered,
            skipped_roles=ctx.skipped_roles,
            validation_errors=ctx.validation_errors,
        )
    
    def _save_trace(self, trace: Trace) -> None:
        """Save trace to file."""
        if not self.trace_dir:
            return
        
        filename = f"trace_{trace.query_id}.json"
        filepath = self.trace_dir / filename
        
        with open(filepath, "w") as f:
            f.write(trace.to_json())
        
        logger.debug(f"Trace saved: {filepath}")


class Parishad:
    """
    High-level API for running Parishad council.
    
    This is the main entry point for users.
    """
    
    def __init__(
        self,
        config: str = "core",
        model_config: Optional[ModelConfig] = None,
        model_config_path: Optional[str | Path] = None,
        profile: Optional[str] = None,
        pipeline_config_path: Optional[str | Path] = None,
        trace_dir: Optional[str | Path] = None,
        strict_validation: bool = False,
        enforce_budget: bool = False,
        mode: Optional[str] = None,
        user_forced_config: Optional[str] = None,
        no_retry: bool = False,
        **kwargs  # Ignore legacy mock/stub args
    ):
        """
        Initialize Parishad.
        
        Args:
            config: Pipeline configuration ("core" or "extended")
            model_config: Direct ModelConfig object (overrides model_config_path + profile)
            model_config_path: Path to models.yaml (defaults to ~/.parishad/models.yaml if exists)
            profile: Model profile to use (defaults to user config, fallback: "local_cpu")
            pipeline_config_path: Path to pipeline config YAML
            trace_dir: Directory to save traces
            strict_validation: If True, fail on schema validation errors
            enforce_budget: If True, skip optional roles when budget is low
            mode: Execution mode ("auto"|"fast"|"balanced"|"thorough") for adaptive routing (defaults to user config, fallback: "balanced")
            user_forced_config: Config explicitly set by user (overrides routing)
            no_retry: If True, disable Worker+Checker retry logic
        """
        from ..config.user_config import load_user_config
        
        self.config_name = config
        
        # Load user config for defaults
        user_cfg = load_user_config()
        
        # Apply defaults from user config if not explicitly provided
        if profile is None:
            profile = user_cfg.default_profile
            logger.debug(f"Using default profile from user config: {profile}")
        
        if mode is None:
            mode = user_cfg.default_mode
            logger.debug(f"Using default mode from user config: {mode}")
        
        # If model_config_path not provided, try to load from config.json
        if model_config_path is None:
            # Try unified config.json first (client-side approach)
            config_json_path = Path.home() / ".parishad" / "config.json"
            if config_json_path.exists():
                try:
                    import json
                    with open(config_json_path) as f:
                        user_config_data = json.load(f)
                    
                    session = user_config_data.get("session", {})
                    model_settings = user_config_data.get("model_config", {})
                    model_path = session.get("model")
                    backend_name = session.get("backend", "llama_cpp")
                    
                    if model_path:
                        # Create ModelConfig directly from config.json
                        from ..models.runner import SlotConfig, Backend
                        
                        # Map all slots to the same model (Laghu Sabha approach)
                        slot_config = SlotConfig(
                            model_id=model_path,
                            backend=Backend(backend_name) if backend_name in [e.value for e in Backend] else Backend.LLAMA_CPP,
                            default_max_tokens=1024,
                            default_temperature=0.5,
                            extra={
                                "n_gpu_layers": model_settings.get("n_gpu_layers", -1),
                                "n_ctx": model_settings.get("n_ctx", 8192),
                            }
                        )
                        
                        model_config = ModelConfig(
                            slots={
                                "small": slot_config,
                                "mid": slot_config,
                                "big": slot_config,
                            }
                        )
                        logger.debug(f"Loaded model config from config.json: {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load model config from config.json: {e}")
            
            # Fall back to models.yaml if config.json didn't provide model config
            if model_config is None:
                user_models_path = Path.home() / ".parishad" / "models.yaml"
                if user_models_path.exists():
                    model_config_path = user_models_path
                    logger.debug(f"Using user models config: {model_config_path}")
        
        # Load configurations
        # Use provided model_config or load from file
        if model_config is None and model_config_path:
            model_config = ModelConfig.from_profile(profile, model_config_path)
        
        pipeline_config = None
        if pipeline_config_path:
            pipeline_config = PipelineConfig.from_yaml(pipeline_config_path)
        
        # Task 4: Handle no_retry flag
        if pipeline_config and no_retry:
            pipeline_config.retry.enabled = False
        
        # Create engine
        self.engine = ParishadEngine(
            model_config=model_config,
            pipeline_config=pipeline_config,
            trace_dir=trace_dir,
            strict_validation=strict_validation,
            enforce_budget=enforce_budget,
            mode=mode,
            user_forced_config=user_forced_config,
        )
    
    def run(self, query: str) -> Trace:
        """
        Run a query through the Parishad council.
        
        Args:
            query: User query to process
            
        Returns:
            Complete execution trace
        """
        return self.engine.run(query, config=self.config_name)
    
    @property
    def final_answer(self) -> Optional[FinalAnswer]:
        """Get the final answer from the last run."""
        # This would need to store the last trace
        return None
