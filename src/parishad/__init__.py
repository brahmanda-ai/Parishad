"""
Parishad - A cost-aware, local-first council of heterogeneous LLMs.

Parishad orchestrates multiple local language models into a structured "council"
that achieves higher reliability than a single model under strict compute budgets.
"""

__version__ = "0.1.7"

from .orchestrator.engine import Parishad, ParishadEngine, PipelineConfig
from .models.runner import ModelRunner, ModelConfig
from .roles.base import (
    Role,
    RoleInput,
    RoleOutput,
    Trace,
    Slot,
    TaskSpec,
    Plan,
    Candidate,
    Verdict,
    FinalAnswer,
)
# Roles
from .roles import (
    Darbari,
    Majumdar,
    Sainik,
    Prerak,
    Raja,
    Pantapradhan,
    SarSenapati,
    Sacheev,
    Dandadhyaksha,
)


__all__ = [
    # Main API
    "Parishad",
    "ParishadEngine",
    
    # Configuration
    "ModelConfig",
    "ModelRunner",
    "PipelineConfig",
    
    # Roles
    "Role",
    "Darbari",
    "Majumdar",
    "Sainik",
    "Prerak",
    "Raja",
    "Pantapradhan",
    "SarSenapati",
    "Sacheev",
    "Dandadhyaksha",
    
    # Data types
    "RoleInput",
    "RoleOutput",
    "Trace",
    "Slot",
    "TaskSpec",
    "Plan",
    "Candidate",
    "Verdict",
    "FinalAnswer",
]
