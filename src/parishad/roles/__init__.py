"""Role implementations for Parishad council."""

from .base import (
    Role,
    RoleInput,
    RoleOutput,
    RoleMetadata,
    Trace,
    Slot,
    TaskType,
    Difficulty,
    OutputFormat,
    TaskSpec,
    Plan,
    PlanStep,
    Candidate,
    Verdict,
    CheckerFlag,
    Evidence,
    FinalAnswer,
)
# New Roles
from .raja import Raja
from .dandadhyaksha import Dandadhyaksha
from .vidushak import Vidushak
from .sacheev import Sacheev
from .prerak import Prerak
from .majumdar import Majumdar
from .pantapradhan import Pantapradhan
from .darbari import Darbari
from .sar_senapati import SarSenapati
from .sainik import Sainik


__all__ = [
    # Base classes
    "Role",
    "RoleInput",
    "RoleOutput",
    "RoleMetadata",
    "Trace",
    
    # Enums
    "Slot",
    "TaskType",
    "Difficulty",
    "OutputFormat",
    
    # Data types
    "TaskSpec",
    "Plan",
    "PlanStep",
    "Candidate",
    "Verdict",
    "CheckerFlag",
    "Evidence",
    "FinalAnswer",
    
    # Roles
    "Raja",
    "Dandadhyaksha",
    "Sacheev",
    "Prerak",
    "Majumdar",
    "Pantapradhan",
    "Darbari",
    "SarSenapati",
    "Sainik",
    "Vidushak",
]

