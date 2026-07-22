from enum import Enum
from dataclasses import dataclass
from typing import Optional

class ErrorType(Enum):
    PLANNER = "planner"
    TOOL = "tool"
    INFRASTRUCTURE = "infrastructure"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class OrionError:
    error_type: ErrorType
    message: str
    recoverable: bool
    source: str = "unknown"
    original_exception: Optional[Exception] = None
    