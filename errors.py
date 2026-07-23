from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorType(Enum):
    PLANNER = "planner"
    TOOL = "tool"
    INFRASTRUCTURE = "infrastructure"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class OrionError(Exception):
    error_type: ErrorType
    message: str
    recoverable: bool
    source: str = "unknown"
    original_exception: Optional[Exception] = None

    def __str__(self):
        return self.message

    def __post_init__(self):
        super().__init__(self.message)