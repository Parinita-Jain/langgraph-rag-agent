from enum import Enum


class FailureReason(str, Enum):
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    VALIDATION = "validation"