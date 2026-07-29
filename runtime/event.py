import time
from dataclasses import dataclass, field
from typing import Any

from runtime.event_types import WorkflowEventType


@dataclass
class WorkflowEvent:
    type: WorkflowEventType
    timestamp: float = field(default_factory=time.time)
    step_id: int | None = None
    tool: str | None = None

    payload: dict[str, Any] = field(default_factory=dict)