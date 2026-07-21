from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """
    Standard return type for every Orion tool.
    """

    success: bool

    output: dict[str, Any] = field(default_factory=dict)

    messages: list[Any] = field(default_factory=list)

    error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)