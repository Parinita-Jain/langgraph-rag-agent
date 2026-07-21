from dataclasses import dataclass, field


@dataclass
class PlanStep:
    """
    One executable step in a plan.
    """

    id: int
    tool: str
    tool_input: str
    depends_on: list[int] = field(default_factory=list)
    output: str | None = None


@dataclass
class Plan:
    """
    Complete execution plan.
    """

    steps: list[PlanStep]