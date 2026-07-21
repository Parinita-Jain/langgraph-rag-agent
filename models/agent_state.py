from typing import TypedDict, Any

from models.plan import PlanStep
from models.execution_record import ExecutionRecord


class AgentState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """

    messages: list[Any]

    steps: list[PlanStep]

    tool_results: dict[int, Any]

    execution_records: list[ExecutionRecord]

    context: dict[str, Any]

    done: bool

    iteration: int

    errors: list[str]