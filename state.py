from typing import Annotated, Sequence, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from schemas import PlanStep


class AgentState(TypedDict):

    messages: Annotated[
        Sequence[BaseMessage],
        add_messages
    ]

    steps: list[PlanStep]

    tool_input: str

    documents: list

    output: dict[str, Any]