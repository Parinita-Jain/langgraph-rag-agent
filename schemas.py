from typing import Literal
from pydantic import BaseModel


class ToolCall(BaseModel):

    tool: Literal[
        "direct",
        "calculator",
        "rag"
    ]

    tool_input: str

class PlannerOutput(BaseModel):

    tool_calls: list[ToolCall]