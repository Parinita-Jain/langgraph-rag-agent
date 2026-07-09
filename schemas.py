from typing import Literal
from pydantic import BaseModel


class ToolDecision(BaseModel):
    tool: Literal[
        "direct",
        "calculator",
        "rag"
    ]

    tool_input: str