from typing import Literal
from pydantic import BaseModel


class PlanStep(BaseModel):

    id: int

    tool: Literal[
        "direct",
        "calculator",
        "rag"
    ]

    tool_input: str

    depends_on: list[int] = []

class PlannerOutput(BaseModel):

    steps: list[PlanStep]