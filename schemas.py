from typing import Literal
from pydantic import BaseModel
from typing import Optional

class PlanStep(BaseModel):

    id: int

    tool: str

    tool_input: str

    depends_on: list[int]

    output: Optional[str] = None

class PlannerOutput(BaseModel):

    steps: list[PlanStep]

class ReplannerOutput(BaseModel):

    done: bool

    steps: list[PlanStep]