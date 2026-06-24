from typing import Literal
from pydantic import BaseModel


class RouteDecision(BaseModel):
    route: Literal["direct", "retrieve"]