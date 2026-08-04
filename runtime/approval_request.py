from dataclasses import dataclass


@dataclass(slots=True)
class ApprovalRequest:

    step_id: int

    tool: str

    reason: str