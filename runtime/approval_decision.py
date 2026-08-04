from enum import Enum


class ApprovalDecision(str, Enum):

    APPROVED = "approved"

    REJECTED = "rejected"