from typing import TypedDict


class AgentState(TypedDict):

    question: str

    route: str

    documents: list

    answer: str