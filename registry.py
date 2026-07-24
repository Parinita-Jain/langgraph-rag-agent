from dataclasses import dataclass
from typing import Callable

TOOL_REGISTRY = {}


@dataclass
class Tool:

    name: str
    function: Callable
    description: str
    outputs: list[str]
    parallel_safe: bool = True
    retries: int = 2
    timeout: int = 30

    def planner_description(self):

        return (
            f"{self.name}\n"
            f"- {self.description}\n"
            f"Outputs: {', '.join(self.outputs)}"
        )

def register_tool(tool: Tool):

    if tool.name in TOOL_REGISTRY:
        raise ValueError(
            f"Tool '{tool.name}' is already registered."
        )

    TOOL_REGISTRY[tool.name] = tool

def clear_registry():
    TOOL_REGISTRY.clear()


def get_tool(name):
    return TOOL_REGISTRY.get(name)

def list_tools():
    return TOOL_REGISTRY


def get_tool_descriptions():

    return "\n\n".join(
        tool.planner_description()
        for tool in TOOL_REGISTRY.values()
    )