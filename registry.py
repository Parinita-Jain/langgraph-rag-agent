from dataclasses import dataclass
from typing import Callable

TOOL_REGISTRY = {}

print("Registry module:", __file__)
print("Registry id:", id(TOOL_REGISTRY))


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
    TOOL_REGISTRY[tool.name] = tool


def get_tool(name):
    return TOOL_REGISTRY.get(name)


def list_tools():
    return TOOL_REGISTRY


def get_tool_descriptions():

    return "\n\n".join(
        tool.planner_description()
        for tool in TOOL_REGISTRY.values()
    )