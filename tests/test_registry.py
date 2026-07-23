#  unit test file for a custom tool registry system

import pytest

from registry import (
    Tool,
    clear_registry,
    get_tool,
    get_tool_descriptions,
    list_tools,
    register_tool,
)


def dummy_tool(state):
    return {
        "success": True,
        "output": {},
    }


@pytest.fixture(autouse=True)
def clean_registry():
    clear_registry()
    yield
    clear_registry()


def test_register_tool():

    tool = Tool(
        name="dummy",
        function=dummy_tool,
        description="Dummy tool",
        outputs=["result"],
    )

    register_tool(tool)

    assert get_tool("dummy") is tool

def test_duplicate_tool_registration():

    tool = Tool(
        name="dummy",
        function=dummy_tool,
        description="Dummy tool",
        outputs=["result"],
    )

    register_tool(tool)

    with pytest.raises(ValueError):
        register_tool(tool)

def test_get_unknown_tool():

    assert get_tool("missing") is None


def test_list_tools():

    tool = Tool(
        name="dummy",
        function=dummy_tool,
        description="Dummy tool",
        outputs=["result"],
    )

    register_tool(tool)

    tools = list_tools()

    assert len(tools) == 1
    assert "dummy" in tools
    assert tools["dummy"] is tool


def test_tool_description():

    tool = Tool(
        name="dummy",
        function=dummy_tool,
        description="Dummy tool",
        outputs=["result"],
    )

    register_tool(tool)

    description = get_tool_descriptions()

    assert "dummy" in description
    assert "Dummy tool" in description
    assert "Outputs: result" in description

def test_planner_description():

    tool = Tool(
        name="calculator",
        function=dummy_tool,
        description="Performs calculations",
        outputs=["answer"],
    )

    description = tool.planner_description()

    assert "calculator" in description
    assert "Performs calculations" in description
    assert "Outputs: answer" in description