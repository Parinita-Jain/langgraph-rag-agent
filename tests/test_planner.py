import pytest

import importlib

import tools

from langchain_core.messages import HumanMessage

from planner import planner_node, MAX_REPAIR_ATTEMPTS

from unittest.mock import patch

from schemas import PlannerOutput, PlanStep

from registry import clear_registry

@pytest.fixture(autouse=True)
def setup_tools():

    clear_registry()

    importlib.reload(tools)

    yield

    clear_registry()

def test_greeting_rule():

    state = {
        "messages": [
            HumanMessage(content="Hello")
        ]
    }

    result = planner_node(state)

    assert len(result["steps"]) == 1
    assert result["steps"][0].tool == "direct"
    assert result["steps"][0].id == 1
    assert result["error"] is None


def test_calculator_rule():

    state = {
        "messages": [
            HumanMessage(content="25*7")
        ]
    }

    result = planner_node(state)

    assert len(result["steps"]) == 1
    assert result["steps"][0].tool == "calculator"
    assert result["steps"][0].tool_input == "25*7"
    assert result["error"] is None


def test_calculator_rule_with_spaces():

    state = {
        "messages": [
            HumanMessage(content="20 + 30")
        ]
    }

    result = planner_node(state)

    assert result["steps"][0].tool == "calculator"


def test_non_greeting_goes_to_llm(monkeypatch):

    class FakeStructuredLLM:

        def invoke(self, prompt):
            raise RuntimeError("LLM called")

    class FakeLLM:

        def with_structured_output(self, schema):
            return FakeStructuredLLM()

    monkeypatch.setattr(
        "planner.llm",
        FakeLLM(),
    )

    state = {
        "messages": [
            HumanMessage(content="Hello there")
        ]
    }

    result = planner_node(state)

    assert result["steps"] == []
    assert result["error"] is not None
    assert result["error"].source == "planner"
    assert result["error"].recoverable is True
    assert result["error"].message == "LLM called"


def test_valid_llm_plan():

    planner_output = PlannerOutput(
        steps=[
            PlanStep(
                id=1,
                tool="llm",
                tool_input="Explain AI",
                depends_on=[]
            )
        ]
    )

    class FakeStructuredLLM:

        def invoke(self, prompt):
            return planner_output

    with patch(
        "planner.get_structured_llm",
        return_value=FakeStructuredLLM(),
    ):

        state = {
            "messages": [
                HumanMessage(content="Explain AI")
            ]
        }

        result = planner_node(state)

    assert len(result["steps"]) == 1
    assert result["steps"][0].tool == "llm"
    assert result["steps"][0].tool_input == "Explain AI"
    assert result["error"] is None

def test_planner_repairs_invalid_plan():

    invalid_plan = PlannerOutput(
        steps=[
            PlanStep(
                id=1,
                tool="unknown_tool",
                tool_input="Explain AI",
                depends_on=[]
            )
        ]
    )

    repaired_plan = PlannerOutput(
        steps=[
            PlanStep(
                id=1,
                tool="llm",
                tool_input="Explain AI",
                depends_on=[]
            )
        ]
    )

    class FakeStructuredLLM:

        def invoke(self, prompt):
            return invalid_plan

    with (
        patch(
            "planner.get_structured_llm",
            return_value=FakeStructuredLLM(),
        ),
        patch(
            "planner.repair_plan",
            return_value=repaired_plan,
        ) as mock_repair,
    ):

        state = {
            "messages": [
                HumanMessage(content="Explain AI")
            ]
        }

        result = planner_node(state)

    mock_repair.assert_called_once()

    assert len(result["steps"]) == 1
    assert result["steps"][0].tool == "llm"
    assert result["error"] is None

def test_planner_repair_failure():

    invalid_plan = PlannerOutput(
        steps=[
            PlanStep(
                id=1,
                tool="unknown_tool",
                tool_input="Explain AI",
                depends_on=[]
            )
        ]
    )

    class FakeStructuredLLM:

        def invoke(self, prompt):
            return invalid_plan

    with (
        patch(
            "planner.get_structured_llm",
            return_value=FakeStructuredLLM(),
        ),
        patch(
            "planner.repair_plan",
            return_value=invalid_plan,
        ) as mock_repair,
    ):

        state = {
            "messages": [
                HumanMessage(content="Explain AI")
            ]
        }

        with pytest.raises(ValueError) as exc:
            planner_node(state)

    assert mock_repair.call_count == MAX_REPAIR_ATTEMPTS

    assert "Planner could not repair the plan" in str(exc.value)