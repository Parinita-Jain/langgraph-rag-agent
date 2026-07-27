import pytest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from registry import Tool

from planner import planner_node
from executor import executor_node
from registry import register_tool, clear_registry
from schemas import PlannerOutput, PlanStep

class FakeLLMSingleStep:

    def invoke(self, prompt):

        return PlannerOutput(
            steps=[
                PlanStep(
                    id=1,
                    tool="echo",
                    tool_input="Hello Orion",
                    depends_on=[]
                )
            ]
        )

class FakeLLMMultiStep:

    def invoke(self, prompt):

        return PlannerOutput(
            steps=[
                PlanStep(
                    id=1,
                    tool="echo",
                    tool_input="Hello Orion",
                    depends_on=[]
                ),
                PlanStep(
                    id=2,
                    tool="echo",
                    tool_input="Repeat: #1.answer",
                    depends_on=[1]
                )
            ]
        )

class FakeLLMParallel:

    def invoke(self, prompt):

        return PlannerOutput(
            steps=[
                PlanStep(
                    id=1,
                    tool="echo",
                    tool_input="Apple",
                    depends_on=[]
                ),
                PlanStep(
                    id=2,
                    tool="echo",
                    tool_input="Banana",
                    depends_on=[]
                ),
                PlanStep(
                    id=3,
                    tool="combine",
                    tool_input="#1.answer + #2.answer",
                    depends_on=[1, 2]
                )
            ]
        )

class FakeLLMFailure:

    def invoke(self, prompt):

        return PlannerOutput(
            steps=[
                PlanStep(
                    id=1,
                    tool="echo",
                    tool_input="Apple",
                    depends_on=[]
                ),
                PlanStep(
                    id=2,
                    tool="fail",
                    tool_input="Boom",
                    depends_on=[]
                ),
                PlanStep(
                    id=3,
                    tool="combine",
                    tool_input="#1.answer + #2.answer",
                    depends_on=[1, 2]
                )
            ]
        )

from langchain_core.messages import AIMessage
from registry import Tool

@pytest.fixture(autouse=True)
def setup_registry():

    clear_registry()

    def echo_tool(state):
        text = state["tool_input"]

        return {
            "messages": [
                AIMessage(content=text)
            ],
            "output": {
                "answer": text
            },
            "success": True,
            "error": None,
        }

    register_tool(
        Tool(
            name="echo",
            function=echo_tool,
            description="Echo tool",
            outputs=["answer"],
        )
    )

    def combine_tool(state):

        text = state["tool_input"]

        return {
            "messages": [
                AIMessage(content=text)
            ],
            "output": {
                "answer": text
            },
            "success": True,
            "error": None,
        }

    register_tool(
        Tool(
            name="combine",
            function=combine_tool,
            description="Combine tool",
            outputs=["answer"],
        )
    )

    def fail_tool(state):

        return {
            "messages": [
                AIMessage(content="Tool failed")
            ],
            "output": {},
            "success": False,
            "error": "Intentional failure",
        }

    register_tool(
        Tool(
            name="fail",
            function=fail_tool,
            description="Always fails",
            outputs=[],
        )
    )

    yield

    clear_registry()

@patch("planner.get_structured_llm")
def test_planner_executor_integration(mock_llm):

    mock_llm.return_value = FakeLLMSingleStep()

    state = {
        "messages": [
            HumanMessage(content="Say hello")
        ],
        "context": {},
        "tool_results": {},
        "execution_records": []
    }

    plan = planner_node(state)

    assert plan["error"] is None
    assert len(plan["steps"]) == 1

    state["steps"] = plan["steps"]

    result = executor_node(state)

    assert result["tool_results"][1]["output"]["answer"] == "Hello Orion"

    assert result["context"]["step_1"]["answer"] == "Hello Orion"

    assert len(result["execution_records"]) == 1

@patch("planner.get_structured_llm")
def test_sequential_workflow_integration(mock_llm):

    mock_llm.return_value = FakeLLMMultiStep()

    state = {
        "messages": [
            HumanMessage(
                content="Say hello and then repeat it."
            )
        ],
        "context": {},
        "tool_results": {},
        "execution_records": [],
    }

    plan = planner_node(state)

    assert plan["error"] is None
    assert len(plan["steps"]) == 2

    state["steps"] = plan["steps"]

    result = executor_node(state)

    # Step 1 output
    assert (
        result["tool_results"][1]["output"]["answer"]
        == "Hello Orion"
    )

    # Step 2 output
    assert (
        result["tool_results"][2]["output"]["answer"]
        == "Repeat: Hello Orion"
    )

    # Context updated correctly
    assert (
        result["context"]["step_1"]["answer"]
        == "Hello Orion"
    )

    assert (
        result["context"]["step_2"]["answer"]
        == "Repeat: Hello Orion"
    )

    # Two execution records
    assert len(result["execution_records"]) == 2

@patch("planner.get_structured_llm")
def test_parallel_workflow_integration(mock_llm):

    mock_llm.return_value = FakeLLMParallel()

    state = {
        "messages": [
            HumanMessage(
                content="Tell me Apple and Banana, then combine them."
            )
        ],
        "context": {},
        "tool_results": {},
        "execution_records": [],
    }

    plan = planner_node(state)

    assert plan["error"] is None
    assert len(plan["steps"]) == 3

    state["steps"] = plan["steps"]

    result = executor_node(state)

    # Step 1
    assert (
        result["tool_results"][1]["output"]["answer"]
        == "Apple"
    )

    # Step 2
    assert (
        result["tool_results"][2]["output"]["answer"]
        == "Banana"
    )

    # Step 3
    assert (
        result["tool_results"][3]["output"]["answer"]
        == "Apple + Banana"
    )

    # Context
    assert result["context"]["step_1"]["answer"] == "Apple"
    assert result["context"]["step_2"]["answer"] == "Banana"
    assert result["context"]["step_3"]["answer"] == "Apple + Banana"

    assert len(result["execution_records"]) == 3

@patch("planner.get_structured_llm")
def test_failure_propagation(mock_llm):

    mock_llm.return_value = FakeLLMFailure()

    state = {
        "messages": [
            HumanMessage(
                content="Run a workflow that fails."
            )
        ],
        "context": {},
        "tool_results": {},
        "execution_records": [],
    }

    plan = planner_node(state)

    assert plan["error"] is None
    assert len(plan["steps"]) == 3

    state["steps"] = plan["steps"]

    result = executor_node(state)

    # Step 1 succeeded
    assert result["tool_results"][1]["success"] is True
    assert result["tool_results"][1]["output"]["answer"] == "Apple"

    # Step 2 failed
    assert result["tool_results"][2]["success"] is False
    assert result["tool_results"][2]["error"] == "Intentional failure"

    # Step 3 should never execute
    assert result["tool_results"][3]["status"] == "SKIPPED"

    assert "Skipped because dependencies" in result["tool_results"][3]["error"]
    assert "[2]" in result["tool_results"][3]["error"]

    # Context should only contain successful step
    assert "step_1" in result["context"]
    assert "step_2" not in result["context"]
    assert "step_3" not in result["context"]



