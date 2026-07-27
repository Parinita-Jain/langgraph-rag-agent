import pytest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from registry import Tool

from planner import planner_node
from executor import executor_node
from registry import register_tool, clear_registry
from schemas import PlannerOutput, PlanStep

class FakeLLM:

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

    yield

    clear_registry()

@patch("planner.get_structured_llm")
def test_planner_executor_integration(mock_llm):

    mock_llm.return_value = FakeLLM()

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

