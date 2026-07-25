import pytest

from langchain_core.messages import AIMessage


from models import PlanStep

from registry import (
    Tool,
    register_tool,
    clear_registry,
)

from executor import (
    execute_step,
    executor_node,
)

def dummy_tool(state):
    return {
        "messages": [AIMessage(content="Done")],
        "output": {
            "answer": "Orion Test Response",
        },
        "success": True,
        "error": None,
    }

def echo_tool(state):
    return {
        "messages": [],
        "output": {
            "received": state["tool_input"],
        },
        "success": True,
        "error": None,
    }

def failing_tool(state):
    raise RuntimeError("Boom!")

def setup_function():
    clear_registry()


def teardown_function():
    clear_registry()


def test_execute_step_success():

    register_tool(
        Tool(
            name="dummy",
            function=dummy_tool,
            description="Dummy tool",
            outputs=["answer"],
        )
    )

    step = PlanStep(
        id=1,
        tool="dummy",
        tool_input="Hello",
        depends_on=[],
    )

    state = {
        "context": {},
    }

    result = execute_step(
        step,
        state,
        {},
    )

    assert result["result"]["success"] is True
    assert result["result"]["output"]["answer"] == "Orion Test Response"

    record = result["record"]

    assert record.step_id == 1
    assert record.tool == "dummy"
    assert record.success is True

def test_execute_step_unknown_tool():

    clear_registry()

    step = PlanStep(
        id=1,
        tool="unknown",
        tool_input="Hello",
        depends_on=[],
    )

    state = {
        "context": {},
    }

    with pytest.raises(ValueError) as exc:
        execute_step(
            step,
            state,
            {},
        )

    assert "not registered" in str(exc.value)

def test_execute_step_tool_without_function():

    register_tool(
        Tool(
            name="dummy",
            function=None,
            description="Dummy tool",
            outputs=["answer"],
        )
    )

    step = PlanStep(
        id=1,
        tool="dummy",
        tool_input="Hello",
        depends_on=[],
    )

    state = {
        "context": {},
    }

    with pytest.raises(ValueError) as exc:
        execute_step(
            step,
            state,
            {},
        )

    assert "has no registered function" in str(exc.value)

def test_execute_step_resolves_context_variables():

    register_tool(
        Tool(
            name="echo",
            function=echo_tool,
            description="Echo tool",
            outputs=["received"],
        )
    )

    step = PlanStep(
        id=1,
        tool="echo",
        tool_input="Weather in {city}",
        depends_on=[],
    )

    state = {
        "context": {
            "city": "Mumbai",
        },
    }

    result = execute_step(
        step,
        state,
        {},
    )

    assert (
        result["result"]["output"]["received"]
        == "Weather in Mumbai"
    )


def test_execute_step_tool_failure():

    register_tool(
        Tool(
            name="failing",
            function=failing_tool,
            description="Always fails",
            outputs=["answer"],
        )
    )

    step = PlanStep(
        id=1,
        tool="failing",
        tool_input="Hello",
        depends_on=[],
    )

    state = {
        "context": {},
    }

    result = execute_step(
        step,
        state,
        {},
    )

    assert result["result"]["success"] is False
    assert result["result"]["output"] == {}
    assert result["result"]["error"] == "Boom!"

    record = result["record"]

    assert record.step_id == 1
    assert record.tool == "failing"
    assert record.success is False
    assert record.error == "Boom!"


def test_execute_step_resolves_step_references(): 



    register_tool(

        Tool(

            name="echo",

            function=echo_tool,

            description="Echo tool",

            outputs=["received"],

        )

    )



    tool_results = {

        1: {

            "output": {

                "answer": "OpenAI",

            }

        }

    }



    step = PlanStep(

        id=2,

        tool="echo",

        tool_input="Who is #1.answer?",

        depends_on=[1],

    )



    state = {

        "context": {},

    }



    result = execute_step(

        step,

        state,

        tool_results,

    )



    assert (

        result["result"]["output"]["received"]

        == "Who is OpenAI?"

    )



def test_executor_node_empty_plan():



    state = {

        "steps": [],

        "context": {},

        "tool_results": {},

        "execution_records": [],

    }



    result = executor_node(state)



    assert result["tool_results"] == {}

    assert result["execution_records"] == []

    assert result["context"] == {}



def test_executor_node_single_step():



    register_tool(

        Tool(

            name="dummy",

            function=dummy_tool,

            description="Dummy tool",

            outputs=["answer"],

        )

    )



    step = PlanStep(

        id=1,

        tool="dummy",

        tool_input="Hello",

        depends_on=[],

        output="result",

    )



    state = {

        "steps": [step],

        "context": {},

        "tool_results": {},

        "execution_records": [],

    }



    result = executor_node(state)



    assert 1 in result["tool_results"]



    assert (

        result["tool_results"][1]["output"]["answer"]

        == "Orion Test Response"

    )



    assert len(result["execution_records"]) == 1



    record = result["execution_records"][0]



    assert record.step_id == 1

    assert record.success is True



    assert result["context"]["step_1"] == {

        "answer": "Orion Test Response"

    }



    assert result["context"]["result"] == "Orion Test Response"   

def test_executor_node_sequential_steps():

    register_tool(
        Tool(
            name="dummy",
            function=dummy_tool,
            description="Dummy tool",
            outputs=["answer"],
        )
    )

    register_tool(
        Tool(
            name="echo",
            function=echo_tool,
            description="Echo tool",
            outputs=["received"],
        )
    )

    step1 = PlanStep(
        id=1,
        tool="dummy",
        tool_input="Hello",
        depends_on=[],
    )

    step2 = PlanStep(
        id=2,
        tool="echo",
        tool_input="Answer is #1.answer",
        depends_on=[1],
    )

    state = {
        "steps": [step1, step2],
        "context": {},
        "tool_results": {},
        "execution_records": [],
    }

    result = executor_node(state)

    assert len(result["tool_results"]) == 2

    assert (
        result["tool_results"][1]["output"]["answer"]
        == "Orion Test Response"
    )

    assert (
        result["tool_results"][2]["output"]["received"]
        == "Answer is Orion Test Response"
    )

    assert len(result["execution_records"]) == 2


def test_executor_node_parallel_steps():

    register_tool(
        Tool(
            name="dummy",
            function=dummy_tool,
            description="Dummy tool",
            outputs=["answer"],
        )
    )

    step1 = PlanStep(
        id=1,
        tool="dummy",
        tool_input="First",
        depends_on=[],
    )

    step2 = PlanStep(
        id=2,
        tool="dummy",
        tool_input="Second",
        depends_on=[],
    )

    state = {
        "steps": [step1, step2],
        "context": {},
        "tool_results": {},
        "execution_records": [],
    }

    result = executor_node(state)

    assert len(result["tool_results"]) == 2
    assert 1 in result["tool_results"]
    assert 2 in result["tool_results"]

    assert len(result["execution_records"]) == 2


def test_executor_node_skips_completed_steps():

    register_tool(
        Tool(
            name="dummy",
            function=dummy_tool,
            description="Dummy tool",
            outputs=["answer"],
        )
    )

    step1 = PlanStep(
        id=1,
        tool="dummy",
        tool_input="Already done",
        depends_on=[],
    )

    step2 = PlanStep(
        id=2,
        tool="dummy",
        tool_input="Execute me",
        depends_on=[1],
    )

    state = {
        "steps": [step1, step2],
        "context": {},
        "tool_results": {
            1: {
                "output": {
                    "answer": "Orion Test Response"
                }
            }
        },
        "execution_records": [],
    }

    result = executor_node(state)

    assert len(result["tool_results"]) == 2
    assert 2 in result["tool_results"]
    assert len(result["execution_records"]) == 1

def test_executor_node_dependency_not_executed_after_failure():

    register_tool(
        Tool(
            name="fail",
            function=failing_tool,
            description="Always fails",
            outputs=["answer"],
        )
    )

    register_tool(
        Tool(
            name="dummy",
            function=dummy_tool,
            description="Dummy tool",
            outputs=["answer"],
        )
    )

    step1 = PlanStep(
        id=1,
        tool="fail",
        tool_input="Fail",
        depends_on=[],
    )

    step2 = PlanStep(
        id=2,
        tool="dummy",
        tool_input="Should not run",
        depends_on=[1],
    )

    state = {
        "steps": [step1, step2],
        "context": {},
        "tool_results": {},
        "execution_records": [],
    }

    with pytest.raises(ValueError, match="No executable step found"):
        executor_node(state)