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
from step_status import StepStatus

from runtime.event_bus import EventBus
from runtime.event_types import WorkflowEventType
from runtime.failure_reason import FailureReason

class FakeListener:
    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append(event)

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
                "messages": [],
                "output": {
                    "answer": "Orion Test Response"
                },
                "success": True,
                "error": None,
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

    result = executor_node(state)

    assert result["tool_results"][1]["success"] is False

    assert result["tool_results"][2]["status"] == StepStatus.SKIPPED

def test_execution_summary_present():
    register_tool(
        Tool(
            name="direct",
            function=dummy_tool,
            description="Direct tool",
            outputs=["answer"],
        )
    )
    state = {
        "steps": [
            PlanStep(
                id=1,
                tool="direct",
                tool_input="Hello",
                depends_on=[],
            )
        ],
        "context": {},
    }

    result = executor_node(state)

    summary = result["execution_summary"]

    assert summary.total_steps == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.skipped == 0

def test_executor_emits_success_events():

    register_tool(
        Tool(
            name="dummy",
            function=dummy_tool,
            description="Dummy tool",
            outputs=["answer"],
        )
    )

    listener = FakeListener()

    bus = EventBus()
    bus.subscribe(listener)

    step = PlanStep(
        id=1,
        tool="dummy",
        tool_input="Hello",
        depends_on=[],
    )

    state = {
        "steps": [step],
        "context": {},
        "tool_results": {},
        "execution_records": [],
        "event_bus": bus,
    }

    executor_node(state)

    event_types = [
        event.type
        for event in listener.events
    ]

    assert event_types == [
        WorkflowEventType.WORKFLOW_STARTED,
        WorkflowEventType.STEP_STARTED,
        WorkflowEventType.STEP_COMPLETED,
        WorkflowEventType.WORKFLOW_COMPLETED,
    ]

def test_executor_emits_failed_event():

    register_tool(
        Tool(
            name="fail",
            function=failing_tool,
            description="Always fails",
            outputs=["answer"],
        )
    )

    listener = FakeListener()

    bus = EventBus()
    bus.subscribe(listener)

    step = PlanStep(
        id=1,
        tool="fail",
        tool_input="Hello",
        depends_on=[],
    )

    state = {
        "steps": [step],
        "context": {},
        "tool_results": {},
        "execution_records": [],
        "event_bus": bus,
    }

    executor_node(state)

    event_types = [
        event.type
        for event in listener.events
    ]

    assert event_types == [
        WorkflowEventType.WORKFLOW_STARTED,
        WorkflowEventType.STEP_STARTED,
        WorkflowEventType.STEP_FAILED,
        WorkflowEventType.WORKFLOW_COMPLETED,
    ]

    failed_event = listener.events[2]

    assert failed_event.payload["reason"] == FailureReason.EXCEPTION

def test_executor_emits_skipped_event():

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

    listener = FakeListener()

    bus = EventBus()
    bus.subscribe(listener)

    state = {
        "steps": [
            PlanStep(
                id=1,
                tool="fail",
                tool_input="Fail",
                depends_on=[],
            ),
            PlanStep(
                id=2,
                tool="dummy",
                tool_input="Should not execute",
                depends_on=[1],
            ),
        ],
        "context": {},
        "tool_results": {},
        "execution_records": [],
        "event_bus": bus,
    }

    executor_node(state)

    event_types = [
        event.type
        for event in listener.events
    ]

    assert WorkflowEventType.STEP_SKIPPED in event_types

import time

from planner import PlanStep
from registry import register_tool
from executor import executor_node
from step_status import StepStatus


def slow_tool(state):
    time.sleep(2)

    return {
        "messages": [],
        "output": {
            "value": 123,
        },
        "success": True,
    }


def test_executor_timeout():

    register_tool(
            Tool(
                name="slow_tool",
                function=slow_tool,
                description="Slow tool",
                outputs=["value"],
                timeout=0.1,
            )
        )

    state = {
        "steps": [
            PlanStep(
                id=1,
                tool="slow_tool",
                tool_input="",
                depends_on=[],
                output="result",
            )
        ],
        "context": {},
    }

    result = executor_node(state)

    tool_result = result["tool_results"][1]
    
    assert tool_result["success"] is False
    assert tool_result["status"] == StepStatus.FAILED
    assert "timed out" in tool_result["error"].lower() \
        or "exceeded timeout" in tool_result["error"].lower()

def test_executor_emits_timeout_reason():

    register_tool(
        Tool(
            name="slow_tool",
            function=slow_tool,
            description="Slow tool",
            outputs=["value"],
            timeout=0.1,
        )
    )

    listener = FakeListener()

    bus = EventBus()
    bus.subscribe(listener)

    state = {
        "steps": [
            PlanStep(
                id=1,
                tool="slow_tool",
                tool_input="",
                depends_on=[],
            )
        ],
        "context": {},
        "event_bus": bus,
    }

    executor_node(state)

    failed_event = listener.events[2]

    assert failed_event.type == WorkflowEventType.STEP_FAILED
    assert failed_event.payload["reason"] == FailureReason.TIMEOUT

def fast_tool(state):

    return {
        "messages": [],
        "output": {
            "value": 123,
        },
        "success": True,
    }


def test_executor_timeout_success():

    register_tool(
        Tool(
            name="fast_tool",
            function=fast_tool,
            description="Fast tool",
            outputs=["value"],
            timeout=5,
        )
    )

    state = {
        "steps": [
            PlanStep(
                id=1,
                tool="fast_tool",
                tool_input="",
                depends_on=[],
                output="result",
            )
        ],
        "context": {},
    }

    result = executor_node(state)

    tool_result = result["tool_results"][1]

    assert tool_result["success"] is True
    assert tool_result["status"] == StepStatus.SUCCESS

def test_timeout_creates_execution_record():

    register_tool(
        Tool(
            name="slow_tool_record",
            function=slow_tool,
            description="Slow tool",
            outputs=["value"],
            timeout=0.1,
        )
    )

    state = {
        "steps": [
            PlanStep(
                id=1,
                tool="slow_tool_record",
                tool_input="",
                depends_on=[],
                output="result",
            )
        ],
        "context": {},
    }

    result = executor_node(state)

    record = result["execution_records"][0]

    assert record.success is False
    assert record.duration > 0
    assert "timed out" in record.error.lower()