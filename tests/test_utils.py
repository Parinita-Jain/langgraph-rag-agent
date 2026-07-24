import pytest

from langchain_core.messages import AIMessage, HumanMessage

from models.plan import PlanStep
from utils import (
    build_conversation,
    can_execute,
    get_ready_steps,
    resolve_context_variables,
    resolve_step_references,
)


def test_build_conversation_empty():

    assert build_conversation([]) == ""


def test_build_conversation():

    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
    ]

    conversation = build_conversation(messages)

    assert "Human: Hello" in conversation
    assert "AI: Hi there!" in conversation


def test_can_execute_true():

    step = PlanStep(
        id=2,
        tool="dummy",
        tool_input="",
        depends_on=[1],
    )

    assert can_execute(step, {1})


def test_can_execute_false():

    step = PlanStep(
        id=2,
        tool="dummy",
        tool_input="",
        depends_on=[1],
    )

    assert not can_execute(step, set())


def test_get_ready_steps():

    steps = [
        PlanStep(1, "dummy", ""),
        PlanStep(2, "dummy", "", depends_on=[1]),
    ]

    ready = get_ready_steps(steps, {1})

    assert len(ready) == 2


def test_get_ready_steps_partial():

    steps = [
        PlanStep(1, "dummy", ""),
        PlanStep(2, "dummy", "", depends_on=[1]),
        PlanStep(3, "dummy", "", depends_on=[2]),
    ]

    ready = get_ready_steps(steps, {1})

    assert len(ready) == 2
    assert ready[0].id == 1
    assert ready[1].id == 2


def test_resolve_step_references():

    tool_results = {
        1: {
            "output": {
                "answer": 42,
            }
        }
    }

    result = resolve_step_references(
        "Result is #1.answer",
        tool_results,
    )

    assert result == "Result is 42"


def test_missing_step_reference():

    with pytest.raises(ValueError):

        resolve_step_references(
            "Result is #5.answer",
            {},
        )


def test_multiple_step_references():

    tool_results = {
        1: {"output": {"x": 10}},
        2: {"output": {"y": 20}},
    }

    result = resolve_step_references(
        "#1.x + #2.y",
        tool_results,
    )

    assert result == "10 + 20"


def test_resolve_context_variables():

    result = resolve_context_variables(
        "x + y",
        {
            "x": 10,
            "y": 20,
        },
    )

    assert result == "10 + 20"


def test_context_variable_missing():

    result = resolve_context_variables(
        "x + z",
        {
            "x": 10,
        },
    )

    assert result == "10 + z"


def test_no_context_variables():

    result = resolve_context_variables(
        "42",
        {},
    )

    assert result == "42"


def test_missing_output_field():

    tool_results = {
        1: {
            "output": {}
        }
    }

    with pytest.raises(
        ValueError,
        match="Output field 'answer' not found for step 1."
    ):
        resolve_step_references(
            "#1.answer",
            tool_results,
        )