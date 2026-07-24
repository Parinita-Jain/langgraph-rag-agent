import pytest

from models.plan import PlanStep
from registry import Tool, clear_registry, register_tool
from validator import validate_plan


def dummy_tool(state):
    return {
        "success": True,
        "output": {},
    }


@pytest.fixture(autouse=True)
def clean_registry():
    clear_registry()

    register_tool(
        Tool(
            name="dummy",
            function=dummy_tool,
            description="Dummy tool",
            outputs=["result"],
        )
    )

    yield

    clear_registry()


def test_empty_plan():

    errors = validate_plan([])

    assert errors == ["Execution plan is empty."]


def test_valid_plan():

    steps = [
        PlanStep(
            id=1,
            tool="dummy",
            tool_input="",
        ),
        PlanStep(
            id=2,
            tool="dummy",
            tool_input="",
            depends_on=[1],
        ),
    ]

    assert validate_plan(steps) == []


def test_duplicate_step_id():

    steps = [
        PlanStep(1, "dummy", ""),
        PlanStep(1, "dummy", ""),
    ]

    errors = validate_plan(steps)

    assert "Duplicate step id: 1" in errors


def test_unknown_tool():

    steps = [
        PlanStep(1, "unknown", ""),
    ]

    errors = validate_plan(steps)

    assert "Unknown tool: unknown" in errors


def test_missing_dependency():

    steps = [
        PlanStep(
            id=1,
            tool="dummy",
            tool_input="",
            depends_on=[2],
        ),
    ]

    errors = validate_plan(steps)

    assert "Step 1 depends on missing step 2" in errors


def test_self_dependency():

    steps = [
        PlanStep(
            id=1,
            tool="dummy",
            tool_input="",
            depends_on=[1],
        ),
    ]

    errors = validate_plan(steps)

    assert "Step 1 depends on itself." in errors


def test_future_dependency():

    steps = [
        PlanStep(
            id=1,
            tool="dummy",
            tool_input="",
            depends_on=[2],
        ),
        PlanStep(
            id=2,
            tool="dummy",
            tool_input="",
        ),
    ]

    errors = validate_plan(steps)

    assert "Step 1 depends on future step 2" in errors


def test_non_sequential_step_ids():

    steps = [
        PlanStep(1, "dummy", ""),
        PlanStep(3, "dummy", ""),
    ]

    errors = validate_plan(steps)

    assert "Expected step id 2, found 3" in errors