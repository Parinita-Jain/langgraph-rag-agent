from schemas import ReplannerOutput
from shared import llm
from registry import get_tool_descriptions
from errors import OrionError, ErrorType

def replanner_node(state):

    print("\n===== REPLANNER NODE =====")

    MAX_REPLANS = 3

    iteration = state.get("iteration", 0)

    if iteration >= MAX_REPLANS:

        return {
            "error": OrionError(
                source="replanner",
                error_type=ErrorType.PLANNER,
                message="Maximum replanning attempts exceeded.",
                recoverable=False
            ),
            "done": True,
            "iteration": iteration
        }

    question = state["messages"][-1].content

    results = ""

    for step in state["steps"]:

        tool_result = state["tool_results"][step.id]

        results += f"""
        Step {step.id}

        Tool:
        {step.tool}

        Success:
        {tool_result["success"]}

        Output:
        {tool_result["output"]}

        Error:
        {tool_result["error"]}

        -------------------------
        """
    tool_descriptions = get_tool_descriptions()
    prompt = f"""
    You are an AI Replanner.

    Your job is to inspect the work completed so far.

    Original User Request:

    {question}

    Completed Steps:

    {results}

    Available tools:

    {tool_descriptions}

    IMPORTANT

    The value of the tool field MUST exactly match one of the tool names shown above.

    Do NOT invent tool names.

    Do NOT rename tools.

    Examples

    Correct:

    tool = rag
    tool = calculator
    tool = llm
    tool = direct

    Incorrect:

    tool = search
    tool = Search
    tool = Calculator
    tool = Retrieval
    tool = Weather

    Determine whether the user's request has been fully satisfied.

    Rules:

    1. If ALL parts of the user's request have been completed:

        done = true

        steps = []

    2. If more work is required:

        done = false

        Return ONLY the additional steps required.

    3. Do NOT recreate completed steps.

    4. New step ids must continue from the previous highest id.

    5. If a new step depends on a previous step,
    use depends_on.

    6. If a previous output is required,
    reference it using

    #<step>.<field>

    Examples:

    #1.value
    #2.answer

    7. ONLY use the tools listed above.

    8. If an existing tool can complete the task, use it.

    9. Never create a tool that is not registered.

    10. Do not repeat failed steps unless the failure is marked recoverable.
    """
    structured_llm = llm.with_structured_output(
        ReplannerOutput
    )

    try:
        result = structured_llm.invoke(prompt)

    except Exception as e:

        return {
            "error": OrionError(
                source="replanner",
                error_type=ErrorType.INFRASTRUCTURE,
                message=str(e),
                recoverable=True,
                original_exception=e,
            ),
            "done": True,
            "iteration": iteration,
        }

    print("Done:", result.done)

    if not result.done:

        print("\n===== NEW STEPS =====")

        for step in result.steps:

            print(
                f"Step {step.id}: "
                f"{step.tool} "
                f"(depends_on={step.depends_on})"
            )
    return {
    "done": result.done,
    "steps": state["steps"] + result.steps,
    "iteration": iteration + 1,
    "error": None,
    }