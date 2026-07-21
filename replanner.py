from schemas import ReplannerOutput
from shared import llm

def replanner_node(state):

    print("\n===== REPLANNER NODE =====")

    question = state["messages"][-1].content

    results = ""

    for step in state["steps"]:

        result = state["tool_results"][step.id]["output"]

        results += f"""
        Step {step.id}

        Tool: {step.tool}

        Output:
        {result}

        -------------------------
        """
    prompt = f"""
    You are an AI Replanner.

    Your job is to inspect the work completed so far.

    Original User Request:

    {question}

    Completed Steps:

    {results}

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
    """
    structured_llm = llm.with_structured_output(
        ReplannerOutput
    )

    result = structured_llm.invoke(prompt)

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
    "steps": state["steps"] + result.steps
    }