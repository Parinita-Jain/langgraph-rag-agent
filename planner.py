import re
from shared import llm
from schemas import PlannerOutput, PlanStep
from validator import validate_plan
from registry import list_tools,get_tool_descriptions
# ===========================
# Planner
# ===========================

def repair_plan(question, previous_plan, errors):

    print("\n===== REPAIRING PLAN =====")

    plan = ""

    for step in previous_plan:

        plan += f"""
            Step {step.id}

            Tool:
            {step.tool}

            Input:
            {step.tool_input}

            Depends On:
            {step.depends_on}

            --------------------
        """

    tool_descriptions = get_tool_descriptions()
    from registry import list_tools

    print("\nPlanner sees tools:")
    print(list_tools())
    prompt = f"""
    The following execution plan is INVALID.

    Original User Question:

    {question}

    Current Plan:

    {plan}

    Validation Errors:

    {chr(10).join(errors)}

    Fix ALL validation errors.

    Available Tools:

    {tool_descriptions} 

    Rules:

    1. Keep as much of the original plan as possible.

    2. Do NOT change correct steps.

    3. Fix ONLY the invalid parts.

    4. Return ONLY the corrected execution plan.
    Do not include explanations.

    5. Output must match PlannerOutput exactly.

    6. Use ONLY the available tools listed below.

    7. Preserve existing step IDs whenever possible.

    8. Preserve output variables unless they must change.
    """

    structured_llm = llm.with_structured_output(
        PlannerOutput
    )

    return structured_llm.invoke(prompt)

def planner_node(state):

    question = state["messages"][-1].content
    question_lower = question.lower().strip()

    print("\n===== PLANNER NODE =====")
    print("Question:", question)

    # -------------------------
    # Rule 1 : Greetings
    # -------------------------

    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    if question_lower in greetings:

        print("Tool: direct (Python Rule)")

        return {
                "steps": [
                    PlanStep(
                        id=1,
                        tool="direct",
                        tool_input="",
                        depends_on=[]
                    )
                ]
            }


    # -------------------------
    # Rule 2 : Calculator
    # -------------------------

    calculator_pattern = r"^[0-9+\-*/().%\s]+$"

    if re.fullmatch(calculator_pattern, question):

        print("Tool: calculator (Python Rule)")

        return {
                    "steps": [
                        PlanStep(
                            id=1,
                            tool="calculator",
                            tool_input=question,
                            depends_on=[]
                        )
                    ]
                }

    # -------------------------
    # Rule 3 : Ask Gemini
    # -------------------------
    
    tool_descriptions = get_tool_descriptions()

    prompt = f"""
    You are an AI planning assistant.

    Your job is to create an execution plan.

    Available tools:
    IMPORTANT

    The value of the tool field MUST exactly match one
    of the tool names shown above.

    Do NOT invent tool names.

    Do NOT rename tools.

    Examples

    Correct:

    tool = rag
    tool = calculator
    tool = llm
    tool = direct

    Incorrect:

    tool = Retrieval
    tool = Search
    tool = LLM
    tool = Calculator
    tool = Weather

    {tool_descriptions}


    Rules:

    1. You may use ONE OR MORE tools.

    2. If multiple independent tasks are requested,
    Return a workflow plan.

    Each step must contain:

        - id
        - tool
        - tool_input

    Return a list called steps.
    For every step, also return depends_on.

    If the step has no dependencies,
    return an empty list.

    3. Keep the original order of execution.

    4. Generate a precise and self-contained tool_input for every tool.

    Each tool_input should contain only the information needed by that tool.

    Do not include parts of the request that belong to another tool.
    5. If a step depends on the output of a previous step, reference the required output field.

        Use this format:

        #<step_id>.<field>

        Examples:

        #1.value
        #2.answer

        Do NOT use only #1 or #2.

        For calculator outputs use:
        #<step>.value

        For llm/rag/direct outputs use:
        #<step>.answer

        Example 1

        User:
        Calculate 15% of ₹800.
        Then multiply the result by 5.

        Steps:

        1.
        tool = calculator
        tool_input = 0.15 * 800

        2.
        tool = calculator
        tool_input = #1.value * 5
        depends_on = [1]

        Example 2

        User:
        Explain RAG.
        Summarize the explanation.

        Steps:

        1.
        tool = rag
        tool_input = Explain RAG

        2.
        tool = llm
        tool_input = Summarize #1.answer
        depends_on = [1]
    If a later step needs a previous result,
    refer to its output variable name.

    Example

    Step1

    calculator

    0.15 * 800

    output=tax

    Step2

    calculator

    salary-tax

    depends_on=[1]


    Question:

    {question}
"""


    structured_llm = llm.with_structured_output(PlannerOutput)
    try:
        result = structured_llm.invoke(prompt)
    except Exception as e:
        print("\nPlanner failed:")
        raise RuntimeError(f"Planner LLM failed: {e}") from e
    

    print(result)
    print(result.steps)

    MAX_REPAIR_ATTEMPTS = 3

    for attempt in range(MAX_REPAIR_ATTEMPTS):
        print("\nBefore validation:")
        print(result.steps)
        errors = validate_plan(result.steps)
        print("\nValidation errors:")
        print(errors)
        if not errors:
            break
        print("\nPlanner after repair:")
        print(result.steps)
        repaired = repair_plan(
            question,
            result.steps,
            errors
        )
        repair_errors = validate_plan(repaired.steps)

        if repair_errors:
            print("Repair produced invalid plan.")
        else:
            result = repaired
        print("\nAfter repair:")
        print(result.steps)
    else:
        
        print("\nRegistered tools:")
        print(list_tools())
        errors = validate_plan(result.steps)

        if errors:

            raise ValueError(
                "Planner could not repair the plan.\n\n"
                + "\n".join(errors)
            )
    print("\n===== EXECUTION PLAN =====")

    for step in result.steps:
        print(
            f"Step {step.id}: "
            f"{step.tool} "
            f"(input='{step.tool_input}', "
            f"depends_on={step.depends_on})"
        )
    print("\nReturning steps:")
    print(result.steps)
    return {
        "steps": result.steps
    }

 