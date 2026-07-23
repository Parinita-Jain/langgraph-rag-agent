import re
from shared import llm
from schemas import PlannerOutput, PlanStep
from validator import validate_plan
from registry import list_tools,get_tool_descriptions
from errors import OrionError, ErrorType
from langgraph.types import Command
from config import logger
# ===========================
# Planner
# ===========================

def repair_plan(question, previous_plan, errors):

    logger.info("Repairing invalid execution plan")

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
    logger.debug("Available tools: %s", list_tools())
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

    logger.info("Planning started")
    logger.debug("User question: %s", question)

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

        logger.info("Planner selected 'direct' tool using greeting rule")

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

        logger.info("Planner selected 'calculator' tool using calculator rule")

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

        logger.exception("Planner invocation failed")

        planner_error = OrionError(
            source="planner",
            error_type=ErrorType.INFRASTRUCTURE,
            message=str(e),
            recoverable=True,
            original_exception=e
        )
        
        
        
        return {
            "steps": [],
            "error": planner_error
        }
        

    logger.debug("Planner output: %s", result)

    MAX_REPAIR_ATTEMPTS = 3

    for attempt in range(MAX_REPAIR_ATTEMPTS):
        
        errors = validate_plan(result.steps)
        logger.debug("Validation attempt %d", attempt + 1)
        
        if not errors:
            break

        logger.warning(
                            "Planner validation failed with %d errors",
                            len(errors),
                        )
        repaired = repair_plan(
            question,
            result.steps,
            errors
        )
        repair_errors = validate_plan(repaired.steps)

        if repair_errors:
            logger.warning("Planner repair produced an invalid plan")
        else:
            result = repaired
            logger.info("Planner repaired execution plan")
        
    else:
        
        logger.debug("Registered tools: %s", list_tools())
        errors = validate_plan(result.steps)

        if errors:

            logger.error(
                "Planner failed after %d repair attempts",
                MAX_REPAIR_ATTEMPTS,
            )

            raise ValueError(
                "Planner could not repair the plan.\n\n"
                + "\n".join(errors)
            )
    logger.info(
        "Planner generated %d execution steps",
        len(result.steps),
    )
    logger.debug("Execution plan: %s", result.steps)

    
    return {
        "steps": result.steps,
        "error":None
    }

#-------------
