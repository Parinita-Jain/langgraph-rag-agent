from utils import build_conversation
from shared import llm
from langchain_core.messages import AIMessage
from errors import OrionError, ErrorType
# ===========================
# Synthesizer
# ===========================

def synthesizer_node(state):

    tool_results = state.get("tool_results", {})

    return synthesize_response(
        state,
        tool_results
    )

def synthesize_response(state, tool_results):

    print("\n===== SYNTHESIZER =====")

    conversation = build_conversation(state["messages"])

    results = ""

    for step in state["steps"]:

        result = tool_results[step.id]

        output = result["output"]

        results += f"""
            Step {step.id} Output:

            {output}

            --------------------------------
            """

    prompt = f"""
            You are a helpful AI assistant.

            The tools have already solved the problem.

            Do NOT call any tools.

            Combine the tool results into one natural response.

            Conversation:

            {conversation}

            Tool Results:

            {results}
            """

    
    try:
        response = llm.invoke(prompt)

    except Exception as e:

        return {
            "error": OrionError(
                source="synthesizer",
                error_type=ErrorType.INFRASTRUCTURE,
                message=str(e),
                recoverable=True,
                original_exception=e,
            ),
            "done": True,
        }

    return {

        "messages":[

            AIMessage(content=response.content)

        ]

    }