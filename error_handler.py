from langchain_core.messages import AIMessage

from errors import ErrorType


def error_handler_node(state):

    print("\n===== ERROR HANDLER STATE =====")
    print(state)

    error = state.get("error")

    print("\n===== ERROR HANDLER =====")

    if error is None:
        message = "An unexpected error occurred."

    elif error.error_type == ErrorType.INFRASTRUCTURE:
        message = (
            "The AI service is temporarily unavailable. "
            "Please try again in a few minutes."
        )

    elif error.error_type == ErrorType.TOOL:
        message = (
            "A tool failed while processing your request."
        )

    elif error.error_type == ErrorType.PLANNER:
        message = error.message

    else:
        message = error.message

    # Log full error for debugging
    if error:
        print(error)

    return {
        "messages": [AIMessage(content=message)],
        "done": True,
    }