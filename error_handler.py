from langchain_core.messages import AIMessage

from config import logger
from errors import ErrorType


def error_handler_node(state):

    logger.info("Error handler invoked")
    logger.debug("Error handler state: %s", state)

    error = state.get("error")

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

    if error:
        logger.error(
            "OrionError | Type=%s | Source=%s | Recoverable=%s | %s",
            error.error_type.value,
            error.source,
            error.recoverable,
            error.message,
        )

        if error.original_exception:
            logger.exception(
                "Original exception",
                exc_info=error.original_exception,
            )

    return {
        "messages": [AIMessage(content=message)],
        "done": True,
    }