import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import AIMessage

from execution import ExecutionRecord
from registry import get_tool

from utils import (
    get_ready_steps,
    resolve_context_variables,
    resolve_step_references,
)
from config import logger
from runtime.retry import execute_with_retry


def execute_step(step, state, tool_results):

    tool_name = step.tool

    tool_input = resolve_step_references(
        step.tool_input,
        tool_results,
    )

    tool_input = resolve_context_variables(
        tool_input,
        state["context"],
    )

    logger.debug("Original input: %s", step.tool_input)
    logger.debug("Resolved input: %s", tool_input)

    tool_state = {
        **state,
        "tool_input": tool_input,
    }

    tool = get_tool(tool_name)

    if tool is None:
        raise ValueError(
        f"Tool '{tool_name}' is not registered."
    )

    max_retries = tool.retries

    # TODO: Implement timeout handling.
    # timeout = tool.timeout

    tool_function = tool.function

    if tool_function is None:
        raise ValueError(
                f"Tool '{tool_name}' has no registered function."
            )

    start_time = time.perf_counter()

    try:

        result = execute_with_retry(
            tool_function,
            tool_state,
            tool_name=tool_name,
            max_retries=max_retries,
        )

        if result["success"]:
            logger.info(
                "Tool '%s' completed successfully.",
                tool_name,
            )
        else:
            logger.warning(
                "Tool '%s' completed with failure: %s",
                tool_name,
                result["error"],
            )

        end_time = time.perf_counter()

        duration = end_time - start_time

        record = ExecutionRecord(
            step_id=step.id,
            tool=tool_name,
            success=result.get("success", True),
            retries=0,          # we'll improve this later
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            error=result.get("error"),
        )

        return {
            "result": result,
            "record": record,
        }
        

    except Exception as e:

        end_time = time.perf_counter()

        duration = end_time - start_time

        record = ExecutionRecord(
            step_id=step.id,
            tool=tool_name,
            success=False,
            retries=max_retries,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            error=str(e),
        )

        return {
            "result": {
                "messages": [
                    AIMessage(
                        content="Tool execution failed."
                    )
                ],
                "output": {},
                "success": False,
                "error": str(e),
            },
            "record": record,
        }

def failed_dependencies(step, tool_results):
    """
    Returns a list of failed dependency IDs.
    """

    return [
        dep
        for dep in step.depends_on
        if dep in tool_results
        and not tool_results[dep]["success"]
    ]

def executor_node(state):

    logger.info("Executor started")

    execution_records = state.get(
        "execution_records",
        [],
    )

    tool_results = state.get(
        "tool_results",
        {},
    )

    completed_steps = set(tool_results)

    state["context"] = state.get(
        "context",
        {},
    )
    
    logger.debug("Execution steps: %s", state["steps"])
    pending_steps = [
        step
        for step in state["steps"]
        if step.id not in completed_steps
    ]

    while pending_steps:

        ready_steps = get_ready_steps(
            pending_steps,
            completed_steps,
        )

        if not ready_steps:

            blocked_steps = [
                step
                for step in pending_steps
                if failed_dependencies(step, tool_results)
            ]

            if blocked_steps:

                logger.info(
                    "Skipping %d blocked step(s).",
                    len(blocked_steps),
                )

                for step in blocked_steps:

                    failed = failed_dependencies(
                            step,
                            tool_results,
                        )

                    tool_results[step.id] = {
                        "messages": [
                            AIMessage(
                                content=(
                                    f"Step {step.id} skipped because "
                                    f"dependencies {failed} failed."
                                )
                            )
                        ],
                        "output": {},
                        "success": False,
                        "status": "SKIPPED",
                        "error": (
                                f"Skipped because dependencies "
                                f"{failed} failed."
                            ),
                    }

                    execution_records.append(
                        ExecutionRecord(
                            step_id=step.id,
                            tool=step.tool,
                            success=False,
                            retries=0,
                            start_time=0,
                            end_time=0,
                            duration=0,
                            error=(
                                f"Skipped because dependencies "
                                f"{failed} failed."
                            ),
                        )
                    )

                    pending_steps.remove(step)

                continue

            raise ValueError(
                f"No executable step found.\n"
                f"Completed: {completed_steps}\n"
                f"Pending: {[step.id for step in pending_steps]}"
            )

        
        logger.info(
                    "Executing %d ready step(s)",
                    len(ready_steps),
                    )
        logger.debug("Ready steps: %s", ready_steps)
                   

        with ThreadPoolExecutor() as executor:

            futures = {}

            for step in ready_steps:

                logger.info(
                "Submitting step %d (%s)",
                step.id,
                step.tool,
            )

                future = executor.submit(
                    execute_step,
                    step,
                    state,
                    tool_results,
                )

                futures[future] = step

            for future in as_completed(futures):

                step = futures[future]

                try:

                    execution = future.result()

                    result = execution["result"]

                    record = execution["record"]

                except Exception as e:

                    logger.exception(
                        "Step %d execution failed",
                        step.id,
                    )

                    result = {
                        "messages": [
                            AIMessage(
                                content=f"Step {step.id} failed."
                            )
                        ],
                        "output": {},
                        "success": False,
                        "error": str(e),
                    }

                    record = ExecutionRecord(
                        step_id=step.id,
                        tool=step.tool,
                        success=False,
                        retries=0,
                        start_time=0,
                        end_time=0,
                        duration=0,
                        error=str(e),
                    )

                execution_records.append(record)

                tool_results[step.id] = result

                if result["success"]:

                    output = result["output"]

                    state["context"][
                        f"step_{step.id}"
                    ] = output

                    if step.output:

                        first_value = next(
                            iter(output.values())
                        )

                        state["context"][
                            step.output
                        ] = first_value

                
                    completed_steps.add(step.id)

                if step in pending_steps:
                    pending_steps.remove(step)

                logger.info(
                    "Completed step %d",
                    step.id,
                )

                logger.debug(
                    "Context updated after step %d: %s",
                    step.id,
                    list(state["context"].keys()),
                )

        

    state["tool_results"] = tool_results
    state["execution_records"] = execution_records

    for record in execution_records:

        logger.debug(
            "Step %d | %s | %.3fs | Retries=%d | %s",
            record.step_id,
            record.tool,
            record.duration,
            record.retries,
            "Success" if record.success else "Failed",
        )

    total_time = sum(
        record.duration
        for record in execution_records
    )
    
    logger.info(
        "Total execution time: %.3fs",
        total_time,
               )

    return {
        "tool_results": tool_results,
        "execution_records": execution_records,
        "context": state["context"],
    }