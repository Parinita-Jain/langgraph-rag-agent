import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import AIMessage

from execution import ExecutionRecord
from registry import get_tool
from synthesizer import synthesize_response
from utils import (
    get_ready_steps,
    resolve_context_variables,
    resolve_step_references,
)


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

    print("Original Input:", step.tool_input)
    print("Resolved Input:", tool_input)

    tool_state = {
        **state,
        "tool_input": tool_input,
    }

    tool = get_tool(tool_name)

    if tool is None:
        raise ValueError(f"Unknown tool: {tool_name}")

    max_retries = tool.retries

    # TODO: Implement timeout handling.
    # timeout = tool.timeout

    tool_function = tool.function

    if tool_function is None:
        raise ValueError(f"Unknown tool: {tool_name}")

    last_exception = None

    start_time = time.perf_counter()

    for attempt in range(max_retries + 1):

        try:

            result = tool_function(tool_state)
            success = result.get("success", True)
            end_time = time.perf_counter()

            duration = end_time - start_time
            record = ExecutionRecord(
                step_id=step.id,
                tool=tool_name,
                success=success,
                retries=attempt,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                error=result.get("error"),
)

            if attempt > 0:
                print(
                    f"{tool_name} succeeded "
                    f"after {attempt} retries."
                )

            return {
                "result": result,
                "record": record,
            }

        except Exception as e:

            last_exception = e

            print(
                f"Attempt {attempt + 1}/{max_retries + 1} "
                f"failed for {tool_name}: {e}"
            )

            if attempt < max_retries:

                wait_time = 2 ** attempt

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

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
        error=str(last_exception),
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
            "error": str(last_exception),
        },
        "record": record,
    }


def executor_node(state):

    print("\n===== EXECUTOR NODE =====")

    execution_records = state.get(
        "execution_records",
        [],
    )

    tool_results = state.get(
        "tool_results",
        {},
    )

    completed_steps = set(tool_results.keys())

    state["context"] = state.get(
        "context",
        {},
    )
    print("\nSTATE STEPS")
    print(state["steps"])
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

            raise ValueError(
                f"No executable step found.\n"
                f"Completed: {completed_steps}\n"
                f"Pending: {[step.id for step in pending_steps]}"
            )

        print("\n===== READY STEPS =====")

        for step in ready_steps:

            print(
                f"Step {step.id}: {step.tool}"
            )

        with ThreadPoolExecutor() as executor:

            futures = {}

            for step in ready_steps:

                print(
                    f"\nSubmitting Step {step.id} "
                    f"(depends_on={step.depends_on})"
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

                    print(f"\nStep {step.id} failed.")

                    print(e)

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

                print(f"Completed Step {step.id}")

                print("\n===== CONTEXT =====")

                print(state["context"])

        print("\nCompleted:", completed_steps)

    state["tool_results"] = tool_results
    state["execution_records"] = execution_records

    print("\n===== TOOL RESULTS =====")
    print(tool_results)

    print("\n===== EXECUTION SUMMARY =====")

    total_time = 0.0

    for record in execution_records:

        total_time += record.duration

        print(
            f"Step {record.step_id:<2} | "
            f"{record.tool:<12} | "
            f"{record.duration:7.3f}s | "
            f"Retries: {record.retries} | "
            f"{'Success' if record.success else 'Failed'}"
        )

    print("-" * 60)
    print(f"Total execution time: {total_time:.3f}s")

    return {
    "tool_results": tool_results,
    "execution_records": execution_records,
    "context": state["context"],
    }