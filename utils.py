import re

from langchain_core.messages import AIMessage, HumanMessage

def build_conversation(messages):

    conversation = ""

    for message in messages:

        if isinstance(message, HumanMessage):
            conversation += f"Human: {message.content}\n\n"

        elif isinstance(message, AIMessage):
            conversation += f"AI: {message.content}\n\n"

    return conversation

def can_execute(step, completed_steps):

    return all(
        dependency in completed_steps
        for dependency in step.depends_on
    )

def get_ready_steps(pending_steps, completed_steps):

    ready_steps = []

    for step in pending_steps:

        if can_execute(step, completed_steps):

            ready_steps.append(step)

    return ready_steps



def resolve_step_references(tool_input, tool_results):

    pattern = r"#(\d+)\.(\w+)"

    matches = re.findall(pattern, tool_input)

    for step_id, field in matches:

        step_id = int(step_id)

        if step_id not in tool_results:
            raise ValueError(
                f"Step {step_id} has not been executed yet."
            )

        
        output = tool_results[step_id]["output"]

        if field not in output:
            raise ValueError(
                f"Output field '{field}' not found for step {step_id}."
            )

        result = output[field]
        tool_input = tool_input.replace(
            f"#{step_id}.{field}",
            str(result)
        )

    return tool_input

def resolve_context_variables(tool_input, context):

    pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"

    variables = re.findall(pattern, tool_input)

    for variable in variables:

        if variable in context:

            tool_input = tool_input.replace(
                variable,
                str(context[variable])
            )

    return tool_input

