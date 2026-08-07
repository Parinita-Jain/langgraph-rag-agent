import re


def evaluate_condition(condition, tool_results):

    if condition is None:
        return True

    pattern = (
        r"#(\d+)\.(\w+)\s*"
        r"(==|!=|>=|<=|>|<)\s*"
        r"(.+)"
    )

    match = re.fullmatch(pattern, condition.strip())

    if not match:
        raise ValueError(
            f"Invalid condition: {condition}"
        )

    step_id = int(match.group(1))
    field = match.group(2)
    operator = match.group(3)
    expected = match.group(4).strip()

    output = tool_results[step_id]["output"]

    actual = output[field]

    if expected.startswith(("'", '"')):
        expected = expected[1:-1]
    elif expected.lower() == "true":
        expected = True
    elif expected.lower() == "false":
        expected = False
    else:
        try:
            expected = int(expected)
        except ValueError:
            try:
                expected = float(expected)
            except ValueError:
                pass

    return {
        "==": actual == expected,
        "!=": actual != expected,
        ">": actual > expected,
        "<": actual < expected,
        ">=": actual >= expected,
        "<=": actual <= expected,
    }[operator]