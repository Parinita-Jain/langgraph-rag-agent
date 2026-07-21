def completion_node(state):

    tool_results = state["tool_results"]

    done = all(
        result["success"]
        for result in tool_results.values()
    )

    print("\n===== COMPLETION CHECK =====")
    print("Done:", done)

    return {
        "done": done
    }