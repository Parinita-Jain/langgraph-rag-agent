from langgraph.graph import StateGraph, START, END

from state import AgentState
print(AgentState)
print(AgentState.__annotations__)
print("AgentState:", AgentState.__annotations__)
from nodes import (
    agent_node,
    planner_node,
    executor_node
)
from completion import completion_node
from replanner import replanner_node
from synthesizer import synthesizer_node
from error_handler import error_handler_node
# Create Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent_node)

workflow.add_node("planner", planner_node)

workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("replanner",replanner_node)
workflow.add_node("completion",completion_node)
workflow.add_node(
    "error_handler",
    error_handler_node
)
def choose_after_replan(state):

    if state["done"]:
        return "synthesizer"

    return "executor"

def choose_next(state):

    if state["done"]:
        return "done"

    return "continue"

def choose_after_planner(state):

    print("\n===== ROUTER STATE =====")
    print(state)

    if state.get("error"):
        return "error"

    return "executor"


def choose_after_replanner(state):

    if state.get("error"):
        return "error"

    if state.get("done"):
        return "synthesizer"

    return "executor"

def choose_after_synthesizer(state):

    if state.get("error"):
        return "error"

    return "done"
# Start Flow
workflow.add_edge(START, "agent")

# Agent → Planner
workflow.add_edge("agent", "planner")

# Routing
workflow.add_conditional_edges(
    "planner",
    choose_after_planner,
    {
        "executor": "executor",
        "error": "error_handler"
    }
)
# RAG Path
workflow.add_edge(
    "executor",
    "replanner"
)
workflow.add_conditional_edges(
    "completion",
    choose_next,
    {
        "done": "synthesizer",
        "continue": "replanner"
    }
)
workflow.add_conditional_edges(
    "replanner",
    choose_after_replanner,
    {
        "executor": "executor",
        "synthesizer": "synthesizer",
        "error": "error_handler"
    }
)

workflow.add_conditional_edges(
    "synthesizer",
    choose_after_synthesizer,
    {
        "done": END,
        "error": "error_handler",
    },
)


workflow.add_edge(
    "error_handler",
    END
)

# Compile Graph
app = workflow.compile() 
