from langgraph.graph import StateGraph, START, END

from state import AgentState

from nodes import (
    agent_node,
    planner_node,
    executor_node
)
from completion import completion_node
from replanner import replanner_node
from synthesizer import synthesizer_node
# Create Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent_node)

workflow.add_node("planner", planner_node)

workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("replanner",replanner_node)
workflow.add_node("completion",completion_node)
def choose_after_replan(state):

    if state["done"]:
        return "synthesizer"

    return "executor"

def choose_next(state):

    if state["done"]:
        return "done"

    return "continue"

# Start Flow
workflow.add_edge(START, "agent")

# Agent → Planner
workflow.add_edge("agent", "planner")

# Routing
workflow.add_edge(
    "planner",
    "executor"
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
    choose_after_replan,
    {
        "executor": "executor",
        "synthesizer": "synthesizer"
    }
)
workflow.add_edge("synthesizer", END)

# Compile Graph
app = workflow.compile() 
