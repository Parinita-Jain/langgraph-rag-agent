from langgraph.graph import StateGraph, START, END

from state import AgentState

from nodes import (
    agent_node,
    planner_node,
    choose_next_node,
    executor_node
)

# Create Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent_node)

workflow.add_node("planner", planner_node)

workflow.add_node("executor", executor_node)

# Start Flow
workflow.add_edge(START, "agent")

# Agent → Planner
workflow.add_edge("agent", "planner")

# Conditional Routing
workflow.add_conditional_edges(
    "planner",
    choose_next_node,
    {
        "direct": "executor",
        "calculator": "executor",
        "rag": "executor"
    }
)
# RAG Path

workflow.add_edge("executor", END)

# Compile Graph
app = workflow.compile() 
