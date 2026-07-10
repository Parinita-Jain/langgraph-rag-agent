from langgraph.graph import StateGraph, START, END

from state import AgentState

from nodes import (
    agent_node,
    router_node,
    choose_route,
    executor_node
)

# Create Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent_node)

workflow.add_node("router", router_node)

workflow.add_node("executor", executor_node)

# Start Flow
workflow.add_edge(START, "agent")

# Agent → Router
workflow.add_edge("agent", "router")

# Conditional Routing
workflow.add_conditional_edges(
    "router",
    choose_route,
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
