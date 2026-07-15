from langgraph.graph import StateGraph, START, END

from state import AgentState

from nodes import (
    agent_node,
    planner_node,
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

# Routing
workflow.add_edge(
    "planner",
    "executor"
)
# RAG Path

workflow.add_edge("executor", END)

# Compile Graph
app = workflow.compile() 
