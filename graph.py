from langgraph.graph import StateGraph, START, END

from state import AgentState

from nodes import (
    agent_node,
    router_node,
    choose_route,
    direct_node,
    retrieve_node,
    generate_node,
    rewrite_node
)

# Create Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent_node)

workflow.add_node("router", router_node)

workflow.add_node("direct", direct_node)

workflow.add_node("retrieve", retrieve_node)

workflow.add_node("rewrite", rewrite_node)

workflow.add_node("generate", generate_node)

# Start Flow
workflow.add_edge(START, "agent")

# Agent → Router
workflow.add_edge("agent", "router")

# Conditional Routing
workflow.add_conditional_edges(
    "router",
    choose_route,
    {
        "direct": "direct",
        "retrieve": "rewrite"
    }
)

# RAG Path

workflow.add_edge("rewrite", "retrieve")

workflow.add_edge("retrieve", "generate")

# End Paths
workflow.add_edge("generate", END)

workflow.add_edge("direct", END)

# Compile Graph
app = workflow.compile() 
