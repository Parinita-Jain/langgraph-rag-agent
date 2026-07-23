# Orion - LangGraph Agent Framework

Orion is a modular Agentic AI framework built with LangGraph. It combines planning, tool execution, replanning, and synthesis to solve multi-step tasks using specialized tools such as RAG, LLM reasoning, and calculators.

---

## Features

### Agent Framework
- Multi-step planning using Gemini
- Tool registry
- Planner → Executor → Replanner → Synthesizer workflow
- Conditional graph execution
- Centralized AgentState
- Execution tracking
- Structured planning using Pydantic

### Tooling
- Calculator Tool
- RAG Tool
- General LLM Tool
- Direct Conversation Tool

### RAG
- PDF ingestion
- ChromaDB vector database
- HuggingFace Embeddings
- Semantic retrieval
- Retrieval-Augmented Generation (RAG)

### Reliability
- Centralized error model (`OrionError`)
- Error handler node
- Plan validation
- Dependency resolution between steps
- Execution records

---

# Architecture

```
                    START
                      │
                      ▼
                  Agent Node
                      │
                      ▼
                 Planner Node
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     Error Handler           Executor
                                  │
                                  ▼
                            Replanner
                     ┌────────┴────────┐
                     │                 │
                     ▼                 ▼
                Executor         Synthesizer
                                       │
                                       ▼
                                      END
```

---

## Directory Structure

```
langgraph-rag-agent/

├── app.py
├── graph.py
├── state.py
├── registry.py
├── errors.py
├── error_handler.py
│
├── planner.py
├── executor.py
├── replanner.py
├── synthesizer.py
├── agent.py
│
├── tools/
│   ├── calculator.py
│   ├── rag.py
│   ├── llm.py
│   └── direct.py
│
├── data/
├── ingest.py
├── README.md
└── requirements.txt
```

---

# Tech Stack

- Python
- LangGraph
- LangChain
- Google Gemini
- ChromaDB
- HuggingFace Embeddings
- Pydantic

---

# Current Workflow

1. User asks a question.
2. Planner decomposes it into executable steps.
3. Executor runs available tools.
4. Replanner checks if additional work is required.
5. Executor continues until complete.
6. Synthesizer generates the final response.
7. Error Handler manages failures gracefully.

---

# Current Capabilities

✅ Multi-tool planning

✅ Dependency resolution

✅ RAG

✅ Calculator

✅ General LLM reasoning

✅ Replanning

✅ Centralized state management

✅ Execution records

✅ Conditional routing

✅ Error handling

---

# Roadmap

## Sprint 1 ✅
- Basic LangGraph RAG agent

## Sprint 2.0A ✅
- Planner
- Executor
- Replanner
- Synthesizer
- Tool Registry
- AgentState
- Error Handling

## Sprint 2.0B 🚧
- Logging
- Retry improvements
- Production-grade exception handling
- Synthesizer optimization

## Sprint 2.1
- Parallel execution

## Sprint 2.2
- Memory

## Sprint 2.3
- FastAPI deployment
- Streaming responses
- Docker

---

# Run

```bash
pip install -r requirements.txt

python ingest.py

python app.py
```