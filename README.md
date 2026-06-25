# LangGraph RAG Agent

A conversational Agentic AI application built with LangGraph, Gemini, ChromaDB, and HuggingFace Embeddings.

## Features

- PDF ingestion
- ChromaDB vector store
- Semantic retrieval
- Retrieval-Augmented Generation (RAG)
- LangGraph workflow
- LLM-based routing
- Structured output using Pydantic
- Message-based conversation state
- Conditional graph execution

## Directory Structure

langgraph-rag-agent/
│
├── data/
│   ├── fastapi.pdf
│   ├── langgraph.pdf
│   └── rag.pdf
│
├── app.py
├── graph.py
├── nodes.py
├── state.py
├── schemas.py
├── ingest.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE (optional)

## Architecture

START
 ↓
Agent
 ↓
Gemini Router (Structured Output)
 ├── Direct Reply
 └── Retrieve
        ↓
     ChromaDB
        ↓
     Gemini
        ↓
      AIMessage
        ↓
END

## Tech Stack

- Python
- LangGraph
- LangChain
- Google Gemini
- ChromaDB
- HuggingFace Embeddings
- Pydantic

## Run

```bash
pip install -r requirements.txt
python ingest.py
python app.py
```