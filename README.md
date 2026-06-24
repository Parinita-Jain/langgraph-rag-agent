# LangGraph RAG Agent

A simple Agentic AI application built using:

- LangGraph
- ChromaDB
- HuggingFace Embeddings
- Google Gemini
- LangChain

## Features

- PDF document ingestion
- Semantic search using ChromaDB
- Retrieval-Augmented Generation (RAG)
- LLM-based routing
- Conditional execution paths using LangGraph

## Architecture

User Question
↓
Agent Node
↓
Router Node (Gemini)
├── Direct Response
└── Retrieval
↓
Generate Answer
↓
Final Response

## Setup

Install dependencies:

pip install -r requirements.txt

Create .env

GOOGLE_API_KEY=your_api_key

Build vector database:

python ingest.py

Run:

python app.py