from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import PlannerOutput
import re

load_dotenv()

#Shared Objects


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# add agent node
 
def agent_node(state):

    print("\n===== AGENT NODE =====")
    print(state)

    return {}

def build_conversation(messages):

    conversation = ""

    for message in messages:

        if isinstance(message, HumanMessage):
            conversation += f"Human: {message.content}\n\n"

        elif isinstance(message, AIMessage):
            conversation += f"AI: {message.content}\n\n"

    return conversation

# rewriting query

def rewrite_node(state):
    print("\n ======= rewrite node =====")
    conversation = build_conversation(state["messages"])

    prompt = f"""
You are a query rewriting assistant.

Rewrite ONLY the user's latest question into a standalone question.

Do NOT answer the question.

Do NOT change the user's intent.

Do NOT infer a different meaning.

Keep abbreviations like RAG, FastAPI, LangGraph exactly as written.

Return only the rewritten question.

Conversation:
{conversation}

Standalone Question:
"""

    response = llm.invoke(prompt)

    print("Rewritten Question:", response.content)

    return {
        "tool_input": response.content
    }

#retrieve node

def retrieve_node(state):
    print("\n ========= Retrieve Node =======")

    question = state["tool_input"]

    docs = vectorstore.similarity_search(question,k=7)

    return {"documents":docs}

#Generate Node

def generate_node(state):
    print("\n ======= generate node =====")

    messages = state["messages"]

    conversation = build_conversation(state["messages"]) 

    docs = state["documents"]

    context="\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
        You are a helpful AI assistant.

        Use BOTH the conversation history and the retrieved context.

        Conversation:
        {conversation}

        Retrieved Context:
        {context}

        Answer the user's latest question.
    """

    response = llm.invoke(prompt)

    return {
    "messages": [
        AIMessage(content=response.content)
    ]
}
def router_node(state):

    question = state["messages"][-1].content
    question_lower = question.lower().strip()

    print("\n===== ROUTER NODE =====")
    print("Question:", question)

    # -------------------------
    # Rule 1 : Greetings
    # -------------------------

    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    if question_lower in greetings:

        print("Tool: direct (Python Rule)")

        return {
            "tool": "direct",
            "tool_input": ""
        }


    # -------------------------
    # Rule 2 : Calculator
    # -------------------------

    calculator_pattern = r"^[0-9+\-*/().%\s]+$"

    if re.fullmatch(calculator_pattern, question):

        print("Tool: calculator (Python Rule)")

        return {
            "tool": "calculator",
            "tool_input": question
        }

    # -------------------------
    # Rule 3 : Ask Gemini
    # -------------------------

    prompt = f"""
You are a tool planning assistant.

Your job is to create a tool execution plan.

Available tools:

direct
- greetings
- casual conversation

calculator
- arithmetic calculations

rag
- questions about LangGraph
- FastAPI
- RAG
- document knowledge

Return exactly one tool call inside the tool_calls list.

Question:
{question}
"""


    structured_llm = llm.with_structured_output(PlannerOutput)
    
    result = structured_llm.invoke(prompt)

    tool_call = result.tool_calls[0]

    print("Tool:", tool_call.tool)
    print("Tool Input:", tool_call.tool_input)

    return {
    "tool": tool_call.tool,
    "tool_input": tool_call.tool_input
    }

#   Route Selector   
def choose_route(state):

    return state["tool"]

#Direct Reply Node


def direct_node(state):

    return {
        "messages": [
            AIMessage(
                content="Hello! How can I help you today?"
            )
        ]
    }

def calculator_node(state):

    print("\n===== CALCULATOR NODE =====")

    expression = state["tool_input"]

  

    try:
        result = eval(expression)

        return {
            "messages": [
                AIMessage(content=str(result))
            ]
        }

    except Exception:

        return {
            "messages": [
                AIMessage(
                    content="Sorry, I couldn't calculate that expression."
                )
            ]
        }

def rag_tool(state):

    print("\n===== RAG TOOL =====")

    # Step 1 : Rewrite
    state.update(rewrite_node(state))

    # Step 2 : Retrieve

    state.update(retrieve_node(state))

    # Step 3 : Generate
    generate_result = generate_node(state)

    return generate_result

TOOLS = {
    "direct": direct_node,
    "calculator": calculator_node,
    "rag": rag_tool,
}

def executor_node(state):

    print("\n===== EXECUTOR NODE =====")

    tool_name = state["tool"]

    print("Executing:", tool_name)

    tool = TOOLS[tool_name]

    return tool(state)

