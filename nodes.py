from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import RouteDecision

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

Your job is to rewrite ONLY the user's latest question into a standalone question.

Conversation:
{conversation}

Standalone Question:
"""

    response = llm.invoke(prompt)

    print("Rewritten Question:", response.content)

    return {
        "rewritten_question": response.content
    }

#retrieve node

def retrieve_node(state):
    print("\n ========= Retrieve Node =======")
    messages = state["messages"]

    question = state["rewritten_question"]

    docs = vectorstore.similarity_search(question,k=7)

    return {"documents":docs}

#Generate Node

def generate_node(state):
    print("\n ======= generate node =====")

    messages = state["messages"]

    conversation = build_conversation(state["messages"]) 

    question = messages[-1].content

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

    messages = state["messages"]

    question = messages[-1].content

    prompt = f"""
You are a routing agent.

Choose one route:

direct
- greetings
- casual conversation

retrieve
- knowledge questions
- questions about LangGraph
- questions about FastAPI
- questions about RAG

Question:
{question}
"""

    structured_llm = llm.with_structured_output(
        RouteDecision
    )

    result = structured_llm.invoke(prompt)

    print("\n===== ROUTER NODE =====")
    print("Question:", question)
    print("Route:", result.route)

    return {
        "route": result.route
    }
#   Route Selector   
def choose_route(state):

    return state["route"]

#Direct Reply Node


def direct_node(state):

    return {
        "messages": [
            AIMessage(
                content="Hello! How can I help you today?"
            )
        ]
    }

