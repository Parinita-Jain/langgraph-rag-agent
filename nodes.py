from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

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

#retrieve node

def retrieve_node(state):
    print("\n ========= Retrieve Node =======")
    question = state["question"]

    docs = vectorstore.similarity_search(question,k=7)

    return {"documents":docs}

#Generate Node

def generate_node(state):
    print("\n ======= generate node =====")

    question=state["question"]
    docs=state["documents"]

    context="\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }

def router_node(state):

    question = state["question"]

    prompt = f"""
You are a routing agent.

Decide whether the user needs:

direct
- greetings
- casual conversation

retrieve
- questions requiring knowledge lookup
- questions about LangGraph
- questions about FastAPI
- questions about RAG

Return ONLY one word:

direct
or

retrieve

Question:
{question}
"""

    response = llm.invoke(prompt)

    route = response.content.strip().lower()

    print("\n===== ROUTER NODE =====")
    print("Question:", question)
    print("Route:", route)

    return {
        "route": route
    }

#   Route Selector   
def choose_route(state):

    return state["route"]

#Direct Reply Node


def direct_node(state):

    return {
        "answer": "Hello! How can I help you today?"
    }

