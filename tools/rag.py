from utils import build_conversation
from shared import llm
from shared import vectorstore
from langchain_core.messages import AIMessage
def rewrite_node(state):
    print("\n ======= rewrite node =====")
    question = state["tool_input"]

    prompt = f"""
        You are a query rewriting assistant.

        Rewrite the following question into a standalone question.

        Do NOT answer it.

        Do NOT change its meaning.

        Question:

        {question}

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

    docs = state["documents"]

    context="\n\n".join(doc.page_content for doc in docs)

    question = state["tool_input"]

    prompt = f"""
    You are the RAG tool.

    Your job is ONLY to answer the assigned tool input.

    Do NOT answer any other part of the user's request.

    Question:

    {question}

    Retrieved Context:

    {context}

    Answer only this question.
    """   

    response = llm.invoke(prompt)

    try:

        response = llm.invoke(prompt)

        return {

            "messages": [
                AIMessage(content=response.content)
            ],

            "output": {
                "answer": response.content
            },

            "success": True,

            "error": None

        }

    except Exception as e:

        return {

            "messages": [
                AIMessage(content="RAG tool failed.")
            ],

            "output": {},

            "success": False,

            "error": str(e)

        }

def rag_tool(state):

    print("\n===== RAG TOOL =====")

    tool_state = state.copy()

    tool_state.update(rewrite_node(tool_state))
    tool_state.update(retrieve_node(tool_state))


    return generate_node(tool_state)
