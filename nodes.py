from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import PlannerOutput, PlanStep
import re

import ast

import operator

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
                "steps": [
                    PlanStep(
                        id=1,
                        tool="direct",
                        tool_input="",
                        depends_on=[]
                    )
                ]
            }


    # -------------------------
    # Rule 2 : Calculator
    # -------------------------

    calculator_pattern = r"^[0-9+\-*/().%\s]+$"

    if re.fullmatch(calculator_pattern, question):

        print("Tool: calculator (Python Rule)")

        return {
                    "steps": [
                        PlanStep(
                            id=1,
                            tool="calculator",
                            tool_input=question,
                            depends_on=[]
                        )
                    ]
                }

    # -------------------------
    # Rule 3 : Ask Gemini
    # -------------------------

    prompt = f"""
You are an AI planning assistant.

Your job is to create an execution plan.

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

Rules:

1. You may use ONE OR MORE tools.

2. If multiple independent tasks are requested,
   Return a workflow plan.

   Each step must contain:

    - id
    - tool
    - tool_input

   Return a list called steps.
   For every step, also return depends_on.

   If the step has no dependencies,
   return an empty list.

3. Keep the original order of execution.

4. Generate a precise and self-contained tool_input for every tool.

   Each tool_input should contain only the information needed by that tool.

   Do not include parts of the request that belong to another tool.

Question:

{question}
"""


    structured_llm = llm.with_structured_output(PlannerOutput)
    
    result = structured_llm.invoke(prompt)

    print("\n===== EXECUTION PLAN =====")

    for step in result.steps:
        print(
            f"Step {step.id}: "
            f"{step.tool} "
            f"(input='{step.tool_input}', "
            f"depends_on={step.depends_on})"
        )

    return {
        "steps": result.steps
    }

 
def choose_route(state):

    step = state["steps"][0]

    return step.tool

#Direct Reply Node


def direct_node(state):

    return {
        "messages": [
            AIMessage(
                content="Hello! How can I help you today?"
            )
        ]
    }

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}

def evaluate(node):

    if isinstance(node, ast.Constant):
        return node.value
    
    if isinstance(node, ast.UnaryOp):

        operand = evaluate(node.operand)

        operator_function = UNARY_OPERATORS.get(type(node.op))

        if operator_function is None:
            raise ValueError("Unsupported unary operator")

        return operator_function(operand)

    if isinstance(node, ast.BinOp):

        left = evaluate(node.left)

        right = evaluate(node.right)

        print("Left:", left)
        print("Right:", right)

        operator_function = OPERATORS.get(type(node.op))
        
        if operator_function is None:
            raise ValueError("Unsupported operator")
        
        return operator_function(left, right)
    raise ValueError("Unsupported expression")


def calculator_node(state):

    print("\n===== CALCULATOR NODE =====")

    expression = state["tool_input"]  

    try:

        tree = ast.parse(expression, mode="eval")

        result = evaluate(tree.body)
       
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

    state.update(rewrite_node(state))

    state.update(retrieve_node(state))

    return generate_node(state)

TOOLS = {
    "direct": direct_node,
    "calculator": calculator_node,
    "rag": rag_tool,
}

def executor_node(state):

    print("\n===== EXECUTOR NODE =====")

    tool_results = []

    for step in state["steps"]:

        print(f"\nExecuting Step {step.id}"
              f" (depends_on={step.depends_on})")

        tool_name = step.tool
        tool_input = step.tool_input

        print("Tool:", tool_name)

        tool_state = state.copy()
        tool_state["tool_input"] = tool_input

        tool_function = TOOLS[tool_name]

        result = tool_function(tool_state)

        tool_results.append(result)

    print("\n===== TOOL RESULTS =====")
    print(tool_results)

    return synthesize_response(
    state,
    tool_results
)


def synthesize_response(state, tool_results):

    print("\n===== SYNTHESIZER =====")

    conversation = build_conversation(state["messages"])

    results = ""

    for step, result in zip(state["steps"], tool_results):

        answer = result["messages"][0].content

        results += f"""
        Step {step.id} Result:
        {answer}
        --------------------------------
        """

    prompt = f"""
You are a helpful AI assistant.

The tools have already solved the problem.

Do NOT call any tools.

Combine the tool results into one natural response.

Conversation:

{conversation}

Tool Results:

{results}
"""

    response = llm.invoke(prompt)

    return {

        "messages":[

            AIMessage(content=response.content)

        ]

    }