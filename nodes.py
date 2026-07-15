from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import PlannerOutput, PlanStep

from concurrent.futures import ThreadPoolExecutor, as_completed
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
            ],

            "output": {
                "answer": response.content
            }

        }

# ===========================
# Planner
# ===========================
def planner_node(state):

    question = state["messages"][-1].content
    question_lower = question.lower().strip()

    print("\n===== PLANNER NODE =====")
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
    - greetings only

    calculator
    - arithmetic calculations

    rag
    - questions requiring document retrieval

    llm
    - reasoning
    - explanations
    - comparisons
    - summarization
    - answering questions that do not require retrieval


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
5. If a step depends on the output of a previous step, reference the required output field.

    Use this format:

    #<step_id>.<field>

    Examples:

    #1.value
    #2.answer

    Do NOT use only #1 or #2.

    For calculator outputs use:
    #<step>.value

    For llm/rag/direct outputs use:
    #<step>.answer

    Example 1

    User:
    Calculate 15% of ₹800.
    Then multiply the result by 5.

    Steps:

    1.
    tool = calculator
    tool_input = 0.15 * 800

    2.
    tool = calculator
    tool_input = #1.value * 5
    depends_on = [1]

    Example 2

    User:
    Explain RAG.
    Summarize the explanation.

    Steps:

    1.
    tool = rag
    tool_input = Explain RAG

    2.
    tool = llm
    tool_input = Summarize #1.answer
    depends_on = [1]


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

 

# ===========================
# Tools
# ===========================

#Direct Reply Node

def greeting_tool(state):

    answer = "Hello! How can I help you today?"

    return {

        "messages": [
            AIMessage(content=answer)
        ],

        "output": {
            "answer": answer
        }

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
                    ],

                    "output": {
                        "value": result
                    }

                }

    except Exception:

        return {
            "messages": [
                AIMessage(
                    content="Sorry, I couldn't calculate that expression."
                )
            ]
        }
    
def llm_tool(state):

    print("\n===== LLM TOOL =====")

    prompt = state["tool_input"]

    response = llm.invoke(prompt)

    return {

        "messages": [
            AIMessage(content=response.content)
        ],

        "output": {
            "answer": response.content
        }

    }

def rag_tool(state):

    print("\n===== RAG TOOL =====")

    tool_state = state.copy()

    tool_state.update(rewrite_node(tool_state))
    tool_state.update(retrieve_node(tool_state))


    return generate_node(tool_state)

TOOLS = {
    "direct": greeting_tool,
    "calculator": calculator_node,
    "rag": rag_tool,
    "llm": llm_tool,
}

# ===========================
# Executor
# ===========================

def can_execute(step, completed_steps):

    return all(
    dependency in completed_steps
    for dependency in step.depends_on
    )

def get_ready_steps(pending_steps, completed_steps):

    ready_steps = []

    for step in pending_steps:

        if can_execute(step, completed_steps):

            ready_steps.append(step)

    return ready_steps



def resolve_step_references(tool_input, tool_results):

    pattern = r"#(\d+)\.(\w+)"

    matches = re.findall(pattern, tool_input)

    for step_id, field in matches:

        step_id = int(step_id)

        if step_id not in tool_results:
            raise ValueError(
                f"Step {step_id} has not been executed yet."
            )

        result = tool_results[step_id]["output"][field]

        tool_input = tool_input.replace(
            f"#{step_id}.{field}",
            str(result)
        )

    return tool_input

def execute_step(step, state, tool_results):

    tool_name = step.tool

    tool_input = resolve_step_references(
        step.tool_input,
        tool_results
    )

    print("Original Input:", step.tool_input)
    print("Resolved Input:", tool_input)

    tool_state = state.copy()
    tool_state["tool_input"] = tool_input

    tool_function = TOOLS[tool_name]

    result = tool_function(tool_state)

    return result

def executor_node(state):

    print("\n===== EXECUTOR NODE =====")

    tool_results = {}

    completed_steps = set()

    pending_steps = list(state["steps"])

    while pending_steps:

        ready_steps = get_ready_steps(
            pending_steps,
            completed_steps
        )

        if not ready_steps:
            raise ValueError(
                f"No executable step found.\n"
                f"Completed: {completed_steps}\n"
                f"Pending: {[step.id for step in pending_steps]}"
            )

        print("\n===== READY STEPS =====")

        for step in ready_steps:
            print(
                f"Step {step.id}: {step.tool}"
            )

        with ThreadPoolExecutor() as executor:

            futures = {}

            for step in ready_steps:

                print(
                    f"\nSubmitting Step {step.id}"
                    f" (depends_on={step.depends_on})"
                )

                future = executor.submit(
                    execute_step,
                    step,
                    state,
                    tool_results
                )

                futures[future] = step

            for future in as_completed(futures):

                step = futures[future]

                result = future.result()

                tool_results[step.id] = result

                completed_steps.add(step.id)

                pending_steps.remove(step)

                print(f"Completed Step {step.id}")

        print("\nCompleted:", completed_steps)

    print("\n===== TOOL RESULTS =====")
    print(tool_results)

    return synthesize_response(
    state,
    tool_results
)

# ===========================
# Synthesizer
# ===========================

def synthesize_response(state, tool_results):

    print("\n===== SYNTHESIZER =====")

    conversation = build_conversation(state["messages"])

    results = ""

    for step in state["steps"]:

        result = tool_results[step.id]

        output = result["output"]

        results += f"""
            Step {step.id} Output:

            {output}

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