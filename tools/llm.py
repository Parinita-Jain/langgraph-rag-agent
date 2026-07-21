from shared import llm
from langchain_core.messages import AIMessage
import os
def llm_tool(state):

    print("\n===== LLM TOOL =====")

    prompt = state["tool_input"]

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
                AIMessage(content="LLM tool failed.")
            ],

            "output": {},

            "success": False,

            "error": str(e)

        }