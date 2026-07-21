from langchain_core.messages import AIMessage
def greeting_tool(state):

    answer = "Hello! How can I help you today?"

    return {

        "messages": [
            AIMessage(content=answer)
        ],

        "output": {
            "answer": answer
        },

        "success": True,

        "error": None

        }