import tools

from graph import app
from langchain_core.messages import HumanMessage

conversation = []

print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    conversation.append(
        HumanMessage(content=question)
    )

    result = app.invoke(
        {
            "messages": conversation
        }
    )

    conversation = result["messages"]

    print("\nAssistant:")
    print(conversation[-1].content)
    print("-" * 50)