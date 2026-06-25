from graph import app

from langchain_core.messages import HumanMessage

conversation = []

print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    # Add user's message
    conversation.append(
        HumanMessage(content=question)
    )

    # Run graph
    result = app.invoke(
        {
            "messages": conversation
        }
    )

    # Save updated conversation
    conversation = result["messages"]

    # Print latest AI reply
    print("\nAssistant:")

    print(conversation[-1].content)

    print("-" * 50)