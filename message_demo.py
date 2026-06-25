from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

conversation = [

    HumanMessage(content="What is LangGraph?"),

    AIMessage(content="LangGraph is a framework for AI workflows."),

    HumanMessage(content="Who created it?")
]

for msg in conversation:

    print(type(msg))#.__name__)
    print(msg.content)
    print("-" * 40)