from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOllama(model="gemma2:2b", temperature=0.2)

# Store messages here
messages = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # 5. Add user message
    messages.append(HumanMessage(content=user_input))

    # 6. Call model with dynamic messages
    response = llm.invoke(messages)

    # 7. Add bot response
    messages.append(AIMessage(content=response.content))

    print("Bot:", response.content)