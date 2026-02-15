from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma2:2b")

chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat_history.append(("user", user_input))

    response = llm.invoke(chat_history)

    chat_history.append(("assistant", response.content))

    print("Bot:", response.content)
print("\n\n\n",chat_history)