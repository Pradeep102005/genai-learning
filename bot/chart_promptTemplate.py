from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Create model
llm = ChatOllama(model="gemma2:2b", temperature=0.2)

# 2. Create prompt template with dynamic chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# 3. Create chain
chain = prompt | llm

# 4. Store messages here
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # 5. Call model with dynamic messages
    response = chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })

    print("Bot:", response.content)

    # 6. Update history
    chat_history.append(("user", user_input))
    chat_history.append(("assistant", response.content))
