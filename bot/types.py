from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create model
llm = ChatOllama(model="gemma2:2b", temperature=0.2)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains things in simple words."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Explain {topic} in {level} level.")
])

# Create chain
chain = prompt | llm

# Start with EMPTY history (fresh chat)
chat_history = []

while True:
    topic = input("Enter topic (or type exit): ")
    if topic.lower() == "exit":
        break

    level = input("Enter level (beginner/intermediate/advanced): ")

    # Call chain
    response = chain.invoke({
        "chat_history": chat_history,
        "topic": topic,
        "level": level
    })

    print("Bot:", response.content)

    # Save conversation
    chat_history.append(("user", f"Explain {topic} in {level} level."))
    chat_history.append(("assistant", response.content))
