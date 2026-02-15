from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
chat = ChatOllama(model="gemma2:2b",temperature=0.1)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

chain = prompt | chat

response = chain.invoke({"question": "Explain recursion with factorial example."})

print(response.content)