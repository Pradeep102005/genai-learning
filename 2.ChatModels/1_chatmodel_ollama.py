from langchain_ollama import ChatOllama

chat = ChatOllama(model="qwen3:8b",temperature=0.1)
response = chat.invoke("What is the capital of India?")
print(response.content)
