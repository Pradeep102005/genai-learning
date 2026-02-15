from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen3:8b")
print(llm.invoke("Hello, how are you?"))
