from langchain_ollama import OllamaEmbeddings

emb = OllamaEmbeddings(model="nomic-embed-text")

vec = emb.embed_query("I love dogs")
print(str(vec))   # a long vector of numbers
