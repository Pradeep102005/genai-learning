from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
emb = OllamaEmbeddings(model="nomic-embed-text")
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]
query="tell me about Indian Fast Bowlers"
doc_embeddings=emb.embed_documents(documents)
query_embedding=emb.embed_query(query)
scores=cosine_similarity([query_embedding],doc_embeddings)[0]
for i,score in enumerate(scores):
    print(f"Document {i+1}: {score}")