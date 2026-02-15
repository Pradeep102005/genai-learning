from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# 1. Create the chat model (small model to save RAM)
chat = ChatOllama(model="gemma2:2b", temperature=0.2)

# 2. Create the prompt template with 3 input variables
template = PromptTemplate(
    template="""
Write the objective of a research paper in clear and simple language.

Topic: {topic}
Method: {method}
Domain: {domain}

Write the objective in 3-4 lines.
""",
    input_variables=["topic", "method", "domain"]
)

# 3. Chain the template and the model
chain = template | chat

# 4. Take input from user
topic = input("Enter research topic: ")
method = input("Enter method used: ")
domain = input("Enter domain: ")

# 5. Invoke the chain with user inputs
response = chain.invoke({
    "topic": topic,
    "method": method,
    "domain": domain
})

# 6. Print the result
print("\nGenerated Research Paper Objective:\n")
print(response.content)
