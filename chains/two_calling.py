from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the model
model = ChatOllama(model="gemma2:2b", temperature=0)

# 1st Prompt: Generate 5 interesting facts
prompt1 = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

# 2nd Prompt: Summarize the given facts
prompt2 = PromptTemplate(
    template="Summarize the following facts in one short paragraph:\n{facts}",
    input_variables=["facts"]
)

parser = StrOutputParser()

# The first chain generates the facts
chain1 = prompt1 | model | parser

# The overall chain takes the output of chain1 (assigned to the "facts" variable) 
# and passes it to prompt2, then to the model and parser
overall_chain = {"facts": chain1} | prompt2 | model | parser

print("Calling the model...")
result = overall_chain.invoke({'topic': 'Indian Cricket'})

print("------------ FINAL SUMMARY ------------")
print(result)

print("\n------------ CHAIN GRAPH ------------")
overall_chain.get_graph().print_ascii()