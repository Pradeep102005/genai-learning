from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
model=ChatOllama(model="gemma2:2b",temperature=0)
template=PromptTemplate(
    template="Generate 5 interesting  facts about {topic}",
    input_variables=["topic"]
)
parser=StrOutputParser()        
chain=template|model|parser
result=chain.invoke({'topic':'Indian Cricket'})
print(result)
chain.get_graph().print_ascii()