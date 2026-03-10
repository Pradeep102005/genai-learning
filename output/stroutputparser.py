from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
model=ChatOllama(model="gemma2:2b",temperature=0)
#1st template
template1=PromptTemplate(
    template="""
    Write a detailed report on {topic}         
    """,
    input_variables=["topic"],
)

#2nd template
template2=PromptTemplate(
    template="""
    Write a 5 line summary on the following text. /n{text}
    """,
    input_variables=["text"],
)
parser=StrOutputParser()

chain=template1|model|parser|template2|model|parser
result=chain.invoke({"topic":"AI"})
print(result)
