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
prompt1=template1.invoke({"topic":"AI"})
result=model.invoke(prompt1)
print(result.content)
prompt2=template2.invoke({"text":result.content})
result2=model.invoke(prompt2)
print(result2.content)
#1st parser
parser1=StrOutputParser()
