from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
model=ChatOllama(model="gemma2:2b")
prompt = PromptTemplate(
    template="""
Summarize the following poem in 3–4 sentences.

Poem:
{poem}
""",
    input_variables=["poem"]
)
parser=StrOutputParser()

loader=TextLoader("poem.txt")
documents=loader.load()
chain=prompt|model|parser
result=chain.invoke({'poem':documents[0].page_content})
print(result)
