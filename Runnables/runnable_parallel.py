from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough,RunnableSequence

# Initialize the model
model = ChatOllama(model="gemma2:2b", temperature=0)

prompt1=PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
) 
prompt2=PromptTemplate(
    template='Generate a linked post on topic {topic}',
    input_variables=['topic']
)
prompt3=PromptTemplate(
    template='Generate a instagram post about {topic}',
    input_variables=['topic']
)

chain1=RunnableSequence(prompt1,model,StrOutputParser())
chain2=RunnableSequence(prompt2,model,StrOutputParser())
chain3=RunnableSequence(prompt3,model,StrOutputParser())

parallel_chain=RunnableParallel({
    'tweet':chain1,
    'linked':chain2,
    'instagram':chain3
})

result=parallel_chain.invoke({'topic':'Indian Cricket'})
print(result)