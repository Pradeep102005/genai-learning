from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough,RunnableSequence,RunnableBranch,RunnableLambda

model=ChatOllama(model="gemma2:2b")
prompt=PromptTemplate(
    template='Write a detailed summary about {topic}',
    input_variables=['topic']
)
parser=StrOutputParser()
summary_chain=RunnableSequence(prompt,model,parser)
prompt2=PromptTemplate(
    template='Write a detailed summary about {topic}',
    input_variables=['topic']
)
branch_chain=RunnableBranch(
    (lambda x:len(x.split())>500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)
final_chain=RunnableSequence(summary_chain,branch_chain)
result=final_chain.invoke({'topic':'IRAN US WAR'})
print(result)