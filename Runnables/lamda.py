from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough,RunnableSequence,RunnableLambda

def word_count(text):
    return len(text.split())
# Initialize the model
model = ChatOllama(model="gemma2:2b", temperature=0)
prompt=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)
parser=StrOutputParser()
joke_gen_chain=RunnableSequence(prompt,model,parser)
parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'wordcount':RunnableLambda(word_count)
})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)
result=final_chain.invoke({'topic':'Unemployment'})
print(result)
