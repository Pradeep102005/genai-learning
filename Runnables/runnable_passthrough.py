from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough,RunnableSequence

# Initialize the model
model = ChatOllama(model="gemma2:2b", temperature=0)
joke_template=PromptTemplate(
    template='Tell me a joke on this topic {topic}',
    input_variables=['topic']
)
parser=StrOutputParser()
joke_chain=RunnableSequence(joke_template,model,parser)
paraller_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':joke_chain
})
result=paraller_chain.invoke({'topic':'Telugu Cinema'})
print(result)

