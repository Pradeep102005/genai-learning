from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage
from langchain.agents import create_agent
@tool
def multiply(a:int,n:int)->int:
    """Given 2 numbers a and b return their product"""
    return a*n
    

# print(multiply.invoke({"a":-10,"n":-20}))
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

model=ChatOllama(model="qwen2.5:7b-instruct")
llm_with_tools=model.bind_tools([multiply])
query= HumanMessage(content='can you multiply 3 and 300')
message=[query]
print(message)
result=llm_with_tools.invoke(message)
message.append(result)
print(message)
# response.tool_calls[0]
tool_result=multiply.invoke(result.tool_calls[0])
message.append(tool_result)
print(message)
final_result=llm_with_tools.invoke(message)
print(final_result.content)
# response=llm_with_tools.invoke([HumanMessage(content="what is the product of 10 and 20")])
# print(response)