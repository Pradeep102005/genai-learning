from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage
from langchain.agents import create_agent
import requests
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import json

@tool
def get_conversion_factor(base_currency:str,target_currency:str)->float:
    """  This function fetches the currency conversion factor between a given base curency and a target currency
    """
    url=f'https://v6.exchangerate-api.com/v6/269504a9c72e647085f9b85c/pair/{base_currency}/{target_currency}'
    response=requests.get(url)
    return response.json()


@tool
def convert(base_currency_value:int,conversion_rate:Annotated[float,InjectedToolArg])->float:
    """ given a currency conversion rate this function calculates the target currency value from a given base currency value"""
    return base_currency_value*conversion_rate

# print(convert.invoke({"base_currency_value":100,"conversion_rate":94.7}))


#tool binding
model=ChatOllama(model="qwen2.5:7b-instruct")
llm_with_tools=model.bind_tools([get_conversion_factor,convert])
messages=[HumanMessage(content="what is the conversion factor between usd and inr, and based on that convert 100 usd to inr")]
ai_message=llm_with_tools.invoke(messages)
messages.append(ai_message)
print(ai_message.tool_calls)
for tool_call in ai_message.tool_calls:
    #execute the 1st tool and get the value of conversion rate
    if tool_call["name"]=="get_conversion_factor":
        total_message1=get_conversion_factor.invoke(tool_call)
        #fetch the conversion rate from the total_message
        conversion_rate=json.loads(total_message1.content)["conversion_rate"]
        #append this tool message to message list
        messages.append(total_message1)
        #now invoke the second tool
        if tool_call["name"]=="convert":
            #fetch the current arg
            tool_call["args"]["conversion_rate"]=conversion_rate
            total_message2=convert.invoke(tool_call)
            messages.append(total_message2)

final_response=llm_with_tools.invoke(messages)
print(final_response.content)
            
        
    

