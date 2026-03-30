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


@tool
def summarize_expenses(expenses:list[dict])->str:
    """Summarizes expenses. Each dict in the list must have keys: 'category' and 'amount'."""
    parts=[]
    total=0
    for expense in expenses:
        parts.append(f"{expense['amount']} on {expense['category']}")
        total+=expense["amount"]
    return f"Total: ₹{total}\n" + "\n".join(parts)


#tool_map
tool_map={
    "get_conversion_factor":get_conversion_factor,
    "convert":convert,
    "summarize_expenses":summarize_expenses
}




message=[HumanMessage(content="I spent 1000 on food and 2000 USD on rent and 500 EUR on transport. Convert all to INR and summarize")]

model=ChatOllama(model="qwen2.5:7b-instruct")
llm_with_tools=model.bind_tools([get_conversion_factor,convert,summarize_expenses])
conversion_rate=None
while True:
    ai_message=llm_with_tools.invoke(message)
    message.append(ai_message)
    if not ai_message.tool_calls:
        print("Final Response: ",ai_message.content)
        break
    for tool_call in ai_message.tool_calls:
        tool_name=tool_call["name"]
        tool_args=tool_call["args"]
        if tool_name in tool_map:
            tool_func=tool_map[tool_name]
            tool_result=tool_func.invoke(tool_args)
            message.append(tool_result)
        else:
            message.append(AIMessage(content=f"Unknown tool: {tool_name}"))