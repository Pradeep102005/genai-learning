from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage
from langchain.agents import create_agent
import requests
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import json
from langchain.agents import create_agent
# from langchain import hub

from langchain_community.tools import DuckDuckGoSearchRun

search_tool=DuckDuckGoSearchRun()
model=ChatOllama(model="qwen2.5:7b-instruct")
@tool
def get_weather_data(city:str)->str:
    """ This function fetches the current weather data for a given city"""
    url=f'https://api.weatherstack.com/current?access_key=271c132bc58cb90a9a72ca7289d6cf23&query={city}'
    response=requests.get(url)
    return response.json()

# prompt=hub.pull("hwchase17/react") #pulls the standard ReAct agent prompt
agent=create_agent(
    model=model,
    tools=[search_tool,get_weather_data]
    # prompt=prompt,
)

#invoke
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Find the capital of Andhra Pradesh, tell its population and current weather"}
    ]
})
print(response)

