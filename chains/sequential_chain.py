import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Initialize the model
model = ChatOllama(model="gemma2:2b", temperature=0)

# The text we will run the chain on
text = """
The James Webb Space Telescope (JWST) is a space telescope designed to conduct infrared astronomy. 
Its high-resolution and high-sensitivity instruments allow it to view objects too old, distant, 
or faint for the Hubble Space Telescope. This will enable investigations across many fields of 
astronomy and cosmology, such as observation of the first stars and the formation of the first galaxies, 
and detailed atmospheric characterization of potentially habitable exoplanets.
"""

# 1st Prompt: Write notes about the text
notes_prompt = PromptTemplate(
    template="Write concise notes about the following text:\n{text}",
    input_variables=["text"]
)

# 2nd Prompt: Write 5 questions about the text
quiz_prompt = PromptTemplate(
    template="Write 5 quiz questions based on the following text:\n{text}",
    input_variables=["text"]
)

# 3rd Prompt: Combine notes and quiz
combine_prompt = PromptTemplate(
    template="Here are the notes:\n{notes}\n\nHere are the quiz questions:\n{quiz}\n\nPlease format them nicely together into a single summary document.",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

# Chain to generate notes
notes_chain = notes_prompt | model | parser

# Chain to generate quiz
quiz_chain = quiz_prompt | model | parser

# We use RunnablePassthrough to pass 'text' to both notes_chain and quiz_chain,
# and capture their outputs as 'notes' and 'quiz', which then go into combine_prompt.
overall_chain = (
    {"notes": notes_chain, "quiz": quiz_chain}
    | combine_prompt
    | model
    | parser
)

print("Running the sequential chain...")
result = overall_chain.invoke({'text': text})

print("\n------------ FINAL COMBINED OUTPUT ------------")
print(result)


print("\n------------ CHAIN GRAPH ------------")
overall_chain.get_graph().print_ascii()
