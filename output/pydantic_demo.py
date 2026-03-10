from langchain_ollama import ChatOllama
from typing import TypedDict,Annotated
from pydantic import BaseModel,Field

# Create model
model = ChatOllama(model="gemma2:2b", temperature=0)

# Define schema using TypedDict
class Review(BaseModel):
    key_themes:list[str]=Field(description="Write down all key themes discussed  in the text in a form of list")
    summary:str=Field(description="Write down a summary of the text")
    sentiment:str=Field(description="Write down the sentiment of the text either as + or -")
    

    

# Create structured model from base model
structured_model = model.with_structured_output(Review)

# Input text
text = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. 
The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. 
What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. 
Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. 
Also, Samsung’s One UI still comes with bloatware. The $1,300 price tag is also a hard pill to swallow.
"""

# Call the model
result = structured_model.invoke(text)

# Print results
print("Summary:", result.summary)
print("Sentiment:", result.sentiment)
print("Key Themes:", result.key_themes)
