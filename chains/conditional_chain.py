from typing import Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough


# Initialize the model
# For structured outputs, we need a model that supports it well.
model = ChatOllama(model="gemma2:2b", temperature=0)

# 1. Define the Pydantic class to enforce strict POSITIVE or NEGATIVE output
class ReviewSentiment(BaseModel):
    sentiment: Literal["POSITIVE", "NEGATIVE"] = Field(
        description="The sentiment of the hotel review. Must be exactly 'POSITIVE' or 'NEGATIVE'."
    )

# 2. Bind the model to the structured output
structured_model = model.with_structured_output(ReviewSentiment)

# The written text about the hotel experience 
# (You can test the negative path by uncommenting the second review)
review = """
The hotel stay was amazing! The room was clean, the staff was very friendly, 
and the breakfast was delicious. I highly recommend this place.
"""

# review = """
# It was a terrible experience. The AC didn't work, my room was dirty, 
# and the staff was very rude to us when we asked for help.
# """

# 3. Chain to analyze the sentiment of the review using the structured model
classification_prompt = PromptTemplate.from_template(
    """Analyze the sentiment of the following hotel review.
    
Review: {review}"""
)
# This chain now directly outputs the Pydantic object
classification_chain = classification_prompt | structured_model


# 4. Define the exact response for a positive review
positive_prompt = PromptTemplate.from_template(
    """You are an automated hotel reply system responding to a positive review. 
Review: {review}
Reply briefly and politely: 'Thank you for your wonderful feedback! We are glad you enjoyed your stay.'

Reply:"""
)
positive_chain = positive_prompt | model | StrOutputParser()


# 5. Define the exact response for a negative review
negative_prompt = PromptTemplate.from_template(
    """You are an automated hotel reply system responding to a negative review. 
Review: {review}
Reply briefly and politely: 'We apologize for the inconvenience. We will raise the complaint with the management.'

Reply:"""
)
negative_chain = negative_prompt | model | StrOutputParser()


# 6. Create the conditional branch
# We route based on the Pydantic object's .sentiment field
branch = RunnableBranch(
    (lambda x: x["sentiment_obj"].sentiment == "POSITIVE", positive_chain),
    (lambda x: x["sentiment_obj"].sentiment == "NEGATIVE", negative_chain),
    positive_chain  # Default fallback if neither matches perfectly
)


# 7. Assemble the full LCEL chain
full_chain = (
    # We pass the initial review directly through as 'review', 
    # simultaneously mapping the review through the classification_chain to get the 'sentiment_obj' (Pydantic model)
    {"review": RunnablePassthrough(), 
    "sentiment_obj": classification_chain}
    | branch
)

print(f"Original Review:\n{review.strip()}\n")
print("-" * 50)

# Run the conditional chain
result = full_chain.invoke(review)

print("Automated Reply:")
print(result.strip())
full_chain.get_graph().print_ascii()
