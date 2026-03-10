from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

# Initialize the model
model = ChatOllama(model="gemma2:2b", temperature=0)

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

# 1. Chain to analyze the sentiment of the review
classification_prompt = PromptTemplate.from_template(
    """Analyze the sentiment of the following hotel review and respond with strictly ONE Word: either POSITIVE or NEGATIVE.

Review: {review}

Sentiment:"""
)
classification_chain = classification_prompt | model | StrOutputParser()

# 2. Define the exact response for a positive review
positive_prompt = PromptTemplate.from_template(
    """You are an automated hotel reply system responding to a positive review. 
Review: {review}
Reply briefly and politely: 'Thank you for your wonderful feedback! We are glad you enjoyed your stay.'

Reply:"""
)
positive_chain = positive_prompt | model | StrOutputParser()

# 3. Define the exact response for a negative review
negative_prompt = PromptTemplate.from_template(
    """You are an automated hotel reply system responding to a negative review. 
Review: {review}
Reply briefly and politely: 'We apologize for the inconvenience. We will raise the complaint with the management.'

Reply:"""
)
negative_chain = negative_prompt | model | StrOutputParser()

# 4. Create the conditional branch
# Using RunnableBranch to route the chain depending on the result of 'sentiment'
branch = RunnableBranch(
    (lambda x: "POSITIVE" in x["sentiment"].upper(), positive_chain),
    (lambda x: "NEGATIVE" in x["sentiment"].upper(), negative_chain),
    positive_chain  # Default fallback if neither matches perfectly
)

# 5. Assemble the full LCEL chain
full_chain = (
    # We pass the initial review directly through as 'review', 
    # simultaneously mapping the review through the classification_chain to get the 'sentiment'
    {"review": RunnablePassthrough(), "sentiment": classification_chain}
    | branch
)

print(f"Original Review:\n{review.strip()}\n")
print("-" * 50)

# Run the conditional chain
result = full_chain.invoke(review)

print("Automated Reply:")
print(result.strip())
