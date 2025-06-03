"""Use a pre-trained Hugging Face model to analyze sentiment in text. Assume
a real-world application, Load the sentiment analysis pipeline.Analyze the
sentiment by giving sentences to input"""

from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

texts= [
    "i love deva",
    "he loves me",
    "RCB will win",
    "CSK hate win",
    "IPL 2025 RCB WINNERS"
]

for text in texts:
    result = sentiment_pipeline(text)[0]
    print(f"Text: {text}")
    print("-"*50)
    print(f"sentiment: {result['label']}, confidence: {result['score']:.2f}")
