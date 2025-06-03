"""7.Summarize long texts using a pre-trained summarization model using Hugging face model.
Load the summarization pipeline. Take a passage as input and obtain the summarized text.
"""

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def sum_text(text, max_length=130, min_length=50):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["sum_text"]

long_text = """
Rcb won 2025 and congratulations for it RCB you deserve this season"""

summary = sum_text(long_text)

print(long_text)
print("0"*40)
print(summary)