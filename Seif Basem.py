from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

texts = [
    "I love this product!",
    "This is the worst experience ever.",
    "The service was okay."
]

results = sentiment_analyzer(texts)

for text, res in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {res['label']} (confidence {res['score']:.2f})")
    print()