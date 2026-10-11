from joblib import load
from sentence_transformers import SentenceTransformer
from cleantext import clean

embedder = SentenceTransformer("models/sentence_transformer.model")
classifier = load("models/classifier.joblib")

labels = {0: "negative", 1: "neutral", 2: "positive"}


def predict(text: str) -> str:
    cleaned_text = clean(text)
    embedding = embedder.encode(cleaned_text)

    prediction = classifier.predict(embedding.reshape(1, -1))
    return labels[prediction[0]]
