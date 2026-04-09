from transformers import pipeline

# REAL multi-label model
classifier = pipeline(
    "text-classification",
    model="unitary/unbiased-toxic-roberta",
    top_k=None
)

def predict_toxicity(text: str):
    results = classifier(text)[0]

    scores = {}

    for item in results:
        label = item["label"].lower()
        score = float(item["score"])
        scores[label] = score

    return scores