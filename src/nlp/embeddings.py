from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

def fit_vectorizer(texts):
    return vectorizer.fit(texts)

def transform(texts):
    return vectorizer.transform(texts).toarray()