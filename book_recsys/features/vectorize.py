"""TF-IDF and bag-of-words vectorization of book documents."""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def tfidf_matrix(documents, max_features: int = 5000):
    """Return (sparse TF-IDF matrix, fitted vectorizer)."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    return vectorizer.fit_transform(documents), vectorizer


def bow_matrix(documents, max_features: int = 5000):
    """Return (sparse bag-of-words count matrix, fitted vectorizer)."""
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    return vectorizer.fit_transform(documents), vectorizer
