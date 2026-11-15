import joblib
import numpy as np
from gensim.models import Word2Vec, FastText

# Load saved artifacts once (no retraining)
_bow_vec = joblib.load("models/bow_vectorizer.joblib")
_tfidf_vec = joblib.load("models/tfidf_vectorizer.joblib")
_cbow_model = Word2Vec.load("models/word2vec_cbow.model")
_skipgram_model = Word2Vec.load("models/word2vec_skipgram.model")
_ft_model = FastText.load("models/fasttext.model")


def _sentence_embedding(tokens, model):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def vectorize_document(text: str, method: str = "tfidf") -> np.ndarray:
    """
    Returns a 1D numerical vector for a raw/cleaned document string.
    Supported methods: 'bow', 'tfidf', 'cbow', 'skipgram', 'fasttext'
    """
    method = method.lower().strip()
    if method == "bow":
        return _bow_vec.transform([text]).toarray()[0]
    elif method == "tfidf":
        return _tfidf_vec.transform([text]).toarray()[0]
    elif method == "cbow":
        return _sentence_embedding(text.split(), _cbow_model)
    elif method == "skipgram":
        return _sentence_embedding(text.split(), _skipgram_model)
    elif method == "fasttext":
        return _sentence_embedding(text.split(), _ft_model)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: bow, tfidf, cbow, skipgram, fasttext.")


if __name__ == "__main__":
    sample = "im getting borderland murder"
    for m in ["bow", "tfidf", "cbow", "skipgram", "fasttext"]:
        vec = vectorize_document(sample, method=m)
        print(f"{m:10s} -> shape: {vec.shape}, sample values: {np.round(vec[:3], 4)}")