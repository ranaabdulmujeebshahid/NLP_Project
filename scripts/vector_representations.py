import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# 1. Load cleaned dataset from Phase 1
df = pd.read_csv('data/cleaned_twitter_data.csv')  # Adjust path if your cleaned file name differs
corpus = df['Cleaned_Text'].fillna('').tolist()

print("--- PART A: Count-Based Representations ---")
# A1: Bag of Words (CountVectorizer)
count_vec = CountVectorizer(max_features=5000)
bow_matrix = count_vec.fit_transform(corpus)

total_entries = bow_matrix.shape[0] * bow_matrix.shape[1]
non_zero = bow_matrix.nnz
sparsity = (1.0 - (non_zero / total_entries)) * 100

print(f"BoW Matrix Shape: {bow_matrix.shape}")
print(f"BoW Sparsity: {sparsity:.2f}%")

# A2: TF-IDF Vectorizer
tfidf_vec = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vec.fit_transform(corpus)
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

# A3: N-grams (Uni, Bi, Tri-grams vocabulary growth)
for n in range(1, 4):
    ngram_vec = CountVectorizer(ngram_range=(n, n), max_features=5000)
    ngram_matrix = ngram_vec.fit_transform(corpus)
    print(f"N-gram ({n},{n}) Vocabulary Size: {len(ngram_vec.get_feature_names_out())}")


print("\n--- PART B: Learned Embeddings ---")
# Prepare tokenized corpus for Gensim
tokenized_corpus = [text.split() for text in corpus]

# Define shared hyperparameters for fair comparison
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 10

# B1: Word2Vec - CBOW (sg=0)
print("Training Word2Vec (CBOW)...")
cbow_model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    sg=0,  # 0 for CBOW
    epochs=EPOCHS
)

# B2: Word2Vec - Skip-gram (sg=1)
print("Training Word2Vec (Skip-gram)...")
skipgram_model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    sg=1,  # 1 for Skip-gram
    epochs=EPOCHS
)

print("Both Word2Vec models trained successfully!")



from gensim.models import FastText

# B4: fastText (same hyperparameters as Word2Vec)
print("Training fastText...")
ft_model = FastText(
    sentences=tokenized_corpus,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    epochs=EPOCHS
)
print("fastText trained successfully!")

# Demonstrate OOV / misspelled word handling (Word2Vec vs fastText)
oov_word = "borderlandss"  # intentional misspelling of 'borderland'
print(f"\n--- OOV Test for '{oov_word}' ---")
try:
    print("Word2Vec (Skip-gram) neighbors:", skipgram_model.wv.most_similar(oov_word, topn=3))
except KeyError as e:
    print(f"Word2Vec (Skip-gram) failed: {e}")

print("fastText neighbors:", ft_model.wv.most_similar(oov_word, topn=3))



import numpy as np

print("\n--- PART A DETAILS (A2, A3, A4) ---")
# A2: Concrete example of HIGH vs LOW TF-IDF weights in a single document
sample_idx = 0
for idx, text in enumerate(corpus):
    if len(text.split()) >= 6:
        sample_idx = idx
        break

sample_doc = corpus[sample_idx]
doc_vec = tfidf_vec.transform([sample_doc])
feature_names = np.array(tfidf_vec.get_feature_names_out())
nonzero_cols = doc_vec.nonzero()[1]
scores = list(zip(feature_names[nonzero_cols], doc_vec.data))
scores.sort(key=lambda x: x[1])

print(f"Sample Doc [{sample_idx}]: '{sample_doc}'")
print("Lowest TF-IDF weights (common words):", scores[:3])
print("Highest TF-IDF weights (distinctive words):", scores[-3:])

# A3: True N-gram vocabulary growth & sparsity (without max_features cap)
for n in range(1, 4):
    ng_vec = CountVectorizer(ngram_range=(n, n))
    ng_mat = ng_vec.fit_transform(corpus)
    ng_sparsity = (1.0 - (ng_mat.nnz / (ng_mat.shape[0] * ng_mat.shape[1]))) * 100
    print(f"N-gram ({n},{n}) -> Vocab Size: {ng_mat.shape[1]} | Sparsity: {ng_sparsity:.4f}%")

# A4: Tiny 3-document toy corpus for hand-worked TF-IDF & Bigram verification
toy_docs = [
    "good game play",
    "bad game",
    "good play"
]
toy_bigram = CountVectorizer(ngram_range=(2, 2))
toy_bi_mat = toy_bigram.fit_transform(toy_docs)
print("\nToy Bigram Features:", toy_bigram.get_feature_names_out().tolist())
print("Toy Bigram Matrix:\n", toy_bi_mat.toarray())

toy_tfidf = TfidfVectorizer()
toy_tf_mat = toy_tfidf.fit_transform(toy_docs)
print("Toy TF-IDF Features:", toy_tfidf.get_feature_names_out().tolist())
print("Toy TF-IDF Matrix (rounded):\n", np.round(toy_tf_mat.toarray(), 4))


print("\n--- PART C: PROBING & COMPARISON ---")
# C1: Top-5 similar words across all 3 embedding models
query_words = ["borderland", "game", "play", "love"]
for word in query_words:
    if word in cbow_model.wv:
        print(f"\nQuery: '{word}'")
        print("  CBOW     :", [w for w, _ in cbow_model.wv.most_similar(word, topn=5)])
        print("  Skip-gram:", [w for w, _ in skipgram_model.wv.most_similar(word, topn=5)])
        print("  fastText :", [w for w, _ in ft_model.wv.most_similar(word, topn=5)])

# C2: Cosine similarity (Expected Similar vs Unrelated)
sim_pair = ("game", "play")
unrel_pair = ("game", "coffee")
if all(w in skipgram_model.wv for w in sim_pair + unrel_pair):
    print(f"\nCosine Similarity (Similar {sim_pair}):")
    print(f"  CBOW: {cbow_model.wv.similarity(*sim_pair):.4f} | Skip-gram: {skipgram_model.wv.similarity(*sim_pair):.4f} | fastText: {ft_model.wv.similarity(*sim_pair):.4f}")
    print(f"Cosine Similarity (Unrelated {unrel_pair}):")
    print(f"  CBOW: {cbow_model.wv.similarity(*unrel_pair):.4f} | Skip-gram: {skipgram_model.wv.similarity(*unrel_pair):.4f} | fastText: {ft_model.wv.similarity(*unrel_pair):.4f}")

# C3: Analogy test (a - b + c ≈ ?) -> e.g., xbox - microsoft + sony
analogy_words = ["xbox", "microsoft", "sony"]
if all(w in skipgram_model.wv for w in analogy_words):
    print("\nAnalogy Test ('xbox' - 'microsoft' + 'sony'):")
    print("  CBOW     :", cbow_model.wv.most_similar(positive=["xbox", "sony"], negative=["microsoft"], topn=3))
    print("  Skip-gram:", skipgram_model.wv.most_similar(positive=["xbox", "sony"], negative=["microsoft"], topn=3))
    print("  fastText :", ft_model.wv.most_similar(positive=["xbox", "sony"], negative=["microsoft"], topn=3))





    import os
import joblib

os.makedirs("models", exist_ok=True)

# D1: Save scikit-learn vectorizers and Gensim models
joblib.dump(count_vec, "models/bow_vectorizer.joblib")
joblib.dump(tfidf_vec, "models/tfidf_vectorizer.joblib")
cbow_model.save("models/word2vec_cbow.model")
skipgram_model.save("models/word2vec_skipgram.model")
ft_model.save("models/fasttext.model")

print("\n--- PART D1: All vectorizers and embedding models saved to 'models/' ---")