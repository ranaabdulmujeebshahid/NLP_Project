import os
import pandas as pd
from collections import Counter
import nltk

# Download necessary NLTK data if not already present
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# 1. Load the dataset
# Adjust the filename below if your Kaggle CSV has a different name
data_path = os.path.join('data', 'twitter_training.csv')

print("Loading dataset...")
# Note: The Kaggle Twitter Entity Sentiment dataset typically doesn't have header rows. 
# Columns are usually: [ID, Topic, Sentiment, Text]
df = pd.read_csv(data_path, header=None, names=['ID', 'Topic', 'Sentiment', 'Text'])

# Drop rows with missing text
df = df.dropna(subset=['Text'])

# 2. Basic Statistics
num_documents = len(df)
print(f"\n--- BASIC STATISTICS ---")
print(f"Number of documents (rows): {num_documents}")

# Target distribution
print("\nTarget (Sentiment) Distribution:")
print(df['Sentiment'].value_counts())

# Document lengths in words
df['Word_Count'] = df['Text'].astype(str).apply(lambda x: len(x.split()))
avg_doc_length = df['Word_Count'].mean()
print(f"\nAverage document length (in words): {avg_doc_length:.2f}")

# 3. Vocabulary Size, Tokens vs Types (on a sample for speed)
sample_texts = df['Text'].astype(str).sample(n=min(5000, num_documents), random_state=42)
all_tokens = []
for text in sample_texts:
    tokens = word_tokenize(text.lower())
    all_tokens.extend(tokens)

total_tokens = len(all_tokens)
vocabulary = set(all_tokens)
vocabulary_size = len(vocabulary)

print(f"\n--- TOKENS vs TYPES (Sample of {len(sample_texts)} docs) ---")
print(f"Total Tokens: {total_tokens}")
print(f"Vocabulary Size (Types): {vocabulary_size}")
print(f"Type-Token Ratio (TTR): {vocabulary_size / total_tokens:.4f}")

# Zipf's Law insight
word_counts = Counter(all_tokens)
print("\nTop 10 most frequent words (Zipf's head):")
print(word_counts.most_common(10))

# 4. Show 3-5 real examples of messy text
print(f"\n--- MESSY TEXT EXAMPLES ---")
messy_samples = df['Text'].head(5).tolist()
for i, text in enumerate(messy_samples, 1):
    print(f"Example {i}: {text}")