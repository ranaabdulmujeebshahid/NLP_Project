import os
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Load data sample
data_path = os.path.join('data', 'twitter_training.csv')
df = pd.read_csv(data_path, header=None, names=['ID', 'Topic', 'Sentiment', 'Text'])
df = df.dropna(subset=['Text'])

# Pick 3 sample sentences showing messy data (mentions, URLs, punctuation)
sample_sentences = [
    "im getting on https://t.co/xyz123 and i will murder @user you all for $50!",
    "I am coming to the #borders and I will kill you all, #game2026",
    "rt @gamer_guy: check out this link http://example.com/game !! 100% cool."
]

print("--- TASK 3.1: TOKENIZATION COMPARISON ---")
for i, text in enumerate(sample_sentences, 1):
    print(f"\nOriginal Sentence {i}: {text}")
    
    # Method 1: Basic Whitespace Tokenization (.split())
    split_tokens = text.split()
    print(f"  .split() tokens: {split_tokens[:8]} ... (Total: {len(split_tokens)})")
    
    # Method 2: NLTK word_tokenize
    nltk_tokens = word_tokenize(text)
    print(f"  NLTK tokens:     {nltk_tokens[:8]} ... (Total: {len(nltk_tokens)})")


print("\n--- TASK 3.2: CUSTOM REGULAR EXPRESSIONS ---")

# Define 3 regex patterns tailored for Twitter data
url_pattern = r'https?://\S+|www\.\S+'
mention_pattern = r'@\w+'
hashtag_pattern = r'#\w+'

test_text = sample_sentences[2]
print(f"Testing on text: '{test_text}'")

# Apply regex
urls_found = re.findall(url_pattern, test_text)
mentions_found = re.findall(mention_pattern, test_text)
hashtags_found = re.findall(hashtag_pattern, test_text)

print(f"  - URLs matched: {urls_found}")
print(f"  - Mentions matched: {mentions_found}")
print(f"  - Hashtags matched: {hashtags_found}")

print("\n--- TASK 3.3: TOKENIZATION CHALLENGES DISCUSSION ---")
print("1. Contractions & Slang: Words like 'im' lack apostrophes, making standard parsing treat them as single tokens rather than 'i' + 'm'.")
print("2. Punctuation & Symbols: Special characters like '@' and '#' stick to words during basic splitting, requiring regex separation.")
print("3. URLs & Links: Web addresses break into meaningless tokens if not stripped out or handled prior to tokenization.")