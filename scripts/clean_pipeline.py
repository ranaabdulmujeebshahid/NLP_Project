import os
import re
import unicodedata
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK datasets
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer and stop words (preserving negations like 'not', 'no')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
negations = {'not', 'no', 'never', 'nor', 'none'}
stop_words = stop_words - negations

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Unicode Normalization (NFC)
    text = unicodedata.normalize('NFC', text)
    
    # 2. Noise Removal (URLs, Mentions, HTML tags, special symbols via Regex)
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'@\w+', '', text)                  # Remove user mentions
    text = re.sub(r'#\w+', '', text)                  # Remove hashtags
    text = re.sub(r'<.*?>', '', text)                 # Remove HTML tags
    
    # 3. Case Folding (Convert to lowercase to standardize text features)
    text = text.lower()
    
    # 4. Tokenization & Filtering (Stopwords & Punctuation/Numbers)
    # Basic word tokenization using regex to extract alphabetic words
    tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text)
    
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    # 5. Lemmatization (Choose lemmatization over stemming to preserve valid dictionary base forms)
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    
    return " ".join(lemmatized_tokens)

# Load dataset
data_path = os.path.join('data', 'twitter_training.csv')
df = pd.read_csv(data_path, header=None, names=['ID', 'Topic', 'Sentiment', 'Text'])
df = df.dropna(subset=['Text'])

# Calculate Vocabulary Size BEFORE cleaning (on sample)
sample_df = df.sample(n=5000, random_state=42)
raw_vocab = set(" ".join(sample_df['Text'].astype(str)).split())

# Apply cleaning pipeline
print("Running cleaning pipeline on dataset...")
df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Calculate Vocabulary Size AFTER cleaning (on same sample mapped)
cleaned_sample = df.sample(n=5000, random_state=42)
cleaned_vocab = set(" ".join(cleaned_sample['Cleaned_Text'].astype(str)).split())

print(f"\n--- VOCABULARY SIZE COMPARISON ---")
print(f"Vocabulary Size BEFORE Cleaning: {len(raw_vocab)}")
print(f"Vocabulary Size AFTER Cleaning:  {len(cleaned_vocab)}")

# Save the cleaned dataset for Phase 2
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'cleaned_twitter_data.csv')
df[['ID', 'Topic', 'Sentiment', 'Cleaned_Text']].to_csv(output_path, index=False)
print(f"\nCleaned dataset successfully saved to: {output_path}")