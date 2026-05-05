import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

def process_single_review(text, stop_words_set, stemmer):
    # 1. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 2. Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # 3. Convert to lowercase
    text = text.lower()
    
    # 4. Tokenization, Remove stopwords, and Stemming
    words = word_tokenize(text)
    return " ".join([stemmer.stem(w) for w in words if w not in stop_words_set])

def preprocess_text(dataset, text_col):
    print(f"Preprocessing text column '{text_col}'... (This may take a minute for 50k+ rows)")
    
    # Ensure resources are available
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    stop_words_set = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    
    # Apply all preprocessing efficiently in a single pass
    dataset[text_col] = dataset[text_col].apply(lambda x: process_single_review(x, stop_words_set, stemmer))
    
    return dataset
