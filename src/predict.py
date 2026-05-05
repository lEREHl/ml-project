import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Ensure perfectly identical preprocessing behavior
from src.preprocessing import process_single_review

def predict_sentiment(text):
    """
    Predicts sentiment for a given textual sequence in real-time.
    Dynamically accesses the previously validated best model and vectorizers locally.
    
    Returns: 'Positive' or 'Negative'
    """
    try:
        model = joblib.load('Models/MRSA_best_model.pkl')
        vectorizer = joblib.load('Models/MRSA_vectorizer.pkl')
    except (FileNotFoundError, Exception) as e:
        raise RuntimeError("Best model or vectorizer not found. Please run the training pipeline first.") from e
        
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    stop_words_set = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    
    # 1. Preprocess the raw input using identical logic 
    cleaned_text = process_single_review(text, stop_words_set, stemmer)
    
    # 2. Vectorize the string directly using our persisted Count/TF-IDF vectorizer configuration
    X_features = vectorizer.transform([cleaned_text]).toarray()
    
    # 3. Execute Prediction
    prediction = model.predict(X_features)[0]
    
    # 4. Human Readable Decoding
    result = "Positive" if prediction == 1 else "Negative"
    return result
