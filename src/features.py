import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def feature_extraction(dataset, text_col, label_col, method='bow', ngram_range=(1, 1), max_features=2000):
    print(f"Extracting features using method: {method.upper()} with ngram_range: {ngram_range}...")
    X = np.array(dataset[text_col].values)
    y = np.array(dataset[label_col].values)
    
    if method == 'bow':
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    else:
        raise ValueError("Method must be 'bow' or 'tfidf'")
        
    X_transformed = vectorizer.fit_transform(X).toarray()
    
    import os
    import joblib
    os.makedirs("Models", exist_ok=True)
    joblib.dump(vectorizer, 'Models/MRSA_vectorizer.pkl')
    
    print(f"=== {method.upper()} Features ===")
    print(f"X shape : {X_transformed.shape}")
    print(f"y shape : {y.shape}\n")
    
    return X_transformed, y, vectorizer
