import nltk
from src.data import load_data
from src.preprocessing import preprocess_text
from src.features import feature_extraction
from src.models import train_models
from src.tune import tune_hyperparameters

# --- CONFIGURATION PIPELINE ---
DATASET_PATH = 'Dataset/IMDB.csv'
TEXT_COLUMN = 'review'
LABEL_COLUMN = 'sentiment'

FEATURE_METHOD = 'bow'  # Options: 'bow' or 'tfidf'
NGRAM_RANGE = (1, 1)    # Options: (1, 1) for unigrams, (1, 2) for bigrams
MAX_FEATURES = 2000     # Default: 2000
RUN_TUNING = False      # Set to True to execute lengthy GridSearchCV phase

def main():
    print("Starting ML Pipeline...")
    
    # Ensure nltk packages are downloaded once before running
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    # 1. Load Data
    print("\n--- Phase 1: Data Loading ---")
    dataset = load_data(DATASET_PATH, TEXT_COLUMN, LABEL_COLUMN)
    
    # 2. Preprocess Text
    print("\n--- Phase 2: Text Preprocessing ---")
    dataset = preprocess_text(dataset, TEXT_COLUMN)
    
    # 3. Validation & Hyperparameter Tuning (Optional)
    if RUN_TUNING:
        print("\n--- Phase 3.5: Hyperparameter Tuning & Selection ---")
        best_pipelines = tune_hyperparameters(dataset, TEXT_COLUMN, LABEL_COLUMN)
        print("Tuning complete. You can update your CONFIG arrays with the best parameters above.")
        
    # 4. Feature Extraction
    print("\n--- Phase 3: Feature Extraction ---")
    X, y, cv = feature_extraction(
        dataset, 
        TEXT_COLUMN,
        LABEL_COLUMN,
        method=FEATURE_METHOD, 
        ngram_range=NGRAM_RANGE, 
        max_features=MAX_FEATURES
    )
    
    # 5. Train Models
    print("\n--- Phase 4: Model Training ---")
    models = train_models(X, y, dataset[TEXT_COLUMN].tolist())
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
