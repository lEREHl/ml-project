import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def tune_hyperparameters(dataset, text_col, label_col):
    print("Starting GridSearchCV for Vectorizers & Models...")
    print("Executing K-Fold CV pipeline to determine optimal and stable configurations...")
    
    # Grab raw text and labels for full pipeline tuning
    X_raw = dataset[text_col].values
    y = dataset[label_col].values
    
    # 1. Tuning the Baseline Model (Naive Bayes)
    nb_pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB())
    ])
    
    nb_params = {
        'vect__max_features': [1000, 2000],
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__alpha': [0.1, 1.0]
    }
    
    print("\n--- Tuning Baseline Model (MultinomialNB) ---")
    # cv=5 establishes a 5-Fold cross validation natively
    grid_nb = GridSearchCV(nb_pipeline, nb_params, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_nb.fit(X_raw, y)
    
    print(f"Best NB CV-F1 Score: {grid_nb.best_score_:.4f}")
    print(f"Best Configuration: {grid_nb.best_params_}")
    
    # 2. Tuning a Comparative Model (Logistic Regression)
    lr_pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=500))
    ])
    
    lr_params = {
        'vect__max_features': [2000, 3000],
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.1, 1.0, 10.0]
    }
    
    print("\n--- Tuning Comparative Model (Logistic Regression) ---")
    grid_lr = GridSearchCV(lr_pipeline, lr_params, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_lr.fit(X_raw, y)
    
    print(f"Best LR CV-F1 Score: {grid_lr.best_score_:.4f}")
    print(f"Best Configuration: {grid_lr.best_params_}")
    
    return {"NB_Best": grid_nb.best_estimator_, "LR_Best": grid_lr.best_estimator_}
