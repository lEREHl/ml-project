import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

def train_models(X, y, raw_texts):
    print("Training models...")
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        X, y, raw_texts, test_size=0.2, random_state=9
    )
    print(f"Train shapes : X = {X_train.shape}, y = {y_train.shape}")
    print(f"Test shapes  : X = {X_test.shape},  y = {y_test.shape}\n")

    baseline_models = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0, fit_prior=True),
        "Bernoulli Naive Bayes": BernoulliNB(alpha=1.0, fit_prior=True)
    }

    comparative_models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Linear SVM": LinearSVC(max_iter=500, dual=False)
    }
    
    os.makedirs("Models", exist_ok=True)
    trained_models = {}
    evaluation_results = []

    print("## Model Evaluation ##")
    
    all_models = {"BASELINE": baseline_models, "COMPARATIVE": comparative_models}
    
    for category, models in all_models.items():
        print(f"\n--- {category} MODELS ---")
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Save model
            filename = f"Models/MRSA_{name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, filename)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='binary')
            rec = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            cm = confusion_matrix(y_test, y_pred)
            
            # K-Fold Cross Validation for stability
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            
            print(f"-> {name} Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
            print(f"   5-Fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"   Confusion Matrix:\n{cm}")
            
            trained_models[name] = model
            evaluation_results.append({
                "Category": category,
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            })

    # Create comparison table
    df_results = pd.DataFrame(evaluation_results)
    print("\n================ FINAL MODEL COMPARISON ================")
    print(df_results.to_string(index=False))
    print("========================================================\n")
    
    # Identify the best model based on F1 Score
    best_model_row = df_results.loc[df_results['F1 Score'].idxmax()]
    best_model_name = best_model_row['Model']
    best_model = trained_models[best_model_name]
    
    # Persist the definitively best model separately for easy downstream access
    joblib.dump(best_model, 'Models/MRSA_best_model.pkl')
    
    print(f"BEST MODEL (By F1 Score): {best_model_name} from {best_model_row['Category']} with F1 = {best_model_row['F1 Score']:.4f}")

    # ================= ERROR ANALYSIS =================
    print("\n================ ERROR ANALYSIS ================")
    y_pred_best = best_model.predict(X_test)
    misclassified_indices = np.where(y_test != y_pred_best)[0]
    
    print(f"Total misclassified samples by {best_model_name}: {len(misclassified_indices)} out of {len(y_test)}")
    
    print("\n--- Why Predictions Typically Fail ---")
    print("1. Sarcasm / Irony: Simple bag-of-words or linear models struggle to detect sarcasm.")
    print("2. Complex Negation: 'It was certainly not the worst movie ever' might be incorrectly flagged as negative due to 'worst' or 'not'.")
    print("3. Nuanced/Mixed Reviews: Users listing pros and cons evenly often confuse straightforward classifiers.")
    
    print("\n--- Sample Misclassifications ---")
    for idx in misclassified_indices[:3]: # Display exactly 3 clear examples
        true_label = "Positive" if y_test[idx] == 1 else "Negative"
        pred_label = "Positive" if y_pred_best[idx] == 1 else "Negative"
        text_snippet = str(texts_test[idx])[:250] + "..."
        
        print(f"-> True: {true_label} | Predicted: {pred_label}")
        print(f"   Review: {text_snippet}\n")
    print("================================================\n")

    return trained_models
