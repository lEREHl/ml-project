# Implementation Details

## 1. Project Overview

This project implements a **sentiment analysis system** that classifies text reviews as **positive** or **negative**.

The system follows a complete machine learning pipeline and includes an interactive UI built using Streamlit.

Naive Bayes is used as the **baseline model**, and its performance is compared with other models.

---

## 2. System Architecture

Raw Text → Preprocessing → Feature Extraction → Model Training → Evaluation → Prediction → UI

The system is modular and reusable across datasets.

---

## 3. Dataset Handling

The project supports any CSV dataset with:

* A text column
* A label column

```python
def load_data(file_path, text_col, label_col):
    df = pd.read_csv(file_path)
    return df[text_col], df[label_col]
```

---

## 4. Data Preprocessing

Steps:

* Remove HTML tags
* Remove special characters
* Convert to lowercase
* Tokenization
* Stopword removal
* Stemming / Lemmatization

---

## 5. Feature Engineering

### Bag of Words

* CountVectorizer

### TF-IDF

* TfidfVectorizer

### Enhancements

* N-grams (unigram + bigram)
* Tunable max_features

---

## 6. Models

### Baseline Model

* Naive Bayes:

  * GaussianNB
  * MultinomialNB
  * BernoulliNB

### Comparative Models

* Logistic Regression
* Linear SVM

---

## 7. Training Strategy

* Train-test split
* K-Fold Cross Validation

---

## 8. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 9. Hyperparameter Tuning

* GridSearchCV used for:

  * Vectorizers
  * Models

---

## 10. Model Comparison

All models are evaluated and compared using metrics and cross-validation scores.

---

## 11. Error Analysis

* Identify misclassified reviews
* Analyze failure patterns

---

## 12. Model Saving

* Best model saved using joblib

---

## 13. Prediction Function

```python
def predict_sentiment(text):
    # preprocess → vectorize → predict
    return result
```

---

## 14. User Interface

Built using Streamlit

Features:

* Input text
* Prediction output
* Model comparison
* Charts
* Explanation of workflow

---

## 15. Visualization

* Confusion matrix
* Model comparison graphs

---

## 16. Key Features

* Dataset-independent pipeline
* Naive Bayes baseline
* Model comparison
* Proper evaluation
* Clean modular code
* Interactive UI

---

## 17. Conclusion

This project demonstrates:

* End-to-end ML pipeline
* Model comparison and evaluation
* Practical sentiment analysis system

It evolves from a basic model into a **complete ML application**.
