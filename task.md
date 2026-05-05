# Project Tasks & Objectives

## 🎯 Objective

To build a **sentiment analysis system** that classifies text reviews as positive or negative using machine learning, with **Naive Bayes as the baseline model**, and present results through an interactive UI.

---

## 📌 Tasks

### 1. Data Loading

* [x] Load dataset from CSV
* [ ] Ensure text and label columns exist

---

### 2. Data Understanding

* [x] Analyze:
  * Class distribution
  * Sample data

---

### 3. Data Preprocessing

* [x] Clean text:
  * Remove HTML
  * Remove special characters
  * Lowercase
* [x] Apply:
  * Tokenization
  * Stopword removal
  * Stemming / Lemmatization

---

### 4. Feature Engineering

* [x] Implement:
  * Bag of Words
  * TF-IDF
* [x] Add:
  * N-grams
  * Vocabulary tuning

---

### 5. Model Development

#### Baseline:

* [x] Naive Bayes:
  * GaussianNB
  * MultinomialNB
  * BernoulliNB

#### Comparative:

* [x] Logistic Regression
* [x] Linear SVM

---

### 6. Model Training

* [x] Train-test split
* [x] Train all models

---

### 7. Model Evaluation

* [x] Accuracy
* [x] Precision
* [x] Recall
* [x] F1 Score
* [x] Confusion Matrix

---

### 8. Validation

* [x] Apply K-Fold Cross Validation

---

### 9. Hyperparameter Tuning

* [x] Use GridSearchCV
* [x] Optimize models and vectorizers

---

### 10. Model Comparison

* [x] Compare all models
* [x] Select best model

---

### 11. Error Analysis

* [x] Analyze incorrect predictions

---

### 12. Visualization

* [x] Confusion matrix
* [ ] Model comparison charts

---

### 13. Model Saving

* [x] Save best model

---

### 14. Prediction Function

* [x] Implement `predict_sentiment(text)`

---

### 15. UI Development

* [ ] Use Streamlit

Features:

* [ ] Input review
* [ ] Show prediction
* [ ] Display model comparison
* [ ] Show charts
* [ ] Explain workflow

---

### 16. Generalization

* [x] Support any dataset
* [x] Avoid hardcoding

---

## 🏁 Deliverables

* Clean code
* Trained models
* Evaluation results
* Visualizations
* UI
* Documentation

---

## ⭐ Outcome

* Complete ML pipeline
* Naive Bayes baseline
* Model comparison
* Reliable results
* Interactive UI

---

## 🚀 Optional

* Add transformer models (BERT)
* Add dataset upload in UI

---

## ✅ Success Criteria

* Accurate models
* Clear evaluation
* Clean code
* Good UI
* Strong understanding
