import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Movie Reviews Sentiment Analysis", page_icon="🎬", layout="centered")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("A machine learning pipeline that classifies movie reviews as either **Positive** or **Negative**.")

# Safely import the modular predict_sentiment method we built
try:
    from src.predict import predict_sentiment
except Exception as e:
    st.error(f"Failed to import core modules. Please ensure you are running this from the project root. Error: {e}")

# --- 1. Test the Model ---
st.header("1. Test the Model")
user_review = st.text_area("Enter your movie review here:", "The cinematography was beautiful, but the plot was honestly so boring!")

if st.button("Predict Sentiment"):
    if user_review.strip():
        with st.spinner("Analyzing sentiment..."):
            try:
                prediction = predict_sentiment(user_review)
                if prediction == "Positive":
                    st.success("🟢 **Positive Sentiment Detected!**")
                else:
                    st.error("🔴 **Negative Sentiment Detected!**")
            except Exception as e:
                st.error(f"Prediction Failed: {e}. Please ensure the models are successfully trained and exist in /Models.")
    else:
        st.warning("Please enter a review first.")

st.divider()

# --- 2. Model Evaluation & Comparison ---
st.header("2. Model Evaluation & Comparison")
try:
    # Load the evaluation results generated during Phase 4 Model Training
    df_results = pd.read_csv("Models/evaluation_results.csv")
    
    st.subheader("Comparison Table")
    st.dataframe(df_results.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], color='lightgreen'))
    
    st.subheader("Performance Chart (F1 Score)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df_results, x='F1 Score', y='Model', hue='Category', ax=ax, palette="viridis")
    ax.set_xlim(0.6, 1.0) # Scale appropriately for clear visual comparison
    plt.tight_layout()
    st.pyplot(fig)
except FileNotFoundError:
    st.info("⚠️ `evaluation_results.csv` not found. Please manually run `python movie_reviews_sentiment_analysis.py` to train models and generate the evaluation metrics before starting the UI.")

st.divider()

# --- 3. Explanation Section ---
st.header("3. How It Works Internally")
st.write("""
When you click **Predict Sentiment**, the following steps automatically occur under the hood:
1. **Text Preprocessing**: Your raw text is stripped of HTML tags and special characters, and converted to lowercase. Stop words (like 'the', 'is') are removed, and words are stemmed to their root form (e.g., 'running' -> 'run').
2. **Feature Extraction (Vectorization)**: The cleaned text is converted into a mathematical vector representation using the same logic the training pipeline used (Bag-of-Words or TF-IDF), identifying the frequency and presence of key vocabulary.
3. **Model Prediction**: This numerical array is fed into our globally **Best Performing Machine Learning Model** (determined autonomously during the training phase). The model mathematically computes a decision boundary and classifies the array as exactly `1` (Positive) or `0` (Negative).
""")
