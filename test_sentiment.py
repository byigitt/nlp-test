from model import predict_sentiment
from preprocess import clean_text, prepare_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

def analyze_sentiment_batch(texts, model=None, batch_size=100):
    """Analyze sentiments for a batch of texts."""
    if model is None:
        model = joblib.load('sentiment_model.pkl')
    
    # Clean all texts at once using vectorization
    cleaned_texts = pd.Series(texts).apply(clean_text)
    
    # Predict in batches
    predictions = []
    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i:i + batch_size]
        predictions.extend(model.predict(batch))
    
    return predictions

def test_model(n_samples=10000):
    """Test the model with sample reviews and generate metrics."""
    print("Loading model and test data...")
    model = joblib.load('sentiment_model.pkl')
    _, test_df = prepare_dataset()
    
    # Sample reviews for individual testing
    sample_reviews = [
        "Ürün harika, çok memnun kaldım",
        "Fiyatına göre idare eder",
        "Berbat bir ürün, asla tavsiye etmem"
    ]
    
    print("\nTesting individual reviews:")
    print("-" * 50)
    for review in sample_reviews:
        sentiment = predict_sentiment(review)
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment}")
    
    # Batch testing on test set
    n_samples = min(n_samples, len(test_df))
    print(f"\nTesting {n_samples} reviews...")
    if n_samples < len(test_df):
        test_df = test_df.sample(n_samples, random_state=42)
    
    print("Analyzing reviews in batches...")
    predictions = analyze_sentiment_batch(test_df['cleaned_text'].values, model, batch_size=100)
    
    # Generate and save confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(test_df['sentiment'], predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_df['sentiment'], predictions))
    
    print("\nConfusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    test_model() 