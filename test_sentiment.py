from model import predict_sentiment
from preprocess import clean_text, prepare_dataset
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_sentiment(text):
    """Analyze the sentiment of a given text."""
    cleaned = clean_text(text)
    sentiment = predict_sentiment(cleaned)
    return sentiment

def bulk_test(n_samples=10000):
    """Test the model on a large number of reviews."""
    print(f"\nTesting {n_samples} reviews...")
    
    # Get test data
    _, test_df = prepare_dataset()
    
    # Sample n reviews
    if len(test_df) > n_samples:
        test_df = test_df.sample(n=n_samples, random_state=42)
    else:
        print(f"Note: Only {len(test_df)} reviews available in test set")
    
    # Predict sentiments
    predictions = []
    for text in tqdm(test_df['cleaned_text'], desc="Analyzing reviews"):
        predictions.append(predict_sentiment(text))
    
    # Calculate and display metrics
    print("\nClassification Report:")
    print(classification_report(test_df['sentiment'], predictions))
    
    # Create confusion matrix
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
    
    # Sample misclassified reviews
    results_df = pd.DataFrame({
        'text': test_df['text'],
        'true_sentiment': test_df['sentiment'],
        'predicted_sentiment': predictions
    })
    
    misclassified = results_df[results_df['true_sentiment'] != results_df['predicted_sentiment']]
    if len(misclassified) > 0:
        print("\nSample of Misclassified Reviews:")
        sample_size = min(5, len(misclassified))
        for _, row in misclassified.sample(n=sample_size).iterrows():
            print(f"\nText: {row['text']}")
            print(f"True Sentiment: {row['true_sentiment']}")
            print(f"Predicted Sentiment: {row['predicted_sentiment']}")

if __name__ == "__main__":
    # Test individual reviews
    test_reviews = [
        "Ürün harika, çok memnun kaldım",
        "Fiyatına göre idare eder",
        "Berbat bir ürün, asla tavsiye etmem"
    ]
    
    print("Testing individual reviews:")
    print("-" * 50)
    for review in test_reviews:
        sentiment = analyze_sentiment(review)
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment}")
    
    # Run bulk testing
    bulk_test(10000) 