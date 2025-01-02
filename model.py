from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from preprocess import prepare_dataset

def create_model():
    """Create a simple sentiment classification pipeline."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', LogisticRegression(random_state=42))
    ])

def train_model():
    """Train the sentiment classification model."""
    # Prepare data
    train_df, test_df = prepare_dataset()
    
    # Create and train the model
    model = create_model()
    model.fit(train_df['cleaned_text'], train_df['sentiment'])
    
    # Evaluate the model
    predictions = model.predict(test_df['cleaned_text'])
    print("\nModel Performance:")
    print(classification_report(test_df['sentiment'], predictions))
    
    # Save the model
    joblib.dump(model, 'sentiment_model.pkl')
    print("\nModel saved as 'sentiment_model.pkl'")

def predict_sentiment(text):
    """Predict sentiment for a given text."""
    model = joblib.load('sentiment_model.pkl')
    return model.predict([text])[0]

if __name__ == "__main__":
    train_model() 