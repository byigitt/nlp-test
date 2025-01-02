import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the JSON data and convert to DataFrame."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all reviews
    all_reviews = []
    for product in data:
        for review in product['reviews']:
            all_reviews.append({
                'text': review['review'],
                'rating': review['star'],
                'product_name': product['name']
            })
    
    return pd.DataFrame(all_reviews)

def clean_text(text):
    """Basic text cleaning function."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def assign_sentiment(rating):
    """Convert star rating to sentiment categories."""
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positive'

def prepare_dataset(file_path='yorumlar.json'):
    """Prepare the complete dataset."""
    # Load data
    df = load_data(file_path)
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Assign sentiment labels
    df['sentiment'] = df['rating'].apply(assign_sentiment)
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = prepare_dataset()
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print("\nSentiment distribution in training set:")
    print(train_df['sentiment'].value_counts()) 