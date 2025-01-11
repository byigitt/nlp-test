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
    """Enhanced text cleaning function."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace common Turkish characters
    text = text.replace('ı', 'i').replace('ğ', 'g').replace('ş', 's') \
               .replace('ü', 'u').replace('ö', 'o').replace('ç', 'c')
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Add: Remove repeated characters (e.g., çokkk -> çok)
    text = re.sub(r'(.)\1+', r'\1\1', text)
    
    # Add: Remove specific product references
    text = re.sub(r'\b(ürün|urun)\b', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove short words but keep important ones
    important_words = {'yok', 'iyi', 'kötü', 'eh'}
    text = ' '.join([w for w in text.split() if len(w) > 2 or w in important_words])
    
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
    """Prepare the complete dataset with improved balancing."""
    df = load_data(file_path)
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Assign sentiment labels
    df['sentiment'] = df['rating'].apply(assign_sentiment)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Improved balancing strategy
    min_class_size = df['sentiment'].value_counts().min()
    max_samples = min_class_size * 3  # Allow up to 3x the minimum class size
    
    balanced_df = pd.concat([
        df[df['sentiment'] == label].sample(
            n=min(len(df[df['sentiment'] == label]), max_samples),
            random_state=42
        )
        for label in df['sentiment'].unique()
    ])
    
    # Add text length as a feature
    balanced_df['text_length'] = balanced_df['cleaned_text'].str.len()
    
    # Split with stratification
    train_df, test_df = train_test_split(
        balanced_df, 
        test_size=0.2, 
        random_state=42,
        stratify=balanced_df['sentiment']
    )
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = prepare_dataset()
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print("\nSentiment distribution in training set:")
    print(train_df['sentiment'].value_counts()) 