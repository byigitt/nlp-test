import os
import warnings
from model_transformer import SentimentClassifier
from preprocess import clean_text
import sys
import json
import torch

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer warnings

# Global model instance for faster prediction
_model = None

def get_model():
    """Get or initialize the model."""
    global _model
    if _model is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _model = SentimentClassifier.load_model('best_model')
            _model.model.eval()  # Set to evaluation mode
            torch.set_grad_enabled(False)  # Disable gradients globally
    return _model

def predict_sentiment(text, return_proba=False):
    """Predict sentiment for a given text."""
    # Get cached model
    classifier = get_model()
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Get prediction and probabilities
    predictions, probabilities = classifier.predict([cleaned_text])
    prediction = predictions[0]
    probs = probabilities[0]
    
    if return_proba:
        return {
            'prediction': prediction,
            'probabilities': {
                'negative': float(probs[0]),
                'neutral': float(probs[1]),
                'positive': float(probs[2])
            }
        }
    return prediction

def batch_predict(texts, batch_size=32):
    """Predict sentiments for multiple texts efficiently."""
    classifier = get_model()
    cleaned_texts = [clean_text(text) for text in texts]
    
    predictions, probabilities = classifier.predict(cleaned_texts, batch_size=batch_size)
    
    results = []
    for pred, probs in zip(predictions, probabilities):
        results.append({
            'prediction': pred,
            'probabilities': {
                'negative': float(probs[0]),
                'neutral': float(probs[1]),
                'positive': float(probs[2])
            }
        })
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <text>")
        sys.exit(1)
    
    text = " ".join(sys.argv[1:])
    result = predict_sentiment(text, return_proba=True)
    
    print("\nInput Text:", text)
    print("\nPrediction:", result['prediction'].upper())
    print("\nConfidence Scores:")
    for sentiment, prob in result['probabilities'].items():
        print(f"{sentiment.capitalize()}: {prob:.3f}")

if __name__ == "__main__":
    main() 