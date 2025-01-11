from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import ParameterGrid, cross_val_score
import joblib
from preprocess import prepare_dataset
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
from torch import cuda

def check_gpu():
    """Check if CUDA GPU is available."""
    if cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("No GPU available, using CPU instead")
        return False

def create_model():
    """Create an optimized sentiment classification pipeline."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,  # Increased from 5000
            ngram_range=(1, 3),  # Changed from (1,2) to capture longer phrases
            min_df=2,
            max_df=0.95,
            analyzer='char_wb',  # Add character n-grams for Turkish
            strip_accents='unicode'
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=200,    # Increased from 100
            max_depth=30,        # Increased from 20
            min_samples_split=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=0
        ))
    ])

def train_model():
    """Train the sentiment classification model with optimized parameters."""
    start_time = time.time()
    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU availability
    has_gpu = check_gpu()
    
    # Prepare data
    print("\nPreparing dataset...")
    train_df, test_df = prepare_dataset()
    print(f"Dataset loaded: {len(train_df)} training samples, {len(test_df)} test samples")
    
    # Create base model
    model = create_model()
    
    # Reduced parameter grid for faster training
    param_grid = {
        'tfidf__max_features': [3000, 5000],  # Reduced options
        'tfidf__ngram_range': [(1, 2)],  # Single option
        'classifier__n_estimators': [50, 100],  # Reduced options
        'classifier__max_depth': [10, 20],  # Removed None option
        'classifier__min_samples_split': [2]  # Single option
    }
    
    # Get all parameter combinations
    param_list = list(ParameterGrid(param_grid))
    total_iterations = len(param_list) * 3  # Reduced to 3-fold cross-validation
    
    print(f"\nTotal parameter combinations: {len(param_list)}")
    print("Each combination will be evaluated with 3-fold cross-validation")
    print(f"Total evaluations: {total_iterations}")
    
    # Initialize progress bar
    pbar = tqdm(total=total_iterations, desc="Training Progress")
    
    best_score = -1
    best_params = None
    best_model = None
    
    try:
        # Manual grid search with progress bar
        for params in param_list:
            # Set parameters
            model.set_params(**params)
            
            # Perform cross-validation
            scores = cross_val_score(
                model, 
                train_df['cleaned_text'], 
                train_df['sentiment'],
                cv=3,  # Reduced from 5 to 3
                scoring='f1_weighted',
                n_jobs=-1 if not has_gpu else 1  # Use single job for GPU
            )
            
            # Update progress bar and show current parameters
            pbar.set_postfix({
                'max_features': params['tfidf__max_features'],
                'n_estimators': params['classifier__n_estimators'],
                'max_depth': params['classifier__max_depth'],
                'current_score': f"{np.mean(scores):.3f}"
            })
            pbar.update(3)  # Update for 3-fold CV
            
            # Track best parameters
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_model = model.set_params(**params)
                print(f"\nNew best score: {mean_score:.3f} with params: {params}")
    
    finally:
        pbar.close()
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_model.fit(train_df['cleaned_text'], train_df['sentiment'])
    
    print("\nBest parameters:", best_params)
    print("Best cross-validation score:", best_score)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions = best_model.predict(test_df['cleaned_text'])
    print("\nModel Performance on Test Set:")
    print(classification_report(test_df['sentiment'], predictions))
    
    # Save the best model
    print("\nSaving model...")
    joblib.dump(best_model, 'sentiment_model.pkl')
    print("Best model saved as 'sentiment_model.pkl'")
    
    # Calculate and display training duration
    end_time = time.time()
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print("\n" + "="*50)
    print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {hours}h {minutes}m {seconds}s")
    print("="*50 + "\n")

def predict_sentiment(text, return_proba=False):
    """Predict sentiment for a given text with probability scores."""
    model = joblib.load('sentiment_model.pkl')
    if return_proba:
        probas = model.predict_proba([text])[0]
        classes = model.classes_
        return {class_: prob for class_, prob in zip(classes, probas)}
    return model.predict([text])[0]

if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {str(e)}") 