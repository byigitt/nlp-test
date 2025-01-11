import joblib
from sklearn.metrics import classification_report
from preprocess import prepare_dataset
import pandas as pd
from datetime import datetime

def evaluate_model(model_path='sentiment_model.pkl', save_prefix=''):
    """Evaluate model and save metrics."""
    print(f"\nEvaluating model: {model_path}")
    model = joblib.load(model_path)
    _, test_df = prepare_dataset()
    
    # Get predictions
    predictions = model.predict(test_df['cleaned_text'])
    
    # Generate report
    report = classification_report(test_df['sentiment'], predictions, output_dict=True)
    
    # Save metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    metrics_file = f'metrics_{save_prefix}_{timestamp}.txt'
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(f"Model Evaluation - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"Overall Accuracy: {report['accuracy']:.3f}\n")
        f.write("\nClass-wise Performance:\n")
        for label in ['positive', 'negative', 'neutral']:
            f.write(f"\n{label.upper()}:\n")
            f.write(f"Precision: {report[label]['precision']:.3f}\n")
            f.write(f"Recall: {report[label]['recall']:.3f}\n")
            f.write(f"F1-Score: {report[label]['f1-score']:.3f}\n")
    
    print(f"\nMetrics saved to {metrics_file}")
    return report

if __name__ == "__main__":
    # Evaluate current model
    print("Evaluating current model...")
    current_metrics = evaluate_model(save_prefix='current')
    
    # After training new model with improvements
    print("\nTrain your new model, then run:")
    print("evaluate_model('new_sentiment_model.pkl', 'improved')") 