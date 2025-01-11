from model_transformer import SentimentClassifier
from preprocess import prepare_dataset
import torch
from datetime import datetime
import json

def save_metrics(metrics, filename=None):
    """Save evaluation metrics to a file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"metrics_transformer_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Model Evaluation - {datetime.now().strftime('%Y%m%d_%H%M')}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.3f}\n\n")
        f.write("Class-wise Performance:\n\n")
        
        for class_name, class_metrics in metrics['class_metrics'].items():
            f.write(f"{class_name.upper()}:\n")
            f.write(f"Precision: {class_metrics['precision']:.3f}\n")
            f.write(f"Recall: {class_metrics['recall']:.3f}\n")
            f.write(f"F1-Score: {class_metrics['f1-score']:.3f}\n\n")

def main():
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Load and prepare data
    print("\nPreparing dataset...")
    train_df, test_df = prepare_dataset()
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Initialize model
    print("\nInitializing model...")
    classifier = SentimentClassifier()

    # Training parameters
    params = {
        'epochs': 3,
        'batch_size': 32,
        'learning_rate': 2e-5
    }

    # Train model
    print("\nStarting training with mixed precision...")
    metrics = classifier.train_model(
        train_df,
        test_df,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        learning_rate=params['learning_rate']
    )

    # Save final metrics
    save_metrics(metrics)
    print("\nMetrics saved to file")

    # Save training parameters
    with open('training_params.json', 'w') as f:
        json.dump(params, f, indent=4)
    print("Training parameters saved")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}") 