import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
import json
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
import warnings

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=96):
        # Convert pandas Series to list for proper indexing
        self.texts = texts.values if hasattr(texts, 'values') else texts
        self.labels = labels.values if hasattr(labels, 'values') else labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SentimentClassifier:
    def __init__(self, model_name="dbmdz/bert-base-turkish-cased", num_labels=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_metrics = []
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler('cuda')
        
    def prepare_dataloader(self, texts, labels, batch_size=32, shuffle=True):
        dataset = SentimentDataset(texts, labels, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Single worker for stability
            pin_memory=True  # Better GPU memory handling
        )

    def train_model(self, train_df, val_df, epochs=3, batch_size=32, learning_rate=2e-5):
        print(f"Using device: {self.device}")
        print("Mixed precision training enabled")
        
        # Convert sentiment labels to numeric values
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        train_labels = train_df['sentiment'].map(sentiment_map).values
        val_labels = val_df['sentiment'].map(sentiment_map).values
        
        # Prepare data
        train_loader = self.prepare_dataloader(
            train_df['cleaned_text'],
            train_labels,
            batch_size=batch_size
        )
        
        val_loader = self.prepare_dataloader(
            val_df['cleaned_text'],
            val_labels,
            batch_size=batch_size * 2,
            shuffle=False
        )

        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Training loop
        best_val_f1 = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            train_pbar = tqdm(train_loader, desc="Training")
            
            for batch in train_pbar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Use mixed precision training
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss

                # Scale loss and backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update weights with scaled gradients
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                
                total_train_loss += loss.item()
                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_train_loss = total_train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            val_metrics = self.evaluate(val_loader)
            self.val_metrics.append(val_metrics)
            
            print(f"\nAverage training loss: {avg_train_loss:.4f}")
            print(f"Validation Metrics:")
            print(f"F1-Score: {val_metrics['weighted_f1']:.4f}")
            print(f"Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['weighted_f1'] > best_val_f1:
                best_val_f1 = val_metrics['weighted_f1']
                self.save_model('best_model')
                print("New best model saved!")
        
        # Training complete
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        
        # Save training history
        self.save_training_history()
        
        return self.val_metrics[-1]

    def evaluate(self, dataloader):
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad(), autocast():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                actual_labels.extend(labels.numpy())

        # Calculate metrics
        report = classification_report(
            actual_labels,
            predictions,
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        )
        
        return {
            'accuracy': report['accuracy'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'class_metrics': {
                'negative': report['negative'],
                'neutral': report['neutral'],
                'positive': report['positive']
            }
        }

    def predict(self, texts, batch_size=32):
        """Predict sentiments for texts efficiently."""
        self.model.eval()  # Ensure eval mode
        torch.set_grad_enabled(False)  # Disable gradients globally
        
        dataloader = self.prepare_dataloader(
            texts,
            [0] * len(texts),  # Dummy labels
            batch_size=batch_size,
            shuffle=False
        )
        
        predictions = []
        probabilities = []
        
        # Process batches efficiently
        for batch in dataloader:
            # Move to GPU in half precision
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Get predictions
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Process on GPU
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            
            # Move to CPU only at the end
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

        # Convert numeric predictions to labels
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predictions = [label_map[pred] for pred in predictions]
        
        return predictions, probabilities

    def save_model(self, path='sentiment_model'):
        """Save the model, tokenizer, and configuration."""
        Path(path).mkdir(exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save model info
        model_info = {
            'device': str(self.device),
            'date_saved': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(f"{path}/model_info.json", 'w') as f:
            json.dump(model_info, f, indent=4)

    def save_training_history(self):
        """Save training metrics history."""
        history = {
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=4)

    @classmethod
    def load_model(cls, path='sentiment_model'):
        """Load a saved model."""
        classifier = cls()
        # Load with warnings suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier.model = AutoModelForSequenceClassification.from_pretrained(
                path,
                local_files_only=True  # Avoid looking for updates
            )
            classifier.tokenizer = AutoTokenizer.from_pretrained(
                path,
                local_files_only=True  # Avoid looking for updates
            )
        classifier.model.to(classifier.device)
        return classifier 