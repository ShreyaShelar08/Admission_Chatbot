"""
Professional Admission Inquiry Chatbot - Model Training Script
Author: College AI Team
Description: Fine-tunes DistilBERT model for intent classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import config

class IntentDataset(Dataset):
    """Custom PyTorch Dataset for intent classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]).strip()
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(file_path):
    """Load dataset from CSV or JSON file"""
    print(f"Loading data from: {file_path}")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("File must be CSV or JSON format")
    
    # Data validation
    if 'text' not in df.columns or 'intent' not in df.columns:
        raise ValueError("Dataset must have 'text' and 'intent' columns")
    
    # Clean data
    df = df.dropna()
    df['text'] = df['text'].str.strip()
    
    print(f"✓ Loaded {len(df)} samples")
    print(f"✓ Found {df['intent'].nunique()} unique intents")
    print("\nIntent Distribution:")
    print(df['intent'].value_counts())
    
    return df['text'].values, df['intent'].values


def plot_training_history(history, output_dir):
    """Plot and save training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Validation Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to {output_dir}/training_history.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_dir}/confusion_matrix.png")
    plt.close()


def evaluate_model(model, data_loader, device):
    """Evaluate model and return predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)


def train_model(data_path, output_dir=None, epochs=None, batch_size=None):
    """Main training function"""
    
    # Use config values if not provided
    output_dir = output_dir or config.MODEL_DIR
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"🚀 ADMISSION INQUIRY CHATBOT - MODEL TRAINING")
    print(f"{'='*70}\n")
    
    # Load data
    texts, intents = load_data(data_path)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(intents)
    num_labels = len(label_encoder.classes_)
    
    # Save label mapping
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(f'{output_dir}/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Save responses configuration
    with open(f'{output_dir}/responses.json', 'w') as f:
        json.dump(config.RESPONSES, f, indent=2)
    
    print(f"\n✓ Label mapping saved")
    print(f"✓ Responses configuration saved")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Load tokenizer and model
    print(f"\n🔄 Loading {config.MODEL_NAME}...")
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=num_labels
    )
    
    # Create datasets
    train_dataset = IntentDataset(X_train, y_train, tokenizer, config.MAX_LENGTH)
    val_dataset = IntentDataset(X_val, y_val, tokenizer, config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_accuracy = 0
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"📚 TRAINING STARTED")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 70)
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}'
            })
        
        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation", leave=False)
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        
        # Print results
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"✓ New best model saved! (Val Acc: {val_accuracy:.4f})")
        
        print()
    
    # Final evaluation
    print(f"{'='*70}")
    print(f"📊 FINAL EVALUATION")
    print(f"{'='*70}\n")
    
    y_true, y_pred = evaluate_model(model, val_loader, device)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    # Save plots
    print("\n📈 Generating visualizations...")
    plot_training_history(history, output_dir)
    plot_confusion_matrix(y_true, y_pred, label_encoder.classes_, output_dir)
    
    # Save training metadata
    metadata = {
        'model_name': config.MODEL_NAME,
        'num_intents': num_labels,
        'intents': list(label_encoder.classes_),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': config.LEARNING_RATE,
        'best_val_accuracy': float(best_val_accuracy),
        'final_train_accuracy': float(train_accuracy),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device)
    }
    
    with open(f'{output_dir}/training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\n📋 Summary:")
    print(f"   Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"   Model saved to: {output_dir}")
    print(f"   Training visualizations saved")
    print(f"\n🚀 Next Steps:")
    print(f"   1. Run the backend: python backend_v2.py")
    print(f"   2. Open frontend: index_v2.html")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Train using the uploaded dataset
    train_model(
        data_path="New_data_set.csv",
        output_dir='./chatbot_model',
        epochs=5,
        batch_size=16
    )