"""
Training script for improved sentiment analysis model.

This script implements a BERT-based sentiment analysis model specifically
trained for product reviews and aspect-based sentiment analysis.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np


class SentimentDataset(Dataset):
    """Dataset for sentiment analysis training."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            texts: List of input texts (review + aspect pairs)
            labels: List of sentiment labels (0: negative, 1: neutral, 2: positive)
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_training_data(file_path: str) -> Tuple[List[str], List[int]]:
    """
    Load training data from JSONL file.
    
    Args:
        file_path: Path to training data file
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Format: [CLS] review [SEP] aspect [SEP]
            text = f"[CLS] {data['review']} [SEP] {data['aspect']} [SEP]"
            texts.append(text)
            labels.append(data['sentiment_label'])  # 0: negative, 1: neutral, 2: positive
    
    return texts, labels


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_sentiment_model(
    train_file: str,
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "./sentiment_model",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """
    Train the sentiment analysis model.
    
    Args:
        train_file: Path to training data
        model_name: Base model name
        output_dir: Output directory for the trained model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,  # negative, neutral, positive
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2}
    )
    
    # Load training data
    texts, labels = load_training_data(train_file)
    
    # Create dataset
    dataset = SentimentDataset(texts, labels, tokenizer)
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")


def create_sample_training_data(output_file: str):
    """
    Create sample training data for demonstration.
    
    Args:
        output_file: Path to output file
    """
    sample_data = [
        {"review": "Battery life is great", "aspect": "battery_life", "sentiment_label": 2},
        {"review": "The camera is blurry", "aspect": "camera_quality", "sentiment_label": 0},
        {"review": "Screen quality is excellent", "aspect": "display_quality", "sentiment_label": 2},
        {"review": "Design is beautiful", "aspect": "design", "sentiment_label": 2},
        {"review": "Price is too high", "aspect": "value_for_money", "sentiment_label": 0},
        {"review": "Fast shipping", "aspect": "delivery", "sentiment_label": 2},
        {"review": "Good packaging", "aspect": "packaging", "sentiment_label": 2},
        {"review": "Easy to use", "aspect": "usability", "sentiment_label": 2},
        {"review": "Customer service is terrible", "aspect": "customer_service", "sentiment_label": 0},
        {"review": "Great performance", "aspect": "performance", "sentiment_label": 2},
        {"review": "Build quality is poor", "aspect": "build_quality", "sentiment_label": 0},
        {"review": "Worth the money", "aspect": "value_for_money", "sentiment_label": 2},
    ]
    
    with open(output_file, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Sample training data created at {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data file")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./sentiment_model",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--create_sample", action="store_true",
                        help="Create sample training data")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_training_data("sample_sentiment_data.jsonl")
        return
    
    train_sentiment_model(
        train_file=args.train_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()
