"""
Training script for improved aspect extraction model.

This script implements a BERT-based aspect extraction model that can be trained
on labeled data for better performance than the baseline spaCy + YAKE approach.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np


class AspectDataset(Dataset):
    """Dataset for aspect extraction training."""
    
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            texts: List of input texts
            labels: List of BIO labels for each text
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
        labels = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Pad labels to match token length
        padded_labels = labels[:self.max_length]
        padded_labels.extend([-100] * (self.max_length - len(padded_labels)))
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(padded_labels, dtype=torch.long)
        }


def load_training_data(file_path: str) -> Tuple[List[str], List[List[int]]]:
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
            texts.append(data['text'])
            labels.append(data['labels'])
    
    return texts, labels


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored indices
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        for pred, lab in zip(prediction, label):
            if lab != -100:
                true_predictions.append(pred)
                true_labels.append(lab)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average='weighted'
    )
    accuracy = accuracy_score(true_labels, true_predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_aspect_model(
    train_file: str,
    model_name: str = "bert-base-uncased",
    output_dir: str = "./aspect_model",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """
    Train the aspect extraction model.
    
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
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=3,  # B, I, O
        id2label={0: "O", 1: "B-ASPECT", 2: "I-ASPECT"},
        label2id={"O": 0, "B-ASPECT": 1, "I-ASPECT": 2}
    )
    
    # Load training data
    texts, labels = load_training_data(train_file)
    
    # Create dataset
    dataset = AspectDataset(texts, labels, tokenizer)
    
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
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
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


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train aspect extraction model")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data file")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./aspect_model",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    
    args = parser.parse_args()
    
    train_aspect_model(
        train_file=args.train_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()
