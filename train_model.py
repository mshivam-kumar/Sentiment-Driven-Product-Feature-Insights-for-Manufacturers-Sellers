"""
Fine-tuning Script for TinyLlama on Product Review Data

This script fine-tunes TinyLlama on product review data to create a domain-specific model
for better sentiment analysis and product insights generation.
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os
from typing import List, Dict, Any

class ProductReviewFineTuner:
    """Fine-tune TinyLlama on product review data using LoRA."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.training_data = []
        
    def load_review_data(self, reviews_data: List[Dict[str, Any]]):
        """Load and format review data for training."""
        print(f"📊 Loading {len(reviews_data)} reviews for fine-tuning...")
        
        training_examples = []
        
        for review in reviews_data:
            text = review.get('text', '') or review.get('review_text', '')
            sentiment = review.get('sentiment_score', 0.0)
            rating = review.get('rating', 0)
            asin = review.get('parent_asin', 'Unknown')
            
            if not text or len(text.strip()) < 20:
                continue
                
            # Create training examples in chat format
            # Example 1: Sentiment analysis
            sentiment_label = "positive" if sentiment > 0.2 else "negative" if sentiment < -0.2 else "neutral"
            training_examples.append({
                "text": f"<|user|>\nAnalyze the sentiment of this review: {text[:200]}...\n<|assistant|>\nThis review has a {sentiment_label} sentiment (score: {sentiment:.2f}). The customer rated it {rating}/5 stars."
            })
            
            # Example 2: Feature extraction
            if 'quality' in text.lower():
                training_examples.append({
                    "text": f"<|user|>\nWhat do customers say about quality in this review: {text[:200]}...\n<|assistant|>\nBased on this review, the customer mentions quality aspects. The sentiment is {sentiment_label} with a score of {sentiment:.2f}."
                })
            
            # Example 3: Product recommendation
            if rating >= 4:
                training_examples.append({
                    "text": f"<|user|>\nShould I recommend this product based on: {text[:200]}...\n<|assistant|>\nYes, this product should be recommended. The customer gave it {rating}/5 stars and the sentiment is {sentiment_label} ({sentiment:.2f})."
                })
        
        self.training_data = training_examples
        print(f"✅ Created {len(self.training_data)} training examples")
        return self.training_data
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer for fine-tuning."""
        print(f"🔧 Setting up {self.model_name} for fine-tuning...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Setup LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print("✅ Model setup complete with LoRA configuration")
        
    def prepare_dataset(self):
        """Prepare dataset for training."""
        if not self.training_data:
            raise ValueError("No training data loaded. Call load_review_data() first.")
        
        print("📝 Preparing dataset for training...")
        
        # Tokenize the data
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        # Create dataset
        dataset = Dataset.from_list(self.training_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're doing causal LM, not masked LM
        )
        
        return tokenized_dataset, data_collator
    
    def train(self, output_dir: str = "./fine_tuned_model", num_epochs: int = 50):
        """Train the model."""
        if not self.model or not self.tokenizer:
            self.setup_model_and_tokenizer()
        
        print(f"🚀 Starting fine-tuning for {num_epochs} epochs...")
        
        # Prepare dataset
        dataset, data_collator = self.prepare_dataset()
        
        # Training arguments with early stopping
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,  # Small batch size for memory efficiency
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=True,  # Use mixed precision
            logging_steps=10,
            save_steps=100,  # Save more frequently
            evaluation_strategy="no",
            save_total_limit=5,  # Keep more checkpoints
            remove_unused_columns=False,
            load_best_model_at_end=True,  # Load best model
            metric_for_best_model="train_loss",
            greater_is_better=False,
            early_stopping_patience=10,  # Stop if no improvement for 10 epochs
        )
        
        # Early stopping: stops when loss stops improving for 10 epochs
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Fine-tuning complete! Model saved to {output_dir}")
        return output_dir
    
    def load_fine_tuned_model(self, model_path: str):
        """Load a fine-tuned model."""
        print(f"📥 Loading fine-tuned model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("✅ Fine-tuned model loaded successfully")
        return self.model, self.tokenizer


def main():
    """Main training function."""
    print("🎯 Starting TinyLlama Fine-tuning on Product Review Data")
    print("=" * 60)
    
    # Initialize fine-tuner
    fine_tuner = ProductReviewFineTuner()
    
    # Load sample review data (you can replace this with your actual data)
    sample_reviews = [
        {
            'text': 'Great product, excellent quality and fast delivery!',
            'sentiment_score': 0.8,
            'parent_asin': 'B123456',
            'rating': 5
        },
        {
            'text': 'Poor quality, broke after one week of use.',
            'sentiment_score': -0.7,
            'parent_asin': 'B123456',
            'rating': 2
        },
        {
            'text': 'Good value for money, works as expected.',
            'sentiment_score': 0.3,
            'parent_asin': 'B123456',
            'rating': 4
        }
    ]
    
    # Load training data
    fine_tuner.load_review_data(sample_reviews)
    
    # Setup model
    fine_tuner.setup_model_and_tokenizer()
    
    # Train
    model_path = fine_tuner.train(num_epochs=3)
    
    print(f"\n🎉 Fine-tuning complete!")
    print(f"📁 Model saved to: {model_path}")
    print(f"🚀 You can now use this fine-tuned model in your RAG system!")


if __name__ == "__main__":
    main()
