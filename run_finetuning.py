"""
Complete Fine-tuning Pipeline for SellerIQ

This script runs the complete fine-tuning pipeline:
1. Prepare training data
2. Fine-tune TinyLlama on product review data
3. Test the fine-tuned model
4. Update the RAG system to use the fine-tuned model
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from prepare_training_data import TrainingDataPreparer
from train_model import ProductReviewFineTuner
from rag_module import RAGSystem

def main():
    """Run the complete fine-tuning pipeline."""
    print("🎯 SellerIQ Fine-tuning Pipeline")
    print("=" * 50)
    
    # Step 1: Prepare training data
    print("\n📊 Step 1: Preparing Training Data")
    print("-" * 30)
    
    preparer = TrainingDataPreparer()
    
    # Load sample data (replace with your actual data)
    reviews_data = preparer.load_sample_data()
    print(f"Loaded {len(reviews_data)} reviews")
    
    # Create training examples
    training_examples = preparer.create_training_examples(reviews_data)
    print(f"Created {len(training_examples)} training examples")
    
    # Save training data
    training_file = preparer.save_training_data(training_examples, "training_data.json")
    
    # Step 2: Fine-tune the model
    print("\n🚀 Step 2: Fine-tuning TinyLlama")
    print("-" * 30)
    
    fine_tuner = ProductReviewFineTuner()
    fine_tuner.load_review_data(reviews_data)
    fine_tuner.setup_model_and_tokenizer()
    
    # Train the model
    model_path = fine_tuner.train(
        output_dir="./fine_tuned_tinyllama",
        num_epochs=3
    )
    
    print(f"✅ Fine-tuning complete! Model saved to: {model_path}")
    
    # Step 3: Test the fine-tuned model
    print("\n🧪 Step 3: Testing Fine-tuned Model")
    print("-" * 30)
    
    # Initialize RAG system with fine-tuned model
    rag_system = RAGSystem(fine_tuned_model_path=model_path)
    
    # Load some test data
    rag_system.load_reviews(reviews_data)
    
    # Test queries
    test_queries = [
        "What do customers say about product quality?",
        "How do customers feel about the design?",
        "Should I buy this product based on reviews?"
    ]
    
    print("Testing fine-tuned model with sample queries:")
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        response = rag_system.query(query, use_transformer=True)
        print(f"📝 Response: {response['answer'][:200]}...")
        print(f"🎯 Model Type: {response['model_type']}")
        print(f"✅ Fine-tuned: {response['is_fine_tuned']}")
    
    # Step 4: Save model info
    print("\n💾 Step 4: Saving Model Information")
    print("-" * 30)
    
    model_info = {
        "model_path": model_path,
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "fine_tuned": True,
        "training_examples": len(training_examples),
        "parameters": "1.1B",
        "training_method": "LoRA fine-tuning",
        "domain": "Product Review Analysis"
    }
    
    with open("model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Model information saved to model_info.json")
    
    # Step 5: Instructions for deployment
    print("\n🚀 Step 5: Deployment Instructions")
    print("-" * 30)
    
    print("""
    To use the fine-tuned model in your Streamlit app:
    
    1. Update your RAG system initialization:
       rag_system = RAGSystem(fine_tuned_model_path="./fine_tuned_tinyllama")
    
    2. The model will automatically load the fine-tuned version
    
    3. Your app will now show "🎯 Fine-tuned TinyLlama" in chat history
    
    4. Resume impact: You now have a domain-specific fine-tuned model!
    """)
    
    print("\n🎉 Fine-tuning pipeline complete!")
    print(f"📁 Fine-tuned model: {model_path}")
    print(f"📊 Training examples: {len(training_examples)}")
    print(f"🎯 Model type: Fine-tuned TinyLlama (1.1B parameters)")
    print(f"💡 Resume impact: Domain-specific AI model for product analysis!")


if __name__ == "__main__":
    main()
