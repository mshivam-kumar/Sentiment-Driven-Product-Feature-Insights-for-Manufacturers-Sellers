#!/usr/bin/env python3
"""
Download Electronics category from Amazon Reviews dataset.
"""

import json
import os
from datasets import load_dataset

def download_electronics_data():
    """Download Electronics category data."""
    try:
        print("Downloading Electronics category from Amazon Reviews dataset...")
        
        # Load the Electronics dataset
        dataset = load_dataset(
            'McAuley-Lab/Amazon-Reviews-2023', 
            'raw_review_Electronics', 
            split='full',
            streaming=False
        )
        
        print(f"Electronics dataset loaded successfully!")
        print(f"Total reviews: {len(dataset)}")
        
        # Create output directory if it doesn't exist
        output_dir = "data_ingest/data_ingest"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save first 100 samples to JSONL file
        samples_to_save = 100
        output_file = os.path.join(output_dir, "raw_review_Electronics.jsonl")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(min(samples_to_save, len(dataset))):
                sample = dataset[i]
                f.write(json.dumps(sample) + '\n')
        
        print(f"Saved {samples_to_save} Electronics reviews to {output_file}")
        
        # Show sample data
        print("\nSample Electronics reviews:")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i+1}:")
            print(f"  ASIN: {sample.get('asin', 'N/A')}")
            print(f"  Rating: {sample.get('rating', 'N/A')}")
            print(f"  Title: {sample.get('title', 'N/A')}")
            print(f"  Text: {sample.get('text', 'N/A')[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"Error downloading Electronics data: {e}")
        print("This might be due to:")
        print("1. Network connectivity issues")
        print("2. Dataset availability on Hugging Face")
        print("3. Authentication requirements")
        return False

if __name__ == "__main__":
    success = download_electronics_data()
    if success:
        print("\n✅ Electronics data downloaded successfully!")
    else:
        print("\n❌ Failed to download Electronics data")
        print("You may need to:")
        print("1. Check your internet connection")
        print("2. Try again later")
        print("3. Use a different category that's already cached")
