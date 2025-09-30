#!/usr/bin/env python3
"""
Download Electronics category data from Hugging Face dataset
"""

import os
import sys
import json
from datasets import load_dataset

def download_electronics_data():
    """Download Electronics category data"""
    try:
        print("📥 Downloading Electronics category data...")
        
        # Load the dataset
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "Electronics", split="train")
        
        print(f"✅ Loaded {len(dataset)} Electronics reviews")
        
        # Convert to JSONL format
        output_file = "data_ingest/data_ingest/raw_review_Electronics.jsonl"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            for i, item in enumerate(dataset):
                if i >= 1000:  # Limit to 1000 reviews for now
                    break
                    
                # Convert to our format
                review_data = {
                    'review_text': item.get('review_text', ''),
                    'parent_asin': item.get('parent_asin', ''),
                    'rating': item.get('rating', 0),
                    'category': 'Electronics'
                }
                
                f.write(json.dumps(review_data) + '\n')
        
        print(f"✅ Saved {min(1000, len(dataset))} Electronics reviews to {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading Electronics data: {e}")
        return False

if __name__ == "__main__":
    success = download_electronics_data()
    if success:
        print("🎉 Electronics data download completed!")
    else:
        print("💥 Electronics data download failed!")