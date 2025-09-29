#!/usr/bin/env python3
"""
Create a simulated Electronics category by modifying existing All_Beauty data.
This is for demonstration purposes to show multi-category functionality.
"""

import json
import random

def create_simulated_electronics():
    """Create simulated Electronics data from All_Beauty data."""
    
    # Electronics-related product descriptions and features
    electronics_features = [
        "battery life", "camera quality", "display resolution", "sound quality", 
        "performance", "build quality", "user interface", "connectivity",
        "storage capacity", "processing speed", "screen size", "durability",
        "charging speed", "wireless connectivity", "software updates"
    ]
    
    electronics_products = [
        "smartphone", "laptop", "tablet", "headphones", "speaker", "camera",
        "smartwatch", "fitness tracker", "bluetooth earbuds", "power bank",
        "wireless charger", "gaming console", "smart home device", "router"
    ]
    
    # Load existing All_Beauty data
    with open('data_ingest/data_ingest/raw_review_All_Beauty_expanded.jsonl', 'r', encoding='utf-8') as f:
        beauty_reviews = [json.loads(line.strip()) for line in f]
    
    # Create simulated Electronics reviews
    electronics_reviews = []
    
    for i, review in enumerate(beauty_reviews[:200]):  # Take first 200 reviews
        # Create a new ASIN for electronics
        new_asin = f"ELEC{i:06d}"
        
        # Modify the review text to be electronics-related
        original_text = review['text']
        
        # Replace beauty-related words with electronics-related words
        text_replacements = {
            'beauty': 'performance',
            'makeup': 'device',
            'skin': 'screen',
            'hair': 'battery',
            'face': 'interface',
            'color': 'display',
            'scent': 'sound',
            'texture': 'build quality',
            'quality': 'performance',
            'nice': 'excellent',
            'good': 'great',
            'bad': 'poor',
            'love': 'recommend',
            'hate': 'disappointed'
        }
        
        modified_text = original_text
        for old_word, new_word in text_replacements.items():
            modified_text = modified_text.replace(old_word, new_word)
        
        # Add some electronics-specific content
        if random.random() < 0.3:  # 30% chance to add electronics features
            feature = random.choice(electronics_features)
            product = random.choice(electronics_products)
            electronics_content = f" The {product} has great {feature}. "
            modified_text = electronics_content + modified_text
        
        # Create new review
        electronics_review = {
            'rating': review['rating'],
            'title': f"Electronics Review {i+1}",
            'text': modified_text,
            'asin': new_asin,
            'parent_asin': new_asin,
            'timestamp': review['timestamp'],
            'user_id': review['user_id'],
            'verified_purchase': review['verified_purchase']
        }
        
        electronics_reviews.append(electronics_review)
    
    # Save simulated Electronics data
    with open('data_ingest/data_ingest/raw_review_Electronics_simulated.jsonl', 'w', encoding='utf-8') as f:
        for review in electronics_reviews:
            f.write(json.dumps(review) + '\n')
    
    print(f"Created {len(electronics_reviews)} simulated Electronics reviews")
    print("Sample Electronics review:")
    sample = electronics_reviews[0]
    print(f"  ASIN: {sample['asin']}")
    print(f"  Text: {sample['text'][:200]}...")
    
    return electronics_reviews

if __name__ == "__main__":
    create_simulated_electronics()
