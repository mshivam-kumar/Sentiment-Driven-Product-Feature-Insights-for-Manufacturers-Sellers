
import argparse
import os
import json
from datetime import datetime
import s3fs
import pandas as pd
from datasets import load_dataset, get_dataset_config_names

def download_and_upload(dataset_id, subset, s3_bucket, s3_prefix, num_samples=None):
    """
    Downloads a specified subset from the Hugging Face dataset, converts it to JSONL,
    and uploads it to an S3 bucket.

    Args:
        dataset_id (str): The ID of the Hugging Face dataset (e.g., "McAuley-Lab/Amazon-Reviews-2023").
        subset (str): The specific subset to download (e.g., "raw_review_All_Beauty").
        s3_bucket (str): The name of the S3 bucket.
        s3_prefix (str): The S3 prefix (e.g., "raw/All_Beauty").
        num_samples (int, optional): Number of samples to download for testing. If None, downloads all.
    """
    fs = s3fs.S3FileSystem()
    current_date = datetime.now().strftime("%Y%m%d")
    output_filename = f"{subset}.jsonl"
    s3_path = f"s3://{s3_bucket}/{s3_prefix}/{current_date}/{output_filename}"

    print(f"Loading dataset: {dataset_id}, subset: {subset}")
    try:
        # Check if the subset exists
        config_names = get_dataset_config_names(dataset_id)
        if subset not in config_names:
            print(f"Error: Subset '{subset}' not found in dataset '{dataset_id}'. Available subsets: {config_names}")
            return

        dataset = load_dataset(dataset_id, subset)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Dataset loaded. Splits available: {dataset.keys()}")

    # We assume 'train' split for the purpose of this example.
    # In a real scenario, you might want to iterate through all splits or specify.
    if 'train' in dataset:
        data_to_process = dataset['train']
    else:
        print("No 'train' split found. Using the first available split.")
        data_to_process = dataset[list(dataset.keys())[0]]

    if num_samples:
        print(f"Sampling {num_samples} records for local validation.")
        data_to_process = data_to_process.select(range(min(num_samples, len(data_to_process))))

    # Convert to pandas DataFrame for easier JSONL conversion
    df = pd.DataFrame(data_to_process)

    # Filter for relevant fields as specified in the prompt
    # Ensure all required fields exist before selecting
    expected_fields = ['rating', 'title', 'text', 'asin', 'parent_asin', 'timestamp', 'user_id', 'verified_purchase']
    # Filter out fields that are not present in the dataframe to prevent errors
    actual_fields = [field for field in expected_fields if field in df.columns]

    # Convert timestamp to integer if it exists and is not already, handling NaNs
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].apply(lambda x: int(x) if pd.notna(x) else None)
    
    # Select and reorder columns
    df_filtered = df[actual_fields]

    local_output_path = f"data_ingest/{output_filename}"
    print(f"Saving {len(df_filtered)} records to local file: {local_output_path}")
    df_filtered.to_json(local_output_path, orient="records", lines=True)

    print(f"Uploading to S3: {s3_path}")
    fs.upload(local_output_path, s3_path)
    print("Upload complete.")

    # Validation: Count and sample 5 records
    print("\n--- Local Validation (first 5 records) ---")
    with open(local_output_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(json.loads(line))
    print(f"Total local records processed: {len(df_filtered)}")

    print(f"\n--- S3 Validation (listing S3 path) ---")
    try:
        s3_files = fs.ls(f"s3://{s3_bucket}/{s3_prefix}/{current_date}/")
        print(f"Files in S3: {s3_files}")

        # Try to read a sample from S3 if possible (requires 'read' access and potentially larger files)
        # For simplicity, we'll just check if the file exists.
        if s3_path in [f"s3://{f}" for f in s3_files]:
             print(f"'{output_filename}' successfully found in S3 at '{s3_path}'")
        else:
             print(f"WARNING: '{output_filename}' not explicitly listed in '{s3_path}'. Check path consistency.")

    except Exception as e:
        print(f"Could not list S3 path or validate S3 content: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Amazon reviews and upload to S3.")
    parser.add_argument("--dataset_id", type=str, default="McAuley-Lab/Amazon-Reviews-2023",
                        help="Hugging Face dataset ID.")
    parser.add_argument("--subset", type=str, default="raw_review_All_Beauty",
                        help="Hugging Face dataset subset (e.g., raw_review_All_Beauty).")
    parser.add_argument("--s3_bucket", type=str, required=True,
                        help="Name of the S3 bucket.")
    parser.add_argument("--s3_prefix", type=str, default="raw/All_Beauty",
                        help="S3 prefix for storing raw data (e.g., raw/All_Beauty).")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to download for testing. Downloads all if None.")

    args = parser.parse_args()

    # Create data_ingest directory if it doesn't exist for local output
    os.makedirs('data_ingest', exist_ok=True)

    download_and_upload(args.dataset_id, args.subset, args.s3_bucket, args.s3_prefix, args.num_samples)