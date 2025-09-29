import pytest
import os
import json
from unittest.mock import patch, MagicMock
from datetime import datetime
from data_ingest.download_and_upload import download_and_upload

# Mock S3FS to prevent actual S3 calls during testing
@pytest.fixture(autouse=True)
def mock_s3fs():
    with patch('data_ingest.download_and_upload.s3fs.S3FileSystem') as mock_fs:
        mock_fs_instance = mock_fs.return_value
        mock_fs_instance.upload = MagicMock()
        mock_fs_instance.ls = MagicMock(return_value=["s3://test-bucket/raw/All_Beauty/20231027/raw_review_All_Beauty.jsonl"])
        yield

# Mock datasets.load_dataset and get_dataset_config_names
@pytest.fixture
def mock_datasets():
    with patch('data_ingest.download_and_upload.load_dataset') as mock_load_dataset, \
         patch('data_ingest.download_and_upload.get_dataset_config_names') as mock_get_dataset_config_names:

        # Mock get_dataset_config_names to return available subsets
        mock_get_dataset_config_names.return_value = ["raw_review_All_Beauty"]

        # Mock the dataset object returned by load_dataset
        mock_dataset = {
            'train': [
                {'rating': 5.0, 'title': 'Great Product', 'text': 'Battery life is excellent.', 'asin': 'B00A123DEF', 'parent_asin': 'B00A123', 'timestamp': 1678886400, 'user_id': 'U1', 'verified_purchase': True},
                {'rating': 3.0, 'title': 'Average', 'text': 'Camera is blurry, but display is good.', 'asin': 'B00A123DEG', 'parent_asin': 'B00A123', 'timestamp': 1678972800, 'user_id': 'U2', 'verified_purchase': False},
                {'rating': 4.0, 'title': 'Good Value', 'text': 'Works as expected.', 'asin': 'B00B456HIJ', 'parent_asin': 'B00B456', 'timestamp': 1679059200, 'user_id': 'U3', 'verified_purchase': True},
                {'rating': 2.0, 'title': 'Disappointed', 'text': 'Poor battery.', 'asin': 'B00C789KLM', 'parent_asin': 'B00C789', 'timestamp': 1679145600, 'user_id': 'U4', 'verified_purchase': True},
                {'rating': 5.0, 'title': 'Love it!', 'text': 'The screen quality is amazing.', 'asin': 'B00D012NOP', 'parent_asin': 'B00D012', 'timestamp': 1679232000, 'user_id': 'U5', 'verified_purchase': False},
                {'rating': 1.0, 'title': 'Bad', 'text': 'Completely broken in a week.', 'asin': 'B00E345QRS', 'parent_asin': 'B00E345', 'timestamp': 1679318400, 'user_id': 'U6', 'verified_purchase': True},
            ]
        }
        mock_load_dataset.return_value = mock_dataset
        yield mock_load_dataset, mock_get_dataset_config_names

@pytest.fixture
def cleanup_local_file():
    # Ensure local directory for output exists
    os.makedirs('data_ingest', exist_ok=True)
    file_path = 'data_ingest/raw_review_All_Beauty.jsonl'
    if os.path.exists(file_path):
        os.remove(file_path)
    yield
    if os.path.exists(file_path):
        os.remove(file_path)

def test_download_and_upload_success(mock_datasets, mock_s3fs, cleanup_local_file):
    """
    Tests that the download_and_upload function successfully processes data,
    creates a local JSONL file, and calls S3 upload.
    """
    s3_bucket = "test-bucket"
    s3_prefix = "raw/All_Beauty"
    subset = "raw_review_All_Beauty"
    num_samples = 3

    download_and_upload(
        dataset_id="McAuley-Lab/Amazon-Reviews-2023",
        subset=subset,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        num_samples=num_samples
    )

    # Check if local file was created
    local_file_path = f"data_ingest/{subset}.jsonl"
    assert os.path.exists(local_file_path)

    # Check content of local file
    with open(local_file_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) == num_samples
        for line in lines:
            record = json.loads(line)
            assert all(field in record for field in ['rating', 'title', 'text', 'asin', 'parent_asin', 'timestamp'])
            assert isinstance(record['timestamp'], int) or record['timestamp'] is None

    # Check if S3 upload was called
    mock_s3fs.return_value.upload.assert_called_once()
    current_date = datetime.now().strftime("%Y%m%d")
    expected_s3_path = f"s3://{s3_bucket}/{s3_prefix}/{current_date}/{subset}.jsonl"
    assert mock_s3fs.return_value.upload.call_args[0][1] == expected_s3_path

    # Check S3 listing was called
    mock_s3fs.return_value.ls.assert_called_once()

def test_download_and_upload_invalid_subset(mock_datasets, mock_s3fs, capsys):
    """
    Tests handling of an invalid subset name.
    """
    # Configure mock_get_dataset_config_names to not include the requested subset
    mock_datasets[1].return_value = ["another_subset"]

    download_and_upload(
        dataset_id="McAuley-Lab/Amazon-Reviews-2023",
        subset="invalid_subset",
        s3_bucket="test-bucket",
        s3_prefix="raw/All_Beauty"
    )

    captured = capsys.readouterr()
    assert "Error: Subset 'invalid_subset' not found" in captured.out
    mock_s3fs.return_value.upload.assert_not_called()

def test_download_and_upload_empty_dataset(mock_datasets, mock_s3fs, cleanup_local_file):
    """
    Tests behavior when the dataset split is empty.
    """
    mock_datasets[0].return_value = {'train': []} # Mock an empty dataset
    subset = "raw_review_All_Beauty"

    download_and_upload(
        dataset_id="McAuley-Lab/Amazon-Reviews-2023",
        subset=subset,
        s3_bucket="test-bucket",
        s3_prefix="raw/All_Beauty",
        num_samples=1
    )

    local_file_path = f"data_ingest/{subset}.jsonl"
    assert os.path.exists(local_file_path)
    with open(local_file_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 0 # Expect an empty file

    mock_s3fs.return_value.upload.assert_called_once()
    # The upload should still be called even if the file is empty, as it creates an empty file on S3.