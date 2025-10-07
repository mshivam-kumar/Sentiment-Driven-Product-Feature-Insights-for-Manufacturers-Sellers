"""
Database connection and utilities
"""

import boto3
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# AWS configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE', 'product_sentiment_insights')

# Initialize AWS clients
dynamodb = None
s3_client = None

def get_dynamodb():
    """Get DynamoDB client"""
    global dynamodb
    if dynamodb is None:
        dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    return dynamodb

def get_s3_client():
    """Get S3 client"""
    global s3_client
    if s3_client is None:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
    return s3_client

def get_database():
    """Get database connection"""
    return {
        "dynamodb": get_dynamodb(),
        "s3": get_s3_client(),
        "table_name": DYNAMODB_TABLE
    }



