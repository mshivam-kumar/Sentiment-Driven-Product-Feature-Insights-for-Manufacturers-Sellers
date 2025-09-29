#!/usr/bin/env python3
"""
Deploy models and create a complete deployment package for cloud platforms.
This script packages the models with the Streamlit app for deployment.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required dependencies for model deployment."""
    print("📦 Installing dependencies...")
    
    # Install core dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", 
                   "streamlit>=1.28.0",
                   "requests>=2.31.0", 
                   "plotly>=5.17.0",
                   "pandas>=2.1.0",
                   "numpy>=1.24.0"], check=True)
    
    # Install model dependencies
    subprocess.run([sys.executable, "-m", "pip", "install",
                   "transformers>=4.30.0",
                   "torch>=2.0.0",
                   "spacy>=3.6.0",
                   "yake>=0.4.8"], check=True)
    
    # Download spaCy model
    try:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("✅ spaCy model downloaded successfully")
    except subprocess.CalledProcessError:
        print("⚠️  Warning: Could not download spaCy model. Will use fallback extraction.")

def create_deployment_structure():
    """Create deployment directory structure."""
    print("📁 Creating deployment structure...")
    
    # Create deployment directory
    deploy_dir = Path("deployment")
    deploy_dir.mkdir(exist_ok=True)
    
    # Copy model files
    model_files = [
        "models/aspect_extractor/infer_aspect.py",
        "models/sentiment/infer_sentiment.py",
        "models/aspect_extractor/requirements.txt",
        "models/sentiment/requirements.txt"
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            dest_path = deploy_dir / Path(file_path).name
            shutil.copy2(file_path, dest_path)
            print(f"✅ Copied {file_path}")
    
    # Copy Streamlit app
    shutil.copy2("streamlit_deploy.py", deploy_dir / "app.py")
    print("✅ Copied Streamlit app")
    
    # Copy requirements
    shutil.copy2("requirements_deploy.txt", deploy_dir / "requirements.txt")
    print("✅ Copied requirements")
    
    return deploy_dir

def create_standalone_app():
    """Create a standalone app that includes models."""
    print("🔧 Creating standalone app...")
    
    standalone_app = """
import streamlit as st
import requests
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

# Add current directory to path for model imports
sys.path.append(str(Path(__file__).parent))

# Import models
try:
    from infer_aspect import AspectExtractor
    from infer_sentiment import SentimentAnalyzer
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Models not available: {e}")
    MODELS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Sentiment-Driven Product Feature Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'https://f3157r5ca4.execute-api.us-east-1.amazonaws.com/dev')

# Custom CSS
st.markdown(\"\"\"
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .feature-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .positive-sentiment { color: #28a745; }
    .negative-sentiment { color: #dc3545; }
    .neutral-sentiment { color: #6c757d; }
</style>
\"\"\", unsafe_allow_html=True)

def analyze_text_locally(text):
    \"\"\"Analyze text using local models.\"\"\"
    if not MODELS_AVAILABLE or not text:
        return None
    
    try:
        # Initialize models
        aspect_extractor = AspectExtractor()
        sentiment_analyzer = SentimentAnalyzer()
        
        # Extract aspects
        aspects = aspect_extractor.extract_aspects(text)
        
        # Analyze sentiment for each aspect
        results = {}
        for aspect, score in aspects.items():
            sentiment = sentiment_analyzer.analyze_sentiment(text)
            results[aspect] = {
                'score': sentiment['score'],
                'confidence': sentiment.get('confidence', 0.8)
            }
        
        return results
    except Exception as e:
        st.error(f"Local analysis error: {e}")
        return None

def fetch_product_sentiment(asin, feature_filter=None, window=None):
    \"\"\"Fetch product sentiment data from API.\"\"\"
    try:
        url = f"{API_BASE_URL}/sentiment/product/{asin}"
        params = {}
        if feature_filter:
            params['feature'] = feature_filter
        if window:
            params['window'] = window
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None

def fetch_feature_search(query, category=None, limit=20):
    \"\"\"Fetch feature search results from API.\"\"\"
    try:
        url = f"{API_BASE_URL}/sentiment/search"
        params = {'query': query, 'limit': limit}
        if category:
            params['category'] = category
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error searching features: {e}")
        return None

def get_sentiment_color(score):
    \"\"\"Get color based on sentiment score.\"\"\"
    if score > 0.1:
        return "positive-sentiment"
    elif score < -0.1:
        return "negative-sentiment"
    else:
        return "neutral-sentiment"

def main():
    \"\"\"Main Streamlit application.\"\"\"
    
    # Header
    st.markdown('<h1 class="main-header">📊 Sentiment-Driven Product Feature Insights</h1>', unsafe_allow_html=True)
    
    # Model status
    if MODELS_AVAILABLE:
        st.success("✅ Local models loaded successfully")
    else:
        st.warning("⚠️  Local models not available. Using API only.")
    
    # Sidebar
    st.sidebar.title("🔍 Analysis Options")
    
    # Analysis type selection
    analysis_type = st.sidebar.radio(
        "Choose Analysis Type",
        ["Product Analysis", "Feature Search", "Local Text Analysis"],
        help="Select analysis type"
    )
    
    if analysis_type == "Local Text Analysis":
        # Local Text Analysis Section
        st.header("📝 Local Text Analysis")
        
        text_input = st.text_area(
            "Enter text to analyze",
            placeholder="Enter product review or any text to analyze sentiment and extract features...",
            height=150
        )
        
        if st.button("🔍 Analyze Text", type="primary"):
            if text_input:
                with st.spinner("Analyzing text locally..."):
                    results = analyze_text_locally(text_input)
                    
                    if results:
                        st.subheader("🎯 Analysis Results")
                        
                        # Display aspects and sentiment
                        for aspect, info in results.items():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{aspect.replace('_', ' ').title()}**")
                            
                            with col2:
                                score = info['score']
                                color_class = get_sentiment_color(score)
                                st.markdown(f'<span class="{color_class}">{score:.2f}</span>', unsafe_allow_html=True)
                            
                            with col3:
                                confidence = info.get('confidence', 0.8)
                                st.progress(confidence)
                                st.caption(f"{confidence:.1%} confidence")
                            
                            st.divider()
                    else:
                        st.error("Failed to analyze text. Please try again.")
            else:
                st.warning("Please enter some text to analyze.")
    
    elif analysis_type == "Product Analysis":
        # Product Analysis Section (same as before)
        st.header("📱 Product Sentiment Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            asin = st.text_input(
                "Product ASIN",
                value="B08JTNQFZY",
                help="Enter Amazon product ASIN"
            )
        
        with col2:
            feature_filter = st.selectbox(
                "Filter by Feature",
                ["All Features", "build_quality", "customer_service", "design", "value_for_money", "style", "size_fit"],
                help="Filter to show only specific feature"
            )
        
        time_window = st.sidebar.selectbox(
            "Time Window",
            ["All Time", "7d", "30d", "90d", "1y", "10y"],
            index=0,
            help="Time window for analysis"
        )
        
        window_param = None if time_window == "All Time" else time_window
        feature_param = None if feature_filter == "All Features" else feature_filter
        
        if st.button("🔍 Analyze Product", type="primary"):
            with st.spinner("Fetching product sentiment data..."):
                data = fetch_product_sentiment(asin, feature_param, window_param)
                
                if data and 'error' not in data:
                    # Display results (same as before)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Features", len(data.get('features', {})))
                    
                    with col2:
                        st.metric("Total Reviews", data.get('total_reviews', 0))
                    
                    with col3:
                        overall_sentiment = data.get('overall_sentiment', 0)
                        st.metric("Overall Sentiment", f"{overall_sentiment:.2f}")
                    
                    with col4:
                        last_updated = data.get('last_updated', 'Unknown')
                        st.metric("Last Updated", last_updated.split('T')[0] if 'T' in last_updated else last_updated)
                    
                    # Features breakdown
                    st.subheader("🎯 Feature Breakdown")
                    
                    features = data.get('features', {})
                    if features:
                        feature_data = []
                        for feature, info in features.items():
                            feature_data.append({
                                'Feature': feature.replace('_', ' ').title(),
                                'Sentiment Score': info['score'],
                                'Review Count': info['count'],
                                'Trend': info.get('trend', 'stable')
                            })
                        
                        df = pd.DataFrame(feature_data)
                        df = df.sort_values('Sentiment Score', ascending=False)
                        
                        for _, row in df.iterrows():
                            with st.container():
                                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                                
                                with col1:
                                    st.markdown(f"**{row['Feature']}**")
                                
                                with col2:
                                    score = row['Sentiment Score']
                                    color_class = get_sentiment_color(score)
                                    st.markdown(f'<span class="{color_class}">{score:.2f}</span>', unsafe_allow_html=True)
                                
                                with col3:
                                    st.text(f"{row['Review Count']} reviews")
                                
                                with col4:
                                    st.text(row['Trend'])
                                
                                st.divider()
                        
                        # Sentiment distribution chart
                        st.subheader("📈 Sentiment Distribution")
                        
                        fig = px.bar(
                            df, 
                            x='Feature', 
                            y='Sentiment Score',
                            color='Sentiment Score',
                            color_continuous_scale=['red', 'yellow', 'green'],
                            title="Feature Sentiment Scores"
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.warning("No features found for this product.")
                        
                elif data and 'error' in data:
                    st.error(f"Error: {data['error']}")
                else:
                    st.error("Failed to fetch product data. Please check the ASIN and try again.")
    
    else:
        # Feature Search Section (same as before)
        st.header("🔍 Feature Search Across Products")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input(
                "Search Feature",
                value="quality",
                help="Search for a specific feature across all products"
            )
        
        with col2:
            search_limit = st.number_input(
                "Results Limit",
                min_value=5,
                max_value=50,
                value=20,
                help="Maximum number of results to show"
            )
        
        if st.button("🔍 Search Features", type="primary"):
            with st.spinner("Searching for features..."):
                data = fetch_feature_search(search_query, limit=search_limit)
                
                if data and 'results' in data:
                    results = data['results']
                    
                    if results:
                        st.subheader(f"🎯 Found {len(results)} products with '{search_query}' feature")
                        
                        results_data = []
                        for result in results:
                            results_data.append({
                                'ASIN': result['asin'],
                                'Feature': result['feature'],
                                'Sentiment Score': result['score'],
                                'Review Count': result['count'],
                                'Category': result.get('category', 'Unknown')
                            })
                        
                        df_results = pd.DataFrame(results_data)
                        df_results = df_results.sort_values('Sentiment Score', ascending=False)
                        
                        st.dataframe(df_results, use_container_width=True, hide_index=True)
                        
                        # Top products chart
                        st.subheader("🏆 Top Products by Sentiment")
                        
                        top_products = df_results.head(10)
                        fig = px.bar(
                            top_products,
                            x='ASIN',
                            y='Sentiment Score',
                            color='Sentiment Score',
                            color_continuous_scale=['red', 'yellow', 'green'],
                            title=f"Top 10 Products for '{search_query}' Feature"
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.warning(f"No results found for '{search_query}'")
                        
                elif data and 'error' in data:
                    st.error(f"Error: {data['error']}")
                else:
                    st.error("Failed to search features. Please try again.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        \"\"\"
        <div style='text-align: center; color: #666;'>
            <p>📊 Sentiment-Driven Product Feature Insights | Powered by AWS & Streamlit</p>
            <p>Built with ❤️ for manufacturers and sellers</p>
            <p style='font-weight: bold; color: #1f77b4;'>Designed by Shivam Kumar<br>IIT Gandhinagar</p>
        </div>
        \"\"\",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
"""
    
    return standalone_app

def create_deployment_files(deploy_dir):
    """Create deployment configuration files."""
    print("📝 Creating deployment files...")
    
    # Create standalone app
    standalone_app = create_standalone_app()
    with open(deploy_dir / "app.py", "w") as f:
        f.write(standalone_app)
    
    # Create Procfile for Heroku
    procfile = "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
    with open(deploy_dir / "Procfile", "w") as f:
        f.write(procfile)
    
    # Create runtime.txt for Python version
    runtime = "python-3.11.0"
    with open(deploy_dir / "runtime.txt", "w") as f:
        f.write(runtime)
    
    # Create setup.sh for dependencies
    setup_sh = """#!/bin/bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
"""
    with open(deploy_dir / "setup.sh", "w") as f:
        f.write(setup_sh)
    os.chmod(deploy_dir / "setup.sh", 0o755)
    
    print("✅ Deployment files created")

def main():
    """Main deployment function."""
    print("🚀 Starting model deployment...")
    
    try:
        # Install dependencies
        install_dependencies()
        
        # Create deployment structure
        deploy_dir = create_deployment_structure()
        
        # Create deployment files
        create_deployment_files(deploy_dir)
        
        print(f"✅ Deployment package created in: {deploy_dir}")
        print("\n📋 Next steps:")
        print("1. Navigate to the deployment directory")
        print("2. Deploy to your preferred platform (Heroku, Streamlit Cloud, etc.)")
        print("3. Set environment variables if needed")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
