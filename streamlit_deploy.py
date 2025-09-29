#!/usr/bin/env python3
"""
Streamlit deployment configuration for cloud platforms.
"""

import streamlit as st
import requests
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Sentiment-Driven Product Feature Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'https://f3157r5ca4.execute-api.us-east-1.amazonaws.com/dev')

# Custom CSS for better styling
st.markdown("""
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
""", unsafe_allow_html=True)

def fetch_product_sentiment(asin, feature_filter=None, window=None):
    """Fetch product sentiment data from API."""
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
    """Fetch feature search results from API."""
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
    """Get color based on sentiment score."""
    if score > 0.1:
        return "positive-sentiment"
    elif score < -0.1:
        return "negative-sentiment"
    else:
        return "neutral-sentiment"

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Sentiment-Driven Product Feature Insights</h1>', unsafe_allow_html=True)
    
    # Information section
    with st.expander("ℹ️ How to Use This App", expanded=False):
        st.markdown("""
        **Welcome to the Sentiment-Driven Product Feature Insights App!**
        
        This app helps manufacturers and sellers understand customer sentiment about specific product features.
        
        ### 🎯 **Product Analysis**
        - Enter an Amazon ASIN (product ID) to see sentiment analysis for different features
        - Use the example ASINs in the sidebar for quick testing
        - Filter by specific features to focus on particular aspects
        
        ### 🔍 **Feature Search**
        - Search for a specific feature across all products in the database
        - See which products perform best for that feature
        - Compare sentiment scores across different products
        
        ### 📊 **Understanding the Results**
        - **Sentiment Score**: Ranges from -1 (very negative) to +1 (very positive)
        - **Review Count**: Number of reviews analyzed for that feature
        - **Trend**: Indicates if sentiment is improving, declining, or stable
        
        ### 💡 **Tips for Best Results**
        - Try the example ASINs first to see the app in action
        - Use common feature names like "quality", "design", "performance"
        - The app works best with products that have multiple reviews
        """)
    
    # Quick start section
    st.markdown("### 🚀 Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📱 Analyze B08JTNQFZY", help="Analyze a hair styling product with multiple features"):
            st.session_state.quick_asin = "B08JTNQFZY"
            st.session_state.quick_analysis_type = "Product Analysis"
    
    with col2:
        if st.button("🔍 Search 'Quality'", help="Search for quality features across all products"):
            st.session_state.quick_feature = "quality"
            st.session_state.quick_analysis_type = "Feature Search"
    
    with col3:
        if st.button("🎯 Search 'Design'", help="Search for design features across all products"):
            st.session_state.quick_feature = "design"
            st.session_state.quick_analysis_type = "Feature Search"
    
    # Sidebar
    st.sidebar.title("🔍 Analysis Options")
    
    # Example ASINs section
    st.sidebar.markdown("### 📱 Example ASINs to Try")
    example_asins = {
        "B08JTNQFZY": "Hair Styling Product (Multiple Features)",
        "B097YYB2GV": "Beauty Tool (Build Quality Focus)",
        "B015ZXMSFQ": "Skincare Product (Material & Value)",
        "B088838886": "Hair Accessory (Design & Quality)",
        "B07FX94GYX": "Skincare Tool (Performance Focus)"
    }
    
    selected_asin = st.sidebar.selectbox(
        "Quick Select ASIN:",
        ["Custom ASIN"] + list(example_asins.keys()),
        format_func=lambda x: f"{x} - {example_asins.get(x, 'Enter your own')}"
    )
    
    # Example features section
    st.sidebar.markdown("### 🎯 Example Features to Search")
    example_features = [
        "quality", "design", "performance", "value_for_money", 
        "build_quality", "customer_service", "style", "material",
        "battery", "camera", "comfort", "durability"
    ]
    
    selected_feature = st.sidebar.selectbox(
        "Quick Select Feature:",
        ["Custom Feature"] + example_features
    )
    
    # Analysis type selection
    analysis_type = st.sidebar.radio(
        "Choose Analysis Type",
        ["Product Analysis", "Feature Search"],
        help="Select whether to analyze a specific product or search for features across products"
    )
    
    # Handle quick start buttons
    if hasattr(st.session_state, 'quick_analysis_type'):
        if st.session_state.quick_analysis_type == "Product Analysis" and hasattr(st.session_state, 'quick_asin'):
            analysis_type = "Product Analysis"
            selected_asin = st.session_state.quick_asin
            # Clear the session state
            del st.session_state.quick_analysis_type
            del st.session_state.quick_asin
        elif st.session_state.quick_analysis_type == "Feature Search" and hasattr(st.session_state, 'quick_feature'):
            analysis_type = "Feature Search"
            selected_feature = st.session_state.quick_feature
            # Clear the session state
            del st.session_state.quick_analysis_type
            del st.session_state.quick_feature
    
    if analysis_type == "Product Analysis":
        # Product Analysis Section
        st.header("📱 Product Sentiment Analysis")
        
        # Input fields
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if selected_asin == "Custom ASIN":
                asin = st.text_input(
                    "Product ASIN",
                    value="B08JTNQFZY",  # Default to a product with multiple features
                    help="Enter Amazon product ASIN (e.g., B08JTNQFZY)"
                )
            else:
                asin = st.text_input(
                    "Product ASIN",
                    value=selected_asin,
                    help=f"Selected: {example_asins[selected_asin]}"
                )
        
        with col2:
            feature_filter = st.selectbox(
                "Filter by Feature",
                ["All Features", "build_quality", "customer_service", "design", "value_for_money", "style", "size_fit"],
                help="Filter to show only specific feature"
            )
        
        # Time window selection
        time_window = st.sidebar.selectbox(
            "Time Window",
            ["All Time", "7d", "30d", "90d", "1y", "10y"],
            index=0,
            help="Time window for analysis"
        )
        
        # Convert "All Time" to None
        window_param = None if time_window == "All Time" else time_window
        feature_param = None if feature_filter == "All Features" else feature_filter
        
        # Fetch and display data
        if st.button("🔍 Analyze Product", type="primary"):
            with st.spinner("Fetching product sentiment data..."):
                data = fetch_product_sentiment(asin, feature_param, window_param)
                
                if data and 'error' not in data:
                    # Display product information
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
                        # Create feature data for visualization
                        feature_data = []
                        for feature, info in features.items():
                            feature_data.append({
                                'Feature': feature.replace('_', ' ').title(),
                                'Sentiment Score': info['score'],
                                'Review Count': info['count'],
                                'Trend': info.get('trend', 'stable')
                            })
                        
                        df = pd.DataFrame(feature_data)
                        
                        # Sort by sentiment score
                        df = df.sort_values('Sentiment Score', ascending=False)
                        
                        # Display features in cards
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
        # Feature Search Section
        st.header("🔍 Feature Search Across Products")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if selected_feature == "Custom Feature":
                search_query = st.text_input(
                    "Search Feature",
                    value="quality",
                    help="Search for a specific feature across all products"
                )
            else:
                search_query = st.text_input(
                    "Search Feature",
                    value=selected_feature,
                    help=f"Selected: {selected_feature}"
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
                        
                        # Create results dataframe
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
                        
                        # Sort by sentiment score
                        df_results = df_results.sort_values('Sentiment Score', ascending=False)
                        
                        # Display results
                        st.dataframe(
                            df_results,
                            use_container_width=True,
                            hide_index=True
                        )
                        
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
        """
        <div style='text-align: center; color: #666;'>
            <p>📊 Sentiment-Driven Product Feature Insights | Powered by AWS & Streamlit</p>
            <p>Built with ❤️ for manufacturers and sellers</p>
            <p style='font-weight: bold; color: #1f77b4;'>Designed by Shivam Kumar<br>IIT Gandhinagar</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
