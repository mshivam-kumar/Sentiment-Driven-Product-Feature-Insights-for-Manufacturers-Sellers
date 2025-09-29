"""
Streamlit Dashboard for Sentiment-Driven Product Feature Insights

This dashboard provides visualization and analysis of product sentiment insights
extracted from Amazon reviews.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'https://api.example.com/v1')
DEFAULT_ASIN = 'B00YQ6X8EO'

# Page configuration
st.set_page_config(
    page_title="Product Sentiment Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
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
    .positive-sentiment {
        color: #28a745;
    }
    .negative-sentiment {
        color: #dc3545;
    }
    .neutral-sentiment {
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

class SentimentDashboard:
    """Main dashboard class."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.api_base_url = API_BASE_URL
    
    def fetch_product_sentiment(self, asin: str, feature: str = None, window: str = "30d"):
        """Fetch product sentiment data from API."""
        try:
            url = f"{self.api_base_url}/sentiment/product/{asin}"
            params = {}
            if feature:
                params['feature'] = feature
            if window:
                params['window'] = window
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def fetch_top_features(self, asin: str, limit: int = 10, sort: str = "score"):
        """Fetch top features for a product."""
        try:
            url = f"{self.api_base_url}/sentiment/product/{asin}/top-features"
            params = {'limit': limit, 'sort': sort}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching top features: {e}")
            return None
    
    def fetch_search_results(self, query: str, category: str = None, limit: int = 20):
        """Search for features across categories."""
        try:
            url = f"{self.api_base_url}/sentiment/search"
            params = {'query': query, 'limit': limit}
            if category:
                params['category'] = category
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error searching features: {e}")
            return None
    
    def get_sentiment_color(self, score: float) -> str:
        """Get color based on sentiment score."""
        if score > 0.3:
            return "#28a745"  # Green
        elif score < -0.3:
            return "#dc3545"  # Red
        else:
            return "#6c757d"  # Gray
    
    def get_sentiment_label(self, score: float) -> str:
        """Get sentiment label based on score."""
        if score > 0.5:
            return "Very Positive"
        elif score > 0.1:
            return "Positive"
        elif score > -0.1:
            return "Neutral"
        elif score > -0.5:
            return "Negative"
        else:
            return "Very Negative"


def main():
    """Main dashboard function."""
    dashboard = SentimentDashboard()
    
    # Header
    st.markdown('<h1 class="main-header">📊 Product Sentiment Insights Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔍 Search & Filter")
    
    # Search options
    search_type = st.sidebar.radio(
        "Search Type",
        ["Product Analysis", "Feature Search"],
        help="Choose between analyzing a specific product or searching for features"
    )
    
    if search_type == "Product Analysis":
        # Product analysis section
        st.sidebar.subheader("Product Analysis")
        
        asin = st.sidebar.text_input(
            "Product ASIN",
            value=DEFAULT_ASIN,
            help="Enter the Amazon Standard Identification Number"
        )
        
        feature_filter = st.sidebar.text_input(
            "Specific Feature (Optional)",
            help="Filter by a specific feature (e.g., battery_life, camera_quality)"
        )
        
        time_window = st.sidebar.selectbox(
            "Time Window",
            ["All Time", "7d", "30d", "90d", "1y", "10y"],
            index=0,
            help="Time window for analysis"
        )
        
        # Fetch and display data
        if st.sidebar.button("Analyze Product", type="primary"):
            with st.spinner("Fetching product sentiment data..."):
                # Convert "All Time" to None to avoid time filtering
                window_param = None if time_window == "All Time" else time_window
                data = dashboard.fetch_product_sentiment(asin, feature_filter, window_param)
                
                if data and 'error' not in data:
                    display_product_analysis(data, dashboard)
                else:
                    st.error(f"Error: {data.get('error', 'Unknown error') if data else 'Failed to fetch data'}")
    
    else:
        # Feature search section
        st.sidebar.subheader("Feature Search")
        
        search_query = st.sidebar.text_input(
            "Search Query",
            help="Search for features across products"
        )
        
        category_filter = st.sidebar.selectbox(
            "Category (Optional)",
            ["All", "All_Beauty", "Electronics", "Home", "Sports"],
            help="Filter by product category"
        )
        
        search_limit = st.sidebar.slider(
            "Max Results",
            min_value=5,
            max_value=50,
            value=20,
            help="Maximum number of results to display"
        )
        
        if st.sidebar.button("Search Features", type="primary"):
            if search_query:
                with st.spinner("Searching features..."):
                    category = None if category_filter == "All" else category_filter
                    results = dashboard.fetch_search_results(search_query, category, search_limit)
                    
                    if results and 'error' not in results:
                        display_search_results(results, dashboard)
                    else:
                        st.error(f"Error: {results.get('error', 'Unknown error') if results else 'Failed to fetch data'}")
            else:
                st.warning("Please enter a search query")


def display_product_analysis(data, dashboard):
    """Display product analysis results."""
    st.header(f"📱 Product Analysis: {data.get('asin', 'Unknown')}")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Sentiment",
            f"{data.get('overall_sentiment', 0):.2f}",
            help="Average sentiment score (-1 to 1)"
        )
    
    with col2:
        st.metric(
            "Total Reviews",
            data.get('total_reviews', 0),
            help="Number of reviews analyzed"
        )
    
    with col3:
        st.metric(
            "Features Found",
            len(data.get('features', {})),
            help="Number of features identified"
        )
    
    with col4:
        last_updated = data.get('last_updated', 'Unknown')
        if last_updated != 'Unknown':
            last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            st.metric(
                "Last Updated",
                last_updated.strftime("%Y-%m-%d"),
                help="Last update timestamp"
            )
    
    # Features analysis
    features = data.get('features', {})
    if features:
        st.subheader("🎯 Feature Analysis")
        
        # Create features DataFrame
        features_data = []
        for feature_name, feature_data in features.items():
            features_data.append({
                'Feature': feature_name.replace('_', ' ').title(),
                'Sentiment Score': feature_data['score'],
                'Mentions': feature_data['count'],
                'Positive Snippets': len(feature_data.get('positive_snippets', [])),
                'Negative Snippets': len(feature_data.get('negative_snippets', [])),
                'Trend': feature_data.get('trend', 'stable')
            })
        
        df = pd.DataFrame(features_data)
        
        # Sort by sentiment score
        df = df.sort_values('Sentiment Score', key=abs, ascending=False)
        
        # Display features table
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Sentiment Score": st.column_config.NumberColumn(
                    "Sentiment Score",
                    help="Sentiment score (-1 to 1)",
                    format="%.2f"
                ),
                "Mentions": st.column_config.NumberColumn(
                    "Mentions",
                    help="Number of mentions"
                )
            }
        )
        
        # Sentiment distribution chart
        fig = px.bar(
            df,
            x='Feature',
            y='Sentiment Score',
            color='Sentiment Score',
            color_continuous_scale=['#dc3545', '#6c757d', '#28a745'],
            title="Feature Sentiment Scores",
            labels={'Sentiment Score': 'Sentiment Score', 'Feature': 'Feature'}
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Mentions vs Sentiment scatter plot
        fig = px.scatter(
            df,
            x='Mentions',
            y='Sentiment Score',
            size='Mentions',
            color='Sentiment Score',
            hover_name='Feature',
            title="Mentions vs Sentiment Score",
            color_continuous_scale=['#dc3545', '#6c757d', '#28a745']
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed feature analysis
        st.subheader("🔍 Detailed Feature Analysis")
        
        selected_feature = st.selectbox(
            "Select a feature for detailed analysis",
            list(features.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if selected_feature:
            feature_data = features[selected_feature]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Positive Feedback")
                positive_snippets = feature_data.get('positive_snippets', [])
                if positive_snippets:
                    for snippet in positive_snippets[:5]:  # Show top 5
                        st.markdown(f"• {snippet}")
                else:
                    st.info("No positive feedback found")
            
            with col2:
                st.markdown("#### Negative Feedback")
                negative_snippets = feature_data.get('negative_snippets', [])
                if negative_snippets:
                    for snippet in negative_snippets[:5]:  # Show top 5
                        st.markdown(f"• {snippet}")
                else:
                    st.info("No negative feedback found")
    
    else:
        st.warning("No features found for this product")


def display_search_results(results, dashboard):
    """Display feature search results."""
    st.header("🔍 Feature Search Results")
    
    query = results.get('query', '')
    search_results = results.get('results', [])
    total_results = results.get('total_results', 0)
    
    st.subheader(f"Search: '{query}' ({total_results} results)")
    
    if search_results:
        # Create results DataFrame
        df = pd.DataFrame(search_results)
        df['Feature'] = df['feature'].str.replace('_', ' ').str.title()
        df['Score'] = df['score'].round(2)
        df['Count'] = df['count']
        df['Category'] = df['category']
        
        # Display results table
        st.dataframe(
            df[['Feature', 'Score', 'Count', 'Category', 'asin']],
            use_container_width=True,
            column_config={
                "Score": st.column_config.NumberColumn(
                    "Score",
                    help="Sentiment score (-1 to 1)",
                    format="%.2f"
                ),
                "asin": st.column_config.TextColumn(
                    "ASIN",
                    help="Product ASIN"
                )
            }
        )
        
        # Results visualization
        if len(search_results) > 1:
            fig = px.scatter(
                df,
                x='Count',
                y='Score',
                size='Count',
                color='Category',
                hover_name='Feature',
                title=f"Feature Search Results: '{query}'",
                labels={'Score': 'Sentiment Score', 'Count': 'Mentions'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No results found for your search query")


if __name__ == "__main__":
    main()
