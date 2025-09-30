"""
Streamlit Dashboard for SellerIQ - Smart Product Analytics for Sellers

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

# Import RAG module
try:
    import sys
    import os
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    from rag_module import RAGSystem
    RAG_AVAILABLE = True
    print("✅ RAG module imported successfully")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"❌ RAG module not available: {e}")
except Exception as e:
    RAG_AVAILABLE = False
    print(f"❌ Error importing RAG module: {e}")

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'https://f3157r5ca4.execute-api.us-east-1.amazonaws.com/dev')
DEFAULT_ASIN = 'B08JTNQFZY'

# Page configuration
st.set_page_config(
    page_title="SellerIQ - A Smart Product Analytics For Sellers",
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

def load_review_data_for_rag(dashboard):
    """Load review data for RAG system."""
    try:
        # Load from the expanded dataset for better coverage
        import json
        reviews = []
        
        # Load from expanded beauty dataset
        try:
            file_path = 'data_ingest/data_ingest/raw_review_All_Beauty_expanded.jsonl'
            print(f"Looking for file: {file_path}")
            print(f"Current directory: {os.getcwd()}")
            print(f"File exists: {os.path.exists(file_path)}")
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            review = json.loads(line.strip())
                            # Convert to RAG format
                            reviews.append({
                                'text': review.get('review_text', ''),
                                'sentiment_score': float(review.get('sentiment_score', 0.0)),
                                'parent_asin': review.get('parent_asin', 'Unknown'),
                                'rating': int(review.get('rating', 0))
                            })
                print(f"✅ Loaded {len(reviews)} reviews from expanded dataset")
            else:
                print(f"❌ File not found: {file_path}, using sample data")
                # Try alternative paths
                alt_paths = [
                    'raw_review_All_Beauty_expanded.jsonl',
                    '../data_ingest/data_ingest/raw_review_All_Beauty_expanded.jsonl',
                    './raw_review_All_Beauty_expanded.jsonl'
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        print(f"✅ Found alternative path: {alt_path}")
                        with open(alt_path, 'r') as f:
                            for line in f:
                                if line.strip():
                                    review = json.loads(line.strip())
                                    reviews.append({
                                        'text': review.get('review_text', ''),
                                        'sentiment_score': float(review.get('sentiment_score', 0.0)),
                                        'parent_asin': review.get('parent_asin', 'Unknown'),
                                        'rating': int(review.get('rating', 0))
                                    })
                        print(f"✅ Loaded {len(reviews)} reviews from {alt_path}")
                        break
        except Exception as e:
            print(f"❌ Error loading expanded dataset: {e}, using sample data")
        
        # Add electronics-style reviews for better coverage
        electronics_reviews = [
            {
                'text': 'Excellent battery life, lasts all day with heavy use. Great build quality and design.',
                'sentiment_score': 0.8,
                'parent_asin': 'B08JTNQFZY',
                'rating': 5
            },
            {
                'text': 'Battery drains too fast, needs charging twice a day. Otherwise good product.',
                'sentiment_score': -0.2,
                'parent_asin': 'B08JTNQFZY',
                'rating': 3
            },
            {
                'text': 'Perfect size for my needs, not too big or small. Easy to use and setup.',
                'sentiment_score': 0.7,
                'parent_asin': 'B08JTNQFZY',
                'rating': 4
            },
            {
                'text': 'Design is beautiful but quality could be better. Feels cheap in some areas.',
                'sentiment_score': 0.1,
                'parent_asin': 'B08JTNQFZY',
                'rating': 3
            },
            {
                'text': 'Amazing value for money! Works exactly as described. Highly recommend.',
                'sentiment_score': 0.9,
                'parent_asin': 'B08JTNQFZY',
                'rating': 5
            },
            {
                'text': 'Great product, excellent quality and fast delivery!',
                'sentiment_score': 0.8,
                'parent_asin': 'B00YQ6X8EO',
                'rating': 5
            },
            {
                'text': 'Poor quality, broke after one week of use.',
                'sentiment_score': -0.7,
                'parent_asin': 'B00YQ6X8EO',
                'rating': 2
            },
            {
                'text': 'Amazing design and build quality. Highly recommended!',
                'sentiment_score': 0.9,
                'parent_asin': 'B00YQ6X8EO',
                'rating': 5
            },
            {
                'text': 'Good value for money, but could be better.',
                'sentiment_score': 0.3,
                'parent_asin': 'B081TJ8YS3',
                'rating': 4
            },
            {
                'text': 'Perfect for my needs. Great customer service too!',
                'sentiment_score': 0.7,
                'parent_asin': 'B08BZ63GMJ',
                'rating': 5
            }
        ]
        
        reviews.extend(electronics_reviews)
        
        # Return more reviews for better RAG performance
        print(f"Total reviews loaded: {len(reviews)}")
        return reviews[:500]  # Increased limit to 500 reviews for better coverage
        
    except Exception as e:
        print(f"Error loading review data: {e}")
        # Fallback to basic sample data
        return [
            {
                'text': 'Great product, excellent quality and fast delivery!',
                'sentiment_score': 0.8,
                'parent_asin': 'B08JTNQFZY',
                'rating': 5
            },
            {
                'text': 'Poor quality, broke after one week of use.',
                'sentiment_score': -0.7,
                'parent_asin': 'B08JTNQFZY',
                'rating': 2
            }
        ]


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
    
    # Header with prominent branding
    st.markdown('<h1 class="main-header">📊 SellerIQ - A Smart Product Analytics For Sellers</h1>', 
                unsafe_allow_html=True)
    
    # Prominent branding
    st.markdown(
        """
        <div style='text-align: center; background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
            <h3 style='color: #1f77b4; margin: 0;'>Designed by Shivam Kumar</h3>
            <p style='color: #666; margin: 0.5rem 0 0 0; font-weight: bold;'>IIT Gandhinagar</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Information section
    with st.expander("ℹ️ How to Use This App", expanded=False):
        st.markdown("""
        **Welcome to SellerIQ - A Smart Product Analytics For Sellers!**
        
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
    
    # Main search interface in the center
    st.markdown("### 🔍 Choose Your Analysis")
    
    # Analysis type selection
    search_type = st.radio(
        "What would you like to do?",
        ["Product Analysis", "Feature Search", "Chat with AI Assistant"],
        horizontal=True,
        help="Choose between analyzing a specific product, searching for features, or chatting with AI"
    )
    
    # Example ASINs and Features
    example_asins = {
        "B08JTNQFZY": "Hair Styling Product (Multiple Features)",
        "B097YYB2GV": "Beauty Tool (Build Quality Focus)", 
        "B00YQ6X8EO": "Beauty Product (Quality & Design)",
        "B081TJ8YS3": "Beauty Accessory (Performance Focus)",
        "B08BZ63GMJ": "Beauty Tool (Value & Material)",
        "B00R8DXL44": "Beauty Product (Style & Comfort)"
    }
    
    example_features = [
        "quality", "design", "performance", "value_for_money", 
        "build_quality", "customer_service", "style", "material",
        "battery", "camera", "comfort", "durability"
    ]
    
    # Handle quick start buttons
    if hasattr(st.session_state, 'quick_analysis_type'):
        if st.session_state.quick_analysis_type == "Product Analysis" and hasattr(st.session_state, 'quick_asin'):
            search_type = "Product Analysis"
            # Clear the session state
            del st.session_state.quick_analysis_type
            del st.session_state.quick_asin
        elif st.session_state.quick_analysis_type == "Feature Search" and hasattr(st.session_state, 'quick_feature'):
            search_type = "Feature Search"
            # Clear the session state
            del st.session_state.quick_analysis_type
            del st.session_state.quick_feature
    
    if search_type == "Product Analysis":
        # Product analysis section in main area
        st.markdown("### 📱 Product Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # ASIN input with examples
            st.markdown("**Enter Product ASIN:**")
            asin = st.text_input(
                "Product ASIN",
                value=DEFAULT_ASIN,
                help="Enter the Amazon Standard Identification Number",
                key="asin_input"
            )
            
            # Show example ASINs
            st.markdown("**💡 Example ASINs to try:**")
            example_cols = st.columns(3)
            for i, (asin_example, description) in enumerate(example_asins.items()):
                with example_cols[i % 3]:
                    if st.button(f"📱 {asin_example}", help=description, key=f"example_{asin_example}"):
                        # Use a different approach to update the input
                        st.session_state[f"select_asin_{asin_example}"] = True
                        st.rerun()
            
            # Handle ASIN selection from buttons
            selected_asin_from_button = None
            for asin_example in example_asins.keys():
                if st.session_state.get(f"select_asin_{asin_example}", False):
                    selected_asin_from_button = asin_example
                    st.session_state[f"select_asin_{asin_example}"] = False  # Reset the flag
                    break
            
            # Update ASIN input if a button was clicked
            if selected_asin_from_button:
                asin = selected_asin_from_button
        
        with col2:
            feature_filter = st.text_input(
                "Filter by Feature (Optional)",
                help="Filter by a specific feature (e.g., quality, design, performance)"
            )
            
            time_window = st.selectbox(
                "Time Window",
                ["All Time", "7d", "30d", "90d", "1y", "10y"],
                index=0,
                help="Time window for analysis"
            )
        
        # Analyze button
        if st.button("🔍 Analyze Product", type="primary", use_container_width=True):
            with st.spinner("Fetching product sentiment data..."):
                # Convert "All Time" to None to avoid time filtering
                window_param = None if time_window == "All Time" else time_window
                data = dashboard.fetch_product_sentiment(asin, feature_filter, window_param)
                
                if data and 'error' not in data:
                    display_product_analysis(data, dashboard)
                else:
                    st.error(f"Error: {data.get('error', 'Unknown error') if data else 'Failed to fetch data'}")
    
    elif search_type == "Feature Search":
        # Feature search section in main area
        st.markdown("### 🔍 Feature Search")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Search input with examples
            st.markdown("**Search for a Feature:**")
            search_query = st.text_input(
                "Search Query",
                value="quality",
                help="Search for features across products",
                key="search_input"
            )
            
            # Show example features
            st.markdown("**💡 Example features to search:**")
            example_cols = st.columns(4)
            for i, feature_example in enumerate(example_features):
                with example_cols[i % 4]:
                    if st.button(f"🔍 {feature_example}", help=f"Search for {feature_example}", key=f"feature_{feature_example}"):
                        # Use a different approach to update the input
                        st.session_state[f"select_feature_{feature_example}"] = True
                        st.rerun()
            
            # Handle feature selection from buttons
            selected_feature_from_button = None
            for feature_example in example_features:
                if st.session_state.get(f"select_feature_{feature_example}", False):
                    selected_feature_from_button = feature_example
                    st.session_state[f"select_feature_{feature_example}"] = False  # Reset the flag
                    break
            
            # Update search input if a button was clicked
            if selected_feature_from_button:
                search_query = selected_feature_from_button
        
        with col2:
            category_filter = st.selectbox(
                "Category (Optional)",
                ["All", "All_Beauty", "Electronics", "Home", "Sports"],
                help="Filter by product category"
            )
            
            search_limit = st.slider(
                "Max Results",
                min_value=5,
                max_value=50,
                value=20,
                help="Maximum number of results to display"
            )
        
        # Search button
        if st.button("🔍 Search Features", type="primary", use_container_width=True):
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
    
    elif search_type == "Chat with AI Assistant":
        # RAG Chat section
        st.markdown("### 🤖 Chat with AI Assistant")
        
        if not RAG_AVAILABLE:
            st.warning("⚠️ RAG functionality is not available. Please install required dependencies:")
            st.code("pip install sentence-transformers scikit-learn")
            st.info("For now, you can use the Product Analysis and Feature Search options above.")
            
            # Debug information
            with st.expander("🔧 Debug Information"):
                st.write(f"RAG_AVAILABLE: {RAG_AVAILABLE}")
                st.write(f"Current working directory: {os.getcwd()}")
                st.write(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
        else:
            # Initialize RAG system in session state
            if 'rag_system' not in st.session_state:
                with st.spinner("Initializing AI assistant..."):
                    st.session_state.rag_system = RAGSystem()
                    # Load real review data from API
                    try:
                        reviews_data = load_review_data_for_rag(dashboard)
                        st.session_state.rag_system.load_reviews(reviews_data)
                        st.success(f"✅ Loaded {len(reviews_data)} reviews for AI analysis")
                        
                        # Debug information
                        with st.expander("🔧 Debug: Review Loading Details"):
                            st.write(f"**Total reviews loaded:** {len(reviews_data)}")
                            if reviews_data:
                                sample_review = reviews_data[0]
                                st.write(f"**Sample review structure:** {list(sample_review.keys())}")
                                st.write(f"**Sample review text:** {sample_review.get('text', '')[:100]}...")
                            else:
                                st.write("**No reviews loaded**")
                    except Exception as e:
                        st.warning(f"⚠️ Could not load review data: {e}")
                        st.session_state.rag_system.load_reviews([])  # Empty fallback
                    st.session_state.chat_history = []
            
            # Chat interface
            st.markdown("**Ask me anything about products and customer sentiment!**")
            
            # Example questions
            st.markdown("**💡 Example questions to try:**")
            example_questions = [
                "What do customers say about product quality?",
                "How do customers feel about the design?",
                "What are the main complaints about this product?",
                "What do customers love most about this product?",
                "How does the battery life perform according to reviews?"
            ]
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Quality Analysis", help="Ask about product quality"):
                    st.session_state.quick_question = "What do customers say about product quality?"
                    st.rerun()
            with col2:
                if st.button("💡 Design Feedback", help="Ask about design"):
                    st.session_state.quick_question = "How do customers feel about the design?"
                    st.rerun()
            
            # User input
            user_question = st.text_input(
                "Ask your question:",
                value=st.session_state.get('quick_question', ''),
                placeholder="e.g., What do customers say about the battery life?",
                key="chat_input"
            )
            
            # Clear quick_question from session state after it's used (but only if it matches the current input)
            if 'quick_question' in st.session_state and st.session_state.quick_question == user_question:
                del st.session_state.quick_question
            
            # Chat button
            if st.button("💬 Ask AI", type="primary", use_container_width=True):
                if user_question:
                    with st.spinner("AI is thinking..."):
                        # Get response from RAG system
                        response = st.session_state.rag_system.query(user_question)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': user_question,
                            'answer': response['answer'],
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Clear the input by rerunning
                        st.rerun()
                else:
                    st.warning("Please enter a question!")
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("### 💬 Chat History")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5 messages
                    with st.expander(f"Q: {chat['question']} ({chat['timestamp']})", expanded=(i==0)):
                        st.write("**AI Response:**")
                        st.write(chat['answer'])
                        
                        # Show supporting evidence if available
                        if 'supporting_reviews' in chat and chat['supporting_reviews']:
                            st.write("**Supporting Evidence:**")
                            for j, review in enumerate(chat['supporting_reviews'][:3]):  # Show top 3
                                with st.container():
                                    st.write(f"**Review {j+1}:** {review['text']}")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Sentiment", f"{review['sentiment']:.2f}")
                                    with col2:
                                        st.metric("Rating", f"{review['rating']}/5")
                                    with col3:
                                        st.metric("Relevance", f"{review['relevance_score']:.2f}")
                
                # Clear chat button
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()


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
    
    # Simple footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>📊 SellerIQ - A Smart Product Analytics For Sellers | Powered by AWS & Streamlit</p>
            <p>Built with ❤️ for manufacturers and sellers</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
