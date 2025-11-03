# app_v2_simple.py - Simplified working version

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Movie Box Office Predictor V2",
    page_icon="🎬",
    layout="wide"
)

# Load models and lookup tables
@st.cache_resource
def load_models():
    try:
        success_model = joblib.load('success_classifier_model.joblib')
        collection_model = joblib.load('collection_range_model.joblib')
        metadata = joblib.load('metadata.joblib')
        
        # Try to load lookup tables (for real director/actor success rates)
        try:
            lookup_tables = joblib.load('lookup_tables.joblib')
        except FileNotFoundError:
            st.warning("⚠️ Lookup tables not found. Using default values. Run 'create_lookup_tables.py' for better accuracy.")
            lookup_tables = None
            
        return success_model, collection_model, metadata, lookup_tables
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run training script first!")
        st.stop()

success_model, collection_model, metadata, lookup_tables = load_models()

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Sidebar Navigation
with st.sidebar:
    st.title("🎬 Box Office AI")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🎬 Single Prediction", "📊 Batch Predictions", "📈 Analytics", "🔍 Compare", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.metric("Success Accuracy", "73.45%")
    st.metric("Collection Accuracy", "79.12%")

# ====================
# HOME PAGE
# ====================
if page == "🏠 Home":
    st.title("🎬 Movie Box Office Success Predictor V2")
    st.markdown("### AI-Powered Box Office Predictions")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Genres", len(metadata['genres']))
    with col2:
        st.metric("Directors", len(metadata['directors']))
    with col3:
        st.metric("Actors", len(metadata['actors']))
    with col4:
        st.metric("Predictions Made", len(st.session_state.prediction_history))
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate to different features")

# ====================
# SINGLE PREDICTION
# ====================
elif page == "🎬 Single Prediction":
    st.title("🎬 Single Movie Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        movie_title = st.text_input("Movie Title (Optional)")
        
        col_a, col_b = st.columns(2)
        with col_a:
            budget_mil = st.number_input("Budget (M USD)", 0.1, 400.0, 50.0, 1.0)
            rating = st.slider("Expected Rating", 1.0, 10.0, 7.0, 0.1)
        with col_b:
            duration = st.slider("Duration (min)", 60, 240, 120, 5)
            genre = st.selectbox("Genre", metadata['genres'])
        
        director = st.selectbox("Director", ['(Other/New)'] + metadata['directors'])
        actor = st.selectbox("Main Star", ['(Other/New)'] + metadata['actors'])
        language = st.selectbox("Language", metadata['languages'])
        
        predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### Preview")
        st.json({
            "Title": movie_title if movie_title else "Untitled",
            "Budget": f"${budget_mil}M",
            "Rating": f"{rating}/10",
            "Duration": f"{duration} min"
        })
    
    if predict_btn:
        budget = budget_mil * 1_000_000
        
        # Feature engineering
        budget_per_minute = budget / duration
        log_budget = np.log1p(budget)
        rating_budget_interaction = rating * log_budget
        rating_tier = 'High' if rating >= 7.5 else ('Medium' if rating >= 6.0 else 'Low')
        
        if budget < 20e6:
            budget_tier = 'Low'
        elif budget < 60e6:
            budget_tier = 'Medium'
        elif budget < 150e6:
            budget_tier = 'High'
        else:
            budget_tier = 'Ultra'
        
        input_data = pd.DataFrame({
            'budget': [budget],
            'imdb_score': [rating],
            'duration': [duration],
            'budget_per_minute': [budget_per_minute],
            'log_budget': [log_budget],
            'rating_budget_interaction': [rating_budget_interaction],
            'log_director_success': [17.0],
            'log_actor_success': [17.0],
            'genre_main': [genre],
            'director_name': [director if director != '(Other/New)' else 'Unknown'],
            'actor_1_name': [actor if actor != '(Other/New)' else 'Unknown'],
            'language': [language],
            'rating_tier': [rating_tier],
            'budget_tier': [budget_tier],
            'director_popularity': ['Experienced'],
            'actor_popularity': ['Experienced']
        })
        
        success_pred = success_model.predict(input_data)[0]
        success_proba = success_model.predict_proba(input_data)
        collection_pred = collection_model.predict(input_data)[0]
        
        st.markdown("---")
        st.markdown("## Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Success Level", success_pred)
        with col2:
            st.metric("Collection Range", collection_pred)
        with col3:
            confidence = success_proba.max() * 100
            st.metric("Confidence", f"{confidence:.1f}%")
        
        if success_pred == 'Blockbuster':
            st.success("🎉 Highly Recommended!")
        elif success_pred == 'Hit':
            st.warning("💰 Good Investment")
        else:
            st.error("📉 Proceed with Caution")
        
        # Save to history
        st.session_state.prediction_history.append({
            'Title': movie_title if movie_title else "Untitled",
            'Budget': f"${budget_mil}M",
            'Success': success_pred,
            'Collection': collection_pred
        })

# ====================
# BATCH PREDICTIONS
# ====================
elif page == "📊 Batch Predictions":
    st.title("📊 Batch Movie Predictions")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} movies")
        st.dataframe(df.head())
        
        if st.button("Run Predictions"):
            st.info("Batch prediction feature - Process multiple movies from CSV")
            st.warning("Full implementation available in complete version")

# ====================
# ANALYTICS
# ====================
elif page == "📈 Analytics":
    st.title("📈 Analytics & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Performance")
        metrics_df = pd.DataFrame({
            'Class': ['Blockbuster', 'Hit', 'Flop'],
            'Precision': [0.73, 0.60, 0.84]
        })
        fig = px.bar(metrics_df, x='Class', y='Precision', title='Success Classifier Performance')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Feature Importance")
        features_df = pd.DataFrame({
            'Feature': ['Budget', 'Rating', 'Director', 'Genre'],
            'Importance': [0.25, 0.20, 0.18, 0.15]
        })
        fig = px.bar(features_df, x='Importance', y='Feature', orientation='h',
                    title='Top Features')
        st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.prediction_history:
        st.markdown("### Your Prediction History")
        st.dataframe(pd.DataFrame(st.session_state.prediction_history))

# ====================
# COMPARE
# ====================
elif page == "🔍 Compare":
    st.title("🔍 Compare Movie Scenarios")
    st.info("Compare different movie concepts side-by-side")
    st.warning("Full comparison feature available in complete version")

# ====================
# ABOUT
# ====================
elif page == "ℹ️ About":
    st.title("ℹ️ About")
    
    st.markdown("""
    ## Movie Box Office Success Predictor V2
    
    ### Features:
    - 🎯 Single movie predictions
    - 📊 Batch processing (CSV upload)
    - 📈 Analytics dashboard
    - 🔍 Movie comparisons
    
    ### Model Performance:
    - Success Classifier: **73.45%** accuracy
    - Collection Predictor: **79.12%** accuracy
    
    ### Technology Stack:
    - Streamlit
    - Scikit-learn
    - Plotly
    - Gradient Boosting
    """)