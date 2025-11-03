# app_v2.py - Movie Box Office Predictor V2

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Movie Box Office Predictor V2",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .warning-card {
        background: linear-gradient(135deg, #f857a6 0%, #ff5858 100%);
    }
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load models (cached for performance)
@st.cache_resource
def load_models():
    try:
        success_model = joblib.load('success_classifier_model.joblib')
        collection_model = joblib.load('collection_range_model.joblib')
        metadata = joblib.load('metadata.joblib')
        return success_model, collection_model, metadata
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run training script first!")
        st.stop()

success_model, collection_model, metadata = load_models()

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/movie-projector.png", width=100)
    st.title("🎬 Box Office AI")
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "Single Prediction", "Batch Predictions", "Analytics", "Compare Movies", "About"],
        icons=["house", "film", "files", "bar-chart", "shuffle", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "#ff4b4b", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#262730",
            },
            "nav-link-selected": {"background-color": "#ff4b4b"},
        }
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Stats")
    st.metric("Success Accuracy", "73.45%")
    st.metric("Collection Accuracy", "79.12%")
    st.metric("Total Predictions", len(st.session_state.prediction_history))

# Route to selected page
if selected == "Home":
    from pages import home
    home.show(metadata)
elif selected == "Single Prediction":
    from pages import single_prediction
    single_prediction.show(success_model, collection_model, metadata)
elif selected == "Batch Predictions":
    from pages import batch_predictions
    batch_predictions.show(success_model, collection_model, metadata)
elif selected == "Analytics":
    from pages import analytics
    analytics.show()
elif selected == "Compare Movies":
    from pages import compare
    compare.show(success_model, collection_model, metadata)
elif selected == "About":
    from pages import about
    about.show()