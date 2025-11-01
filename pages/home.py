# pages/home.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def show(metadata):
    # Hero Section
    st.markdown('<h1 class="main-header">🎬 Movie Box Office Success Predictor V2</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
        Predict box office success with AI-powered machine learning models<br>
        <strong>73.45% Success Rate</strong> | <strong>79.12% Collection Accuracy</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown("---")
    st.markdown("## ✨ Key Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <div style='font-size: 3rem;'>🎯</div>
            <h3>Single Prediction</h3>
            <p>Predict success for individual movie concepts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <div style='font-size: 3rem;'>📊</div>
            <h3>Batch Processing</h3>
            <p>Upload CSV and predict multiple movies at once</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <div style='font-size: 3rem;'>📈</div>
            <h3>Analytics</h3>
            <p>Visualize trends and model insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <div style='font-size: 3rem;'>🔍</div>
            <h3>Compare</h3>
            <p>Compare different movie scenarios</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats Dashboard
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h2 style='margin: 0;'>🎬</h2>
            <h3 style='margin: 0.5rem 0;'>{}</h3>
            <p style='margin: 0; opacity: 0.9;'>Genres Available</p>
        </div>
        """.format(len(metadata['genres'])), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card success-card'>
            <h2 style='margin: 0;'>🎥</h2>
            <h3 style='margin: 0.5rem 0;'>{}</h3>
            <p style='margin: 0; opacity: 0.9;'>Directors in Database</p>
        </div>
        """.format(len(metadata['directors'])), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card warning-card'>
            <h2 style='margin: 0;'>⭐</h2>
            <h3 style='margin: 0.5rem 0;'>{}</h3>
            <p style='margin: 0; opacity: 0.9;'>Actors Tracked</p>
        </div>
        """.format(len(metadata['actors'])), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card info-card'>
            <h2 style='margin: 0;'>🌍</h2>
            <h3 style='margin: 0.5rem 0;'>{}</h3>
            <p style='margin: 0; opacity: 0.9;'>Languages Supported</p>
        </div>
        """.format(len(metadata['languages'])), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Genre Distribution Chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎭 Top Genres")
        genre_data = pd.DataFrame({
            'Genre': metadata['genres'][:10],
            'Count': [100, 85, 75, 65, 60, 55, 50, 45, 40, 35]  # Sample data
        })
        fig = px.bar(genre_data, x='Count', y='Genre', orientation='h',
                     color='Count', color_continuous_scale='Viridis')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Success Distribution")
        success_data = pd.DataFrame({
            'Category': ['Blockbuster', 'Hit', 'Flop'],
            'Count': [488, 1296, 2095]
        })
        fig = px.pie(success_data, values='Count', names='Category',
                     color_discrete_sequence=['#38ef7d', '#f5576c', '#667eea'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Getting Started
    st.markdown("---")
    st.markdown("## 🚀 Getting Started")
    
    st.info("""
    **Step 1:** Navigate to **Single Prediction** to predict one movie at a time  
    **Step 2:** Use **Batch Predictions** to analyze multiple movies from a CSV file  
    **Step 3:** Explore **Analytics** to understand model behavior and trends  
    **Step 4:** Use **Compare Movies** to test different scenarios side-by-side
    """)
    
    # Recent Predictions
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("## 📝 Recent Predictions")
        recent_df = pd.DataFrame(st.session_state.prediction_history[-5:])
        st.dataframe(recent_df, use_container_width=True, hide_index=True)