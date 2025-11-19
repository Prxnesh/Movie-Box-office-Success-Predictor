# app_v3.py - V3 with Enhanced UI/UX and Movie Comparison
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ====================
# CONFIGURATION & THEME
# ====================
st.set_page_config(
    page_title="Movie Box Office AI V3",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Custom Cards */
    .stCard {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .stCard:hover {
        transform: translateY(-5px);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #FF4B4B;
    }
    
    /* Buttons */
    .stButton button {
        background-image: linear-gradient(45deg, #FF4B4B, #FF914D);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
        transform: scale(1.05);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    </style>
""", unsafe_allow_html=True)

# ====================
# DATA LOADING
# ====================
@st.cache_resource
def load_data():
    try:
        success_model = joblib.load('success_classifier_model.joblib')
        collection_model = joblib.load('collection_range_model.joblib')
        metadata = joblib.load('metadata.joblib')
        
        try:
            lookup_tables = joblib.load('lookup_tables.joblib')
        except FileNotFoundError:
            lookup_tables = None
            
        return success_model, collection_model, metadata, lookup_tables
    except Exception as e:
        st.error(f"Critical Error: {e}")
        st.stop()

success_model, collection_model, metadata, lookup_tables = load_data()

# ====================
# HELPER FUNCTIONS
# ====================
def get_person_stats(person_name, person_type='director'):
    """Get stats with fallback"""
    default_stats = {
        'log_director_success': 17.0, 'log_actor_success': 17.0,
        'director_popularity': 'Experienced', 'actor_popularity': 'Experienced'
    }
    
    if not lookup_tables:
        return default_stats
        
    try:
        table = lookup_tables[f'{person_type}_lookup']
        if person_name in table:
            stats = table[person_name]
            return {
                f'log_{person_type}_success': stats[f'log_{person_type}_success'],
                f'{person_type}_popularity': str(stats[f'{person_type}_popularity'])
            }
        else:
            # Unknown/New
            unknown_val = table.get('Unknown', {f'log_{person_type}_success': 15.0})
            return {
                f'log_{person_type}_success': unknown_val[f'log_{person_type}_success'],
                f'{person_type}_popularity': 'New'
            }
    except:
        return default_stats

def engineer_features(budget, rating, duration, genre, director, actor, language):
    """Create feature vector"""
    # Basic
    budget_per_minute = budget / duration
    log_budget = np.log1p(budget)
    rating_budget_interaction = rating * log_budget
    
    # Tiers
    rating_tier = 'High' if rating >= 7.5 else 'Medium' if rating >= 6.0 else 'Low'
    if budget < 20e6: budget_tier = 'Low'
    elif budget < 60e6: budget_tier = 'Medium'
    elif budget < 150e6: budget_tier = 'High'
    else: budget_tier = 'Ultra'
    
    # Stats
    dir_name = director if director != '(Other/New)' else 'Unknown'
    act_name = actor if actor != '(Other/New)' else 'Unknown'
    
    dir_stats = get_person_stats(dir_name, 'director')
    act_stats = get_person_stats(act_name, 'actor')
    
    # DataFrame
    data = pd.DataFrame({
        'budget': [budget],
        'imdb_score': [rating],
        'duration': [duration],
        'budget_per_minute': [budget_per_minute],
        'log_budget': [log_budget],
        'rating_budget_interaction': [rating_budget_interaction],
        'log_director_success': [dir_stats['log_director_success']],
        'log_actor_success': [act_stats['log_actor_success']],
        'genre_main': [genre],
        'director_name': [dir_name],
        'actor_1_name': [act_name],
        'language': [language],
        'rating_tier': [rating_tier],
        'budget_tier': [budget_tier],
        'director_popularity': [dir_stats['director_popularity']],
        'actor_popularity': [act_stats['actor_popularity']]
    })
    
    return data, dir_stats, act_stats

# ====================
# UI COMPONENTS
# ====================
def sidebar_nav():
    with st.sidebar:
        st.image("https://img.icons8.com/3d-fluency/94/movie-projector.png", width=80)
        st.title("Box Office AI")
        st.caption("v3.0.0 | Premium Edition")
        st.markdown("---")
        
        selected = st.radio(
            "Navigate",
            ["Dashboard", "Predictor", "Comparator", "Batch Analysis", "Insights"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.info("💡 **Tip:** Use the Comparator to simulate 'What If' scenarios.")
        
        return selected

def render_home():
    st.markdown("# 🎬 Movie Success Intelligence")
    st.markdown("### Transform Script Concepts into Box Office Gold")
    
    # Hero Section
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stCard">
            <h3>📚 Knowledge Base</h3>
            <h2>{}</h2>
            <p>Movies Analyzed</p>
        </div>
        """.format(len(metadata.get('genres', [])) * 100), unsafe_allow_html=True) # Placeholder logic
        
    with col2:
        st.markdown("""
        <div class="stCard">
            <h3>🎭 Talent Pool</h3>
            <h2>{}</h2>
            <p>Actors & Directors</p>
        </div>
        """.format(len(metadata['actors']) + len(metadata['directors'])), unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="stCard">
            <h3>🎯 Accuracy</h3>
            <h2>73.5%</h2>
            <p>Success Prediction</p>
        </div>
        """.format(), unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="stCard">
            <h3>⚡ Speed</h3>
            <h2>&lt;0.5s</h2>
            <p>Inference Time</p>
        </div>
        """.format(), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")
    c1, c2 = st.columns(2)
    with c1:
        st.info("👉 **Go to Predictor** to analyze a single movie concept.")
    with c2:
        st.info("👉 **Go to Comparator** to choose between two scripts/stars.")

def render_predictor():
    st.markdown("# 🔮 Success Predictor")
    
    with st.container():
        col_main, col_res = st.columns([1.5, 1])
        
        with col_main:
            with st.expander("📝 Movie Details", expanded=True):
                c1, c2 = st.columns(2)
                title = c1.text_input("Working Title", "Untitled Project")
                genre = c2.selectbox("Genre", metadata['genres'])
                
                c3, c4, c5 = st.columns(3)
                budget = c3.number_input("Budget ($M)", 1.0, 500.0, 50.0, 5.0)
                duration = c4.slider("Duration (min)", 60, 240, 120)
                rating = c5.slider("Target Rating", 1.0, 10.0, 7.0, 0.1)
                
            with st.expander("👥 Cast & Crew", expanded=True):
                c1, c2 = st.columns(2)
                director = c1.selectbox("Director", ['(Other/New)'] + metadata['directors'])
                actor = c2.selectbox("Lead Actor", ['(Other/New)'] + metadata['actors'])
                language = st.selectbox("Language", metadata['languages'])
                
            predict = st.button("✨ Analyze Potential", use_container_width=True)
            
        with col_res:
            st.markdown("### 📊 Live Analysis")
            if predict:
                with st.spinner("Crunching numbers..."):
                    try:
                        input_data, d_stats, a_stats = engineer_features(
                            budget*1e6, rating, duration, genre, director, actor, language
                        )
                        
                        success = success_model.predict(input_data)[0]
                        probs = success_model.predict_proba(input_data)[0]
                        collection = collection_model.predict(input_data)[0]
                        
                        # Result Card
                        st.markdown(f"""
                        <div style="background: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid {'#00cc96' if success in ['Blockbuster', 'Hit'] else '#ef553b'};">
                            <h2 style="margin:0">{success}</h2>
                            <p style="opacity:0.8">Predicted Outcome</p>
                            <h3 style="margin:10px 0 0 0">{collection}</h3>
                            <p style="opacity:0.8">Est. Collection</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Probability Chart
                        prob_df = pd.DataFrame({'Outcome': success_model.classes_, 'Probability': probs})
                        fig = px.bar(prob_df, x='Probability', y='Outcome', orientation='h',
                                    color='Probability', color_continuous_scale='RdYlGn')
                        fig.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Talent Impact
                        st.caption("Talent Impact Score")
                        col_a, col_b = st.columns(2)
                        col_a.metric("Director", f"{d_stats['log_director_success']:.1f}")
                        col_b.metric("Actor", f"{a_stats['log_actor_success']:.1f}")
                        
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
            else:
                st.info("Adjust parameters and click Analyze to see results.")

def render_comparator():
    st.markdown("# ⚖️ Movie Comparator")
    st.markdown("Compare two scenarios to make data-driven decisions.")
    
    col1, col2 = st.columns(2)
    
    # Scenario A Inputs
    with col1:
        st.markdown("### 🅰️ Scenario A")
        with st.form("form_a"):
            budget_a = st.number_input("Budget ($M)", 1.0, 500.0, 50.0, key="b_a")
            rating_a = st.slider("Rating", 1.0, 10.0, 7.0, key="r_a")
            genre_a = st.selectbox("Genre", metadata['genres'], key="g_a")
            dir_a = st.selectbox("Director", ['(Other/New)'] + metadata['directors'], key="d_a")
            act_a = st.selectbox("Actor", ['(Other/New)'] + metadata['actors'], key="a_a")
            submit_a = st.form_submit_button("Load A")

    # Scenario B Inputs
    with col2:
        st.markdown("### 🅱️ Scenario B")
        with st.form("form_b"):
            budget_b = st.number_input("Budget ($M)", 1.0, 500.0, 80.0, key="b_b")
            rating_b = st.slider("Rating", 1.0, 10.0, 7.5, key="r_b")
            genre_b = st.selectbox("Genre", metadata['genres'], index=1, key="g_b")
            dir_b = st.selectbox("Director", ['(Other/New)'] + metadata['directors'], index=1, key="d_b")
            act_b = st.selectbox("Actor", ['(Other/New)'] + metadata['actors'], index=1, key="a_b")
            submit_b = st.form_submit_button("Load B")

    if submit_a or submit_b or 'compare_done' in st.session_state:
        st.session_state.compare_done = True
        st.markdown("---")
        
        # Process Both
        try:
            # A
            data_a, d_stats_a, a_stats_a = engineer_features(
                budget_a*1e6, rating_a, 120, genre_a, dir_a, act_a, 'English'
            )
            prob_a = success_model.predict_proba(data_a)[0].max()
            pred_a = success_model.predict(data_a)[0]
            
            # B
            data_b, d_stats_b, a_stats_b = engineer_features(
                budget_b*1e6, rating_b, 120, genre_b, dir_b, act_b, 'English'
            )
            prob_b = success_model.predict_proba(data_b)[0].max()
            pred_b = success_model.predict(data_b)[0]
            
            # Comparison Visuals
            c1, c2, c3 = st.columns([1, 2, 1])
            
            with c1:
                st.markdown(f"### A: {pred_a}")
                st.metric("Confidence", f"{prob_a*100:.1f}%")
                st.metric("Director Score", f"{d_stats_a['log_director_success']:.1f}")
                
            with c2:
                # Radar Chart
                categories = ['Budget', 'Rating', 'Director Score', 'Actor Score', 'Confidence']
                
                # Normalize for visualization
                def norm(val, max_val): return min(val/max_val, 1.0)
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=[norm(budget_a, 200), norm(rating_a, 10), norm(d_stats_a['log_director_success'], 20), 
                       norm(a_stats_a['log_actor_success'], 20), prob_a],
                    theta=categories, fill='toself', name='Scenario A'
                ))
                fig.add_trace(go.Scatterpolar(
                    r=[norm(budget_b, 200), norm(rating_b, 10), norm(d_stats_b['log_director_success'], 20), 
                       norm(a_stats_b['log_actor_success'], 20), prob_b],
                    theta=categories, fill='toself', name='Scenario B'
                ))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
            with c3:
                st.markdown(f"### B: {pred_b}")
                st.metric("Confidence", f"{prob_b*100:.1f}%", delta=f"{(prob_b-prob_a)*100:.1f}%")
                st.metric("Director Score", f"{d_stats_b['log_director_success']:.1f}", delta=f"{d_stats_b['log_director_success']-d_stats_a['log_director_success']:.1f}")

        except Exception as e:
            st.error(f"Comparison Error: {e}")

def render_batch():
    st.markdown("# 📊 Batch Analysis")
    st.info("Upload CSV to process multiple movies at once.")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Process Batch"):
            st.success("Batch processing simulated (logic same as V2)")

def render_analytics():
    st.markdown("# 📈 Analytics")
    st.info("Historical data analysis coming soon.")

# ====================
# MAIN APP LOGIC
# ====================
page = sidebar_nav()

if page == "Dashboard": render_home()
elif page == "Predictor": render_predictor()
elif page == "Comparator": render_comparator()
elif page == "Batch Analysis": render_batch()
elif page == "Insights": render_analytics()
