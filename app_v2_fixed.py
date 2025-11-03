# app_v2_fixed.py - Fixed version with proper lookup tables

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
        
        # Load lookup tables (CRITICAL FIX)
        try:
            lookup_tables = joblib.load('lookup_tables.joblib')
            st.success("✅ Lookup tables loaded successfully!")
        except FileNotFoundError:
            st.error("❌ Lookup tables not found! Predictions will be inaccurate.")
            st.info("Run: python create_lookup_tables.py")
            lookup_tables = None
            
        return success_model, collection_model, metadata, lookup_tables
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        st.stop()

success_model, collection_model, metadata, lookup_tables = load_models()

# Helper function to get director/actor stats
def get_person_stats(person_name, person_type='director'):
    """Get actual stats from lookup tables or return defaults"""
    if lookup_tables is None:
        # Fallback defaults if lookup tables don't exist
        return {
            'log_director_success': 17.0,
            'log_actor_success': 17.0,
            'director_popularity': 'Experienced',
            'actor_popularity': 'Experienced'
        }
    
    try:
        if person_type == 'director':
            if person_name in lookup_tables['director_lookup']:
                stats = lookup_tables['director_lookup'][person_name]
                return {
                    'log_director_success': stats['log_director_success'],
                    'director_popularity': str(stats['director_popularity'])
                }
            else:
                # New/unknown director
                return {
                    'log_director_success': lookup_tables['director_lookup'].get(
                        'Unknown', {'log_director_success': 15.0}
                    )['log_director_success'],
                    'director_popularity': 'New'
                }
        else:  # actor
            if person_name in lookup_tables['actor_lookup']:
                stats = lookup_tables['actor_lookup'][person_name]
                return {
                    'log_actor_success': stats['log_actor_success'],
                    'actor_popularity': str(stats['actor_popularity'])
                }
            else:
                # New/unknown actor
                return {
                    'log_actor_success': lookup_tables['actor_lookup'].get(
                        'Unknown', {'log_actor_success': 15.0}
                    )['log_actor_success'],
                    'actor_popularity': 'New'
                }
    except Exception as e:
        st.warning(f"Error loading stats: {e}")
        return {
            'log_director_success': 17.0,
            'log_actor_success': 17.0,
            'director_popularity': 'Experienced',
            'actor_popularity': 'Experienced'
        }

# Feature engineering helper
def engineer_features(budget, rating, duration, genre, director, actor, language):
    """Create all required features for prediction"""
    try:
        # Basic calculations
        budget_per_minute = budget / duration
        log_budget = np.log1p(budget)
        rating_budget_interaction = rating * log_budget
        
        # Rating tier
        if rating >= 7.5:
            rating_tier = 'High'
        elif rating >= 6.0:
            rating_tier = 'Medium'
        else:
            rating_tier = 'Low'
        
        # Budget tier
        if budget < 20e6:
            budget_tier = 'Low'
        elif budget < 60e6:
            budget_tier = 'Medium'
        elif budget < 150e6:
            budget_tier = 'High'
        else:
            budget_tier = 'Ultra'
        
        # Get director stats (CRITICAL FIX)
        director_name = director if director != '(Other/New)' else 'Unknown'
        director_stats = get_person_stats(director_name, 'director')
        
        # Get actor stats (CRITICAL FIX)
        actor_name = actor if actor != '(Other/New)' else 'Unknown'
        actor_stats = get_person_stats(actor_name, 'actor')
        
        # Build input dataframe
        input_data = pd.DataFrame({
            'budget': [budget],
            'imdb_score': [rating],
            'duration': [duration],
            'budget_per_minute': [budget_per_minute],
            'log_budget': [log_budget],
            'rating_budget_interaction': [rating_budget_interaction],
            'log_director_success': [director_stats['log_director_success']],
            'log_actor_success': [actor_stats['log_actor_success']],
            'genre_main': [genre],
            'director_name': [director_name],
            'actor_1_name': [actor_name],
            'language': [language],
            'rating_tier': [rating_tier],
            'budget_tier': [budget_tier],
            'director_popularity': [director_stats['director_popularity']],
            'actor_popularity': [actor_stats['actor_popularity']]
        })
        
        return input_data, director_stats, actor_stats
        
    except Exception as e:
        st.error(f"Error in feature engineering: {e}")
        raise

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
    
    if lookup_tables:
        st.success("✅ Lookup Tables Active")
    else:
        st.error("❌ Using Default Values")

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
    
    # Quick test with famous movie
    with st.expander("🧪 Test with Interstellar"):
        if st.button("Run Interstellar Test"):
            st.info("Testing with: Interstellar ($165M budget, 8.6 rating, Christopher Nolan)")
            try:
                input_data, dir_stats, act_stats = engineer_features(
                    budget=165_000_000,
                    rating=8.6,
                    duration=169,
                    genre='Sci-Fi',
                    director='Christopher Nolan',
                    actor='Matthew McConaughey',
                    language='English'
                )
                
                pred = success_model.predict(input_data)[0]
                proba = success_model.predict_proba(input_data)[0]
                
                st.success(f"Prediction: **{pred}**")
                st.write("Probabilities:")
                for cls, p in zip(success_model.classes_, proba):
                    st.progress(p, text=f"{cls}: {p*100:.1f}%")
                
                st.info(f"Director Success Score: {dir_stats['log_director_success']:.2f}")
                st.info(f"Actor Success Score: {act_stats['log_actor_success']:.2f}")
                
                st.write("**Reality Check:**")
                st.write("- Budget: $165M")
                st.write("- Worldwide Gross: $677M")
                st.write("- ROI: 310% ✓")
                
            except Exception as e:
                st.error(f"Test failed: {e}")
    
    st.info("👈 Use the sidebar to navigate to different features")

# ====================
# SINGLE PREDICTION
# ====================
elif page == "🎬 Single Prediction":
    st.title("🎬 Single Movie Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        movie_title = st.text_input("Movie Title (Optional)", "")
        
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
            "Duration": f"{duration} min",
            "Genre": genre,
            "Director": director,
            "Actor": actor
        })
    
    if predict_btn:
        try:
            budget = budget_mil * 1_000_000
            
            # Feature engineering with actual lookup
            input_data, director_stats, actor_stats = engineer_features(
                budget, rating, duration, genre, director, actor, language
            )
            
            # Make predictions
            success_pred = success_model.predict(input_data)[0]
            success_proba = success_model.predict_proba(input_data)[0]
            collection_pred = collection_model.predict(input_data)[0]
            
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            # Main metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Success Level", success_pred)
            with col2:
                st.metric("Collection Range", collection_pred)
            with col3:
                confidence = success_proba.max() * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Probability breakdown
            st.markdown("### Probability Distribution")
            prob_df = pd.DataFrame({
                'Class': success_model.classes_,
                'Probability': success_proba
            }).sort_values('Probability', ascending=False)
            
            fig = px.bar(prob_df, x='Class', y='Probability', 
                        title='Success Probability by Class',
                        color='Probability',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # Success indicator
            if success_pred == 'Blockbuster':
                st.success("🎉 **Highly Recommended!** Strong potential for massive success")
            elif success_pred == 'Hit':
                st.warning("💰 **Good Investment** - Solid returns expected")
            else:
                st.error("📉 **Proceed with Caution** - High risk of underperformance")
            
            # Show feature values used
            with st.expander("🔍 Feature Details (Debug Info)"):
                st.write("**Director Stats:**")
                st.json(director_stats)
                st.write("**Actor Stats:**")
                st.json(actor_stats)
                st.write("**All Features:**")
                st.dataframe(input_data)
            
            # Save to history
            st.session_state.prediction_history.append({
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Title': movie_title if movie_title else "Untitled",
                'Budget': f"${budget_mil}M",
                'Rating': rating,
                'Director': director,
                'Actor': actor,
                'Success': success_pred,
                'Collection': collection_pred,
                'Confidence': f"{confidence:.1f}%"
            })
            
            st.success(f"✅ Prediction saved to history!")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)

# ====================
# BATCH PREDICTIONS
# ====================
elif page == "📊 Batch Predictions":
    st.title("📊 Batch Movie Predictions")
    
    st.info("Upload a CSV file with columns: budget, imdb_score, duration, genre_main, director_name, actor_1_name, language")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} movies")
            st.dataframe(df.head())
            
            if st.button("🚀 Run Predictions", type="primary"):
                with st.spinner("Processing predictions..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        try:
                            input_data, _, _ = engineer_features(
                                budget=row['budget'],
                                rating=row['imdb_score'],
                                duration=row['duration'],
                                genre=row['genre_main'],
                                director=row.get('director_name', 'Unknown'),
                                actor=row.get('actor_1_name', 'Unknown'),
                                language=row.get('language', 'English')
                            )
                            
                            pred = success_model.predict(input_data)[0]
                            proba = success_model.predict_proba(input_data)[0].max()
                            
                            results.append({
                                'Movie': row.get('movie_title', f'Movie {idx+1}'),
                                'Prediction': pred,
                                'Confidence': f"{proba*100:.1f}%"
                            })
                            
                        except Exception as e:
                            results.append({
                                'Movie': row.get('movie_title', f'Movie {idx+1}'),
                                'Prediction': 'ERROR',
                                'Confidence': str(e)
                            })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    results_df = pd.DataFrame(results)
                    st.success("✅ Batch processing complete!")
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "batch_predictions.csv",
                        "text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ====================
# ANALYTICS
# ====================
elif page == "📈 Analytics":
    st.title("📈 Analytics & Insights")
    
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Prediction Distribution")
            success_counts = history_df['Success'].value_counts()
            fig = px.pie(values=success_counts.values, names=success_counts.index,
                        title='Success Categories')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Budget Distribution")
            fig = px.histogram(history_df, x='Budget', title='Budget Ranges')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Prediction History")
        st.dataframe(history_df, use_container_width=True)
        
        # Export option
        csv = history_df.to_csv(index=False)
        st.download_button(
            "📥 Export History",
            csv,
            "prediction_history.csv",
            "text/csv"
        )
    else:
        st.info("No predictions yet. Make some predictions to see analytics!")

# ====================
# COMPARE
# ====================
elif page == "🔍 Compare":
    st.title("🔍 Compare Movie Scenarios")
    st.info("Compare two different movie concepts side-by-side")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎬 Movie A")
        # Add comparison form here
        
    with col2:
        st.markdown("### 🎬 Movie B")
        # Add comparison form here

# ====================
# ABOUT
# ====================
elif page == "ℹ️ About":
    st.title("ℹ️ About")
    
    st.markdown("""
    ## Movie Box Office Success Predictor V2 (Fixed)
    
    ### ✅ What's Fixed:
    - **Proper lookup table integration** - No more hardcoded values!
    - **Error handling** - Better error messages and debugging
    - **Feature visibility** - See actual director/actor scores used
    - **Batch processing** - Fully functional CSV upload
    - **Export functionality** - Download your predictions
    
    ### Features:
    - 🎯 Single movie predictions with real statistics
    - 📊 Batch processing (CSV upload)
    - 📈 Analytics dashboard with history tracking
    - 🔍 Movie comparisons (coming soon)
    
    ### Model Performance:
    - Success Classifier: **73.45%** accuracy
    - Collection Predictor: **79.12%** accuracy
    
    ### Technology Stack:
    - Streamlit
    - Scikit-learn
    - Plotly
    - Gradient Boosting
    
    ### Debug Mode:
    Expand "Feature Details" in predictions to see:
    - Actual director success scores
    - Actual actor success scores
    - All engineered features
    """)