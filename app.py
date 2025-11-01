# app.py

import streamlit as st
import pandas as pd
import joblib
from streamlit_searchbox import st_searchbox

# --- 1. Load Pre-trained Assets ---
try:
    success_model = joblib.load('success_classifier_model.joblib')
    collection_model = joblib.load('collection_range_model.joblib')
    metadata = joblib.load('metadata.joblib')
    print("Models and metadata loaded.")
except FileNotFoundError:
    st.error("Model files not found. Please run 'movie_predictor_train.py' first to generate the necessary files.")
    st.stop()

# --- Helper Functions for Searchbox ---
def search_genres(searchterm: str):
    """Search function for genres"""
    if not searchterm:
        return metadata['genres'][:10]  # Return first 10 if no search
    return [genre for genre in metadata['genres'] if searchterm.lower() in genre.lower()]

def search_directors(searchterm: str):
    """Search function for directors"""
    options = ['(Other/New Director)'] + metadata['directors']
    if not searchterm:
        return options[:10]
    return [director for director in options if searchterm.lower() in director.lower()]

def search_actors(searchterm: str):
    """Search function for actors"""
    options = ['(Other/New Star)'] + metadata['actors']
    if not searchterm:
        return options[:10]
    return [actor for actor in options if searchterm.lower() in actor.lower()]

def search_languages(searchterm: str):
    """Search function for languages"""
    if not searchterm:
        return metadata['languages'][:10]
    return [language for language in metadata['languages'] if searchterm.lower() in language.lower()]

# --- 2. Streamlit UI Setup ---
st.set_page_config(
    page_title="Movie Box Office Success Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 Movie Box Office Success Predictor")
st.markdown("""
Welcome, Producer! Enter the details of your movie concept below to predict its **success level** and **estimated collection range**.
This tool uses Machine Learning models trained on historical box office data.
""")

# --- 3. Input Form ---
with st.sidebar:
    st.header("Movie Details Input")
    st.markdown("---")

    # Numerical Inputs
    budget_mil = st.number_input("Budget (in Millions USD)", min_value=0.1, max_value=400.0, value=50.0, step=1.0)
    budget = budget_mil * 1_000_000 
    
    rating = st.slider("IMDb/Average Rating (Expected)", min_value=1.0, max_value=10.0, value=7.0, step=0.1)
    duration = st.slider("Duration (Minutes)", min_value=60, max_value=240, value=100, step=5)

    # Categorical Inputs with Autocomplete/Search
    st.markdown("### 🔍 Searchable Fields")
    
    genre = st_searchbox(
        search_genres,
        key="genre_search",
        placeholder="Type to search genres...",
        label="Main Genre"
    )
    
    director = st_searchbox(
        search_directors,
        key="director_search",
        placeholder="Type to search directors...",
        label="Director Name"
    )
    
    actor = st_searchbox(
        search_actors,
        key="actor_search",
        placeholder="Type to search actors...",
        label="Main Star (Actor 1)"
    )
    
    language = st_searchbox(
        search_languages,
        key="language_search",
        placeholder="Type to search languages...",
        label="Original Language"
    )
    
    st.markdown("---")
    predict_button = st.button("🚀 Predict Success")

# --- 4. Prediction Logic and Output ---
if predict_button:
    # Validate inputs
    if not all([genre, director, actor, language]):
        st.error("⚠️ Please fill in all fields before predicting!")
        st.stop()
    
    st.subheader("Prediction Results")
    
    # Create the DataFrame for prediction with engineered features
    import numpy as np
    
    # Calculate derived features
    budget_per_minute = budget / duration
    log_budget = np.log1p(budget)
    rating_budget_interaction = rating * log_budget
    
    # Categorize rating and budget
    if rating < 6.0:
        rating_tier = 'Low'
    elif rating < 7.5:
        rating_tier = 'Medium'
    else:
        rating_tier = 'High'
    
    if budget < 20e6:
        budget_tier = 'Low'
    elif budget < 60e6:
        budget_tier = 'Medium'
    elif budget < 150e6:
        budget_tier = 'High'
    else:
        budget_tier = 'Ultra'
    
    # Default values for popularity features (will be handled by model)
    log_director_success = 17.0  # Average log success
    log_actor_success = 17.0
    director_popularity = 'Experienced'
    actor_popularity = 'Experienced'
    
    input_data = pd.DataFrame({
        'budget': [budget],
        'imdb_score': [rating],
        'duration': [duration],
        'budget_per_minute': [budget_per_minute],
        'log_budget': [log_budget],
        'rating_budget_interaction': [rating_budget_interaction],
        'log_director_success': [log_director_success],
        'log_actor_success': [log_actor_success],
        'genre_main': [genre],
        'director_name': [director if director != '(Other/New Director)' else 'Unknown'],
        'actor_1_name': [actor if actor != '(Other/New Star)' else 'Unknown'],
        'language': [language],
        'rating_tier': [rating_tier],
        'budget_tier': [budget_tier],
        'director_popularity': [director_popularity],
        'actor_popularity': [actor_popularity]
    })
    
    with st.spinner("Calculating predictions..."):
        # Predict Success Class
        success_pred = success_model.predict(input_data)[0]
        success_proba = success_model.predict_proba(input_data)
        
        # Get probability for the predicted class
        class_index_s = success_model.classes_.tolist().index(success_pred)
        confidence_s = success_proba[0][class_index_s] * 100
        
        # Predict Collection Range
        collection_range_pred = collection_model.predict(input_data)[0]
        collection_proba = collection_model.predict_proba(input_data)
        
        # Get probability for the predicted class
        class_index_r = collection_model.classes_.tolist().index(collection_range_pred)
        confidence_r = collection_proba[0][class_index_r] * 100

    # Display Results
    col1, col2 = st.columns(2)

    # --- Success Prediction Output ---
    with col1:
        st.metric(
            label="Predicted Success Level", 
            value=success_pred, 
            delta=f"{confidence_s:.1f}% Confidence"
        )
        
        st.markdown("**Investment Recommendation:**")
        if success_pred == 'Blockbuster':
            st.success("🎉 **HIGHLY RECOMMENDED!** Strong potential for a Blockbuster return.")
        elif success_pred == 'Hit':
            st.warning("💰 **RECOMMENDED.** A solid Hit is predicted, offering a good profit.")
        else:
            st.error("📉 **CAUTION.** The model predicts a Flop. Re-evaluate budget or creative features.")

    # --- Collection Range Output ---
    with col2:
        st.metric(
            label="Estimated Collection Range (Gross)", 
            value=collection_range_pred, 
            delta=f"{confidence_r:.1f}% Confidence"
        )
        
        st.markdown("**Model Confidence:**")
        st.info("The model estimates the gross collection will fall within the predicted range.")

    st.markdown("---")
    
    # --- Detailed Probabilities ---
    st.subheader("Detailed Probability Breakdown")
    
    col3, col4 = st.columns(2)
    
    # Success Probabilities
    success_df = pd.DataFrame({
        'Success Level': success_model.classes_,
        'Probability (%)': success_proba[0] * 100
    }).sort_values('Probability (%)', ascending=False)
    
    with col3:
        st.markdown("##### Success Level Probabilities")
        st.dataframe(success_df, hide_index=True, use_container_width=True)
        
    # Collection Range Probabilities
    collection_df = pd.DataFrame({
        'Collection Range': collection_model.classes_,
        'Probability (%)': collection_proba[0] * 100
    }).sort_values('Probability (%)', ascending=False)
    
    with col4:
        st.markdown("##### Collection Range Probabilities")
        st.dataframe(collection_df, hide_index=True, use_container_width=True)

# --- 5. Run Instructions ---
else:
    st.info("Enter the movie details in the sidebar (budget, rating, director, etc.) and click 'Predict Success' to see the box office forecast.")