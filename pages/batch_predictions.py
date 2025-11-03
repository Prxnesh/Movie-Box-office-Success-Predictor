# pages/batch_predictions.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import io

def show(success_model, collection_model, metadata):
    st.title("📊 Batch Movie Predictions")
    st.markdown("Upload a CSV file with multiple movie concepts to predict them all at once")
    
    # Instructions
    with st.expander("📋 CSV Format Instructions", expanded=True):
        st.markdown("""
        Your CSV file should contain the following columns:
        - **movie_title** (optional): Movie name
        - **budget**: Budget in USD (e.g., 50000000 for $50M)
        - **imdb_score**: Expected rating (1.0 to 10.0)
        - **duration**: Runtime in minutes
        - **genre_main**: Main genre
        - **director_name**: Director name (or 'Unknown')
        - **actor_1_name**: Main actor name (or 'Unknown')
        - **language**: Original language
        
        [Download sample CSV template](#)
        """)
        
        # Sample CSV template
        sample_data = pd.DataFrame({
            'movie_title': ['Action Movie', 'Drama Film'],
            'budget': [50000000, 30000000],
            'imdb_score': [7.5, 8.0],
            'duration': [120, 110],
            'genre_main': ['Action', 'Drama'],
            'director_name': ['Unknown', 'Unknown'],
            'actor_1_name': ['Unknown', 'Unknown'],
            'language': ['English', 'English']
        })
        
        csv_buffer = io.StringIO()
        sample_data.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv_buffer.getvalue(),
            file_name="sample_movies.csv",
            mime="text/csv"
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file with movie data"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} movies.")
            
            # Show preview
            with st.expander("👀 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Validate columns
            required_cols = ['budget', 'imdb_score', 'duration', 'genre_main', 
                           'director_name', 'actor_1_name', 'language']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                return
            
            # Process button
            if st.button("🚀 Run Batch Predictions", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(df)} movies..."):
                    # Feature engineering for each row
                    df['budget_per_minute'] = df['budget'] / df['duration']
                    df['log_budget'] = np.log1p(df['budget'])
                    df['rating_budget_interaction'] = df['imdb_score'] * df['log_budget']
                    
                    # Categorizations
                    df['rating_tier'] = pd.cut(df['imdb_score'], 
                                               bins=[0, 6.0, 7.5, 10],
                                               labels=['Low', 'Medium', 'High'])
                    df['budget_tier'] = pd.cut(df['budget'],
                                               bins=[0, 20e6, 60e6, 150e6, float('inf')],
                                               labels=['Low', 'Medium', 'High', 'Ultra'])
                    
                    # Add default values
                    df['log_director_success'] = 17.0
                    df['log_actor_success'] = 17.0
                    df['director_popularity'] = 'Experienced'
                    df['actor_popularity'] = 'Experienced'
                    
                    # Prepare features
                    feature_cols = [
                        'budget', 'imdb_score', 'duration', 'budget_per_minute',
                        'log_budget', 'rating_budget_interaction', 'log_director_success',
                        'log_actor_success', 'genre_main', 'director_name', 'actor_1_name',
                        'language', 'rating_tier', 'budget_tier', 'director_popularity',
                        'actor_popularity'
                    ]
                    
                    X = df[feature_cols]
                    
                    # Make predictions
                    df['predicted_success'] = success_model.predict(X)
                    df['success_confidence'] = success_model.predict_proba(X).max(axis=1) * 100
                    
                    df['predicted_collection'] = collection_model.predict(X)
                    df['collection_confidence'] = collection_model.predict_proba(X).max(axis=1) * 100
                
                st.success("✅ Predictions completed!")
                
                # Results summary
                st.markdown("---")
                st.markdown("## 📊 Results Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    blockbusters = (df['predicted_success'] == 'Blockbuster').sum()
                    st.metric("🎉 Blockbusters", blockbusters, 
                             delta=f"{blockbusters/len(df)*100:.1f}%")
                
                with col2:
                    hits = (df['predicted_success'] == 'Hit').sum()
                    st.metric("💰 Hits", hits,
                             delta=f"{hits/len(df)*100:.1f}%")
                
                with col3:
                    flops = (df['predicted_success'] == 'Flop').sum()
                    st.metric("📉 Flops", flops,
                             delta=f"{flops/len(df)*100:.1f}%")
                
                with col4:
                    avg_confidence = df['success_confidence'].mean()
                    st.metric("⭐ Avg Confidence", f"{avg_confidence:.1f}%")
                
                # Visualizations
                st.markdown("### 📈 Prediction Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(df, names='predicted_success',
                                title='Success Categories',
                                color_discrete_map={
                                    'Blockbuster': '#38ef7d',
                                    'Hit': '#f5576c',
                                    'Flop': '#667eea'
                                })
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(df, x='predicted_collection',
                                      title='Collection Range Distribution',
                                      color_discrete_sequence=['#4facfe'])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.markdown("### 📋 Detailed Results")
                
                # Prepare display dataframe
                display_cols = ['movie_title'] if 'movie_title' in df.columns else []
                display_cols += ['budget', 'imdb_score', 'genre_main', 'predicted_success',
                                'success_confidence', 'predicted_collection', 'collection_confidence']
                
                results_df = df[display_cols].copy()
                results_df['budget'] = results_df['budget'].apply(lambda x: f"${x/1e6:.1f}M")
                results_df['success_confidence'] = results_df['success_confidence'].apply(lambda x: f"{x:.1f}%")
                results_df['collection_confidence'] = results_df['collection_confidence'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Export results
                st.markdown("---")
                st.markdown("### 💾 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Full results CSV
                    csv_output = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results (CSV)",
                        data=csv_output,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Summary report
                    summary = f"""
                    BATCH PREDICTION SUMMARY
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    Total Movies Analyzed: {len(df)}
                    
                    Success Predictions:
                    - Blockbusters: {blockbusters} ({blockbusters/len(df)*100:.1f}%)
                    - Hits: {hits} ({hits/len(df)*100:.1f}%)
                    - Flops: {flops} ({flops/len(df)*100:.1f}%)
                    
                    Average Confidence: {avg_confidence:.2f}%
                    """
                    
                    st.download_button(
                        label="📥 Download Summary Report",
                        data=summary,
                        file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please check your CSV format and try again.")