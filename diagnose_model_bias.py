# diagnose_model_bias.py
# Check if the model is biased toward predicting Flop

import pandas as pd
import numpy as np
import joblib

# Load everything
success_model = joblib.load('success_classifier_model.joblib')
collection_model = joblib.load('collection_range_model.joblib')
metadata = joblib.load('metadata.joblib')
lookup_tables = joblib.load('lookup_tables.joblib')

# Load training data
df = pd.read_csv("movie_metadata.csv")
REQUIRED_COLS = ['gross', 'budget', 'director_name', 'actor_1_name', 'genres', 'imdb_score', 'duration', 'language']
df_cleaned = df.dropna(subset=REQUIRED_COLS).copy()
df_cleaned = df_cleaned[(df_cleaned['budget'] > 1000) & (df_cleaned['gross'] > 1000)].copy()
df_cleaned['genre_main'] = df_cleaned['genres'].apply(lambda x: x.split('|')[0])

# Create success labels
df_cleaned['profit_margin'] = df_cleaned['gross'] / df_cleaned['budget']
df_cleaned['success'] = df_cleaned['profit_margin'].apply(
    lambda x: 'Blockbuster' if x >= 4.0 else ('Hit' if x >= 1.2 else 'Flop')
)

print("="*70)
print("MODEL BIAS DIAGNOSIS")
print("="*70)

# Check training data distribution
print("\n1️⃣ TRAINING DATA DISTRIBUTION:")
print(df_cleaned['success'].value_counts())
print(f"\nPercentages:")
print(df_cleaned['success'].value_counts(normalize=True) * 100)

flop_pct = (df_cleaned['success'] == 'Flop').sum() / len(df_cleaned) * 100
print(f"\n⚠️ {flop_pct:.1f}% of training data are FLOPS!")
print("This creates a strong bias toward predicting Flop")

# Test realistic scenarios
def test_scenario(name, budget_mil, rating, duration, genre, director, actor):
    budget = budget_mil * 1_000_000
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
    
    director_stats = lookup_tables['director_lookup'].get(director, {})
    log_director_success = director_stats.get('log_director_success', lookup_tables['default_director_success'])
    director_popularity = str(director_stats.get('director_popularity', 'Experienced'))
    
    actor_stats = lookup_tables['actor_lookup'].get(actor, {})
    log_actor_success = actor_stats.get('log_actor_success', lookup_tables['default_actor_success'])
    actor_popularity = str(actor_stats.get('actor_popularity', 'Experienced'))
    
    input_data = pd.DataFrame({
        'budget': [budget], 'imdb_score': [rating], 'duration': [duration],
        'budget_per_minute': [budget_per_minute], 'log_budget': [log_budget],
        'rating_budget_interaction': [rating_budget_interaction],
        'log_director_success': [log_director_success], 'log_actor_success': [log_actor_success],
        'genre_main': [genre], 'director_name': [director], 'actor_1_name': [actor],
        'language': ['English'], 'rating_tier': [rating_tier], 'budget_tier': [budget_tier],
        'director_popularity': [director_popularity], 'actor_popularity': [actor_popularity]
    })
    
    pred = success_model.predict(input_data)[0]
    proba = success_model.predict_proba(input_data)[0]
    
    return pred, proba

print("\n" + "="*70)
print("2️⃣ TESTING VARIOUS SCENARIOS:")
print("="*70)

scenarios = [
    ("Low Budget Indie", 5, 7.5, 95, 'Drama', 'Unknown', 'Unknown'),
    ("Mid Budget Action", 50, 7.5, 110, 'Action', 'Unknown', 'Unknown'),
    ("High Budget Blockbuster", 120, 7.5, 130, 'Action', 'Unknown', 'Unknown'),
    ("Spielberg Sci-Fi", 150, 8.0, 140, 'Sci-Fi', 'Steven Spielberg', 'Tom Hanks'),
    ("Nolan Action", 100, 8.5, 150, 'Action', 'Christopher Nolan', 'Christian Bale'),
    ("Cameron Sci-Fi", 200, 8.0, 160, 'Sci-Fi', 'James Cameron', 'Sam Worthington'),
    ("Perfect Storm", 80, 9.0, 120, 'Action', 'Christopher Nolan', 'Leonardo DiCaprio'),
]

results = []
for name, *params in scenarios:
    pred, proba = test_scenario(name, *params)
    results.append((name, pred, proba.max() * 100))
    
    print(f"\n{name} (${params[0]}M, {params[1]}/10, {params[3]}):")
    print(f"  → {pred} ({proba.max()*100:.0f}% confidence)")
    for cls, p in zip(success_model.classes_, proba):
        if p > 0.01:
            print(f"     {cls}: {p*100:.0f}%")

# Count predictions
pred_counts = pd.DataFrame(results, columns=['Scenario', 'Prediction', 'Confidence'])
print("\n" + "="*70)
print("3️⃣ PREDICTION SUMMARY:")
print("="*70)
print(pred_counts['Prediction'].value_counts())

flop_predictions = (pred_counts['Prediction'] == 'Flop').sum()
print(f"\n⚠️ {flop_predictions}/{len(scenarios)} predictions are FLOP!")

# Check the actual profit thresholds
print("\n" + "="*70)
print("4️⃣ PROFIT THRESHOLD ANALYSIS:")
print("="*70)

print("\nCurrent thresholds:")
print("  Flop: profit_margin < 1.2x (less than 20% profit)")
print("  Hit: 1.2x ≤ profit_margin < 4.0x")
print("  Blockbuster: profit_margin ≥ 4.0x (4x return)")

print("\n💡 PROBLEM IDENTIFIED:")
print("  The threshold might be too strict!")
print("  profit_margin = gross / budget")
print("  So a $100M movie needs $120M+ just to be a 'Hit'")
print("  But marketing costs mean it needs $200M+ to truly profit")

# Suggest fix
print("\n" + "="*70)
print("5️⃣ SUGGESTED FIXES:")
print("="*70)

print("""
OPTION 1: Adjust profit thresholds (in training script):
  Current:
    Flop: profit_margin < 1.2
    Hit: 1.2 ≤ profit_margin < 4.0
    Blockbuster: profit_margin ≥ 4.0
  
  Suggested (more realistic):
    Flop: profit_margin < 0.8 (lose money)
    Hit: 0.8 ≤ profit_margin < 2.5
    Blockbuster: profit_margin ≥ 2.5

OPTION 2: Use ROI instead of profit margin:
    ROI = (gross - budget) / budget
    Flop: ROI < 0 (lose money)
    Hit: 0 ≤ ROI < 1.0 (break even to 2x)
    Blockbuster: ROI ≥ 1.0 (double or more)

OPTION 3: Account for marketing (rule of thumb: 2x budget to break even):
    Flop: gross < 2 × budget
    Hit: 2 × budget ≤ gross < 4 × budget
    Blockbuster: gross ≥ 4 × budget

RECOMMENDATION: Use Option 3 - it's industry standard!
""")

print("Would you like me to create a retrain script with better thresholds?")