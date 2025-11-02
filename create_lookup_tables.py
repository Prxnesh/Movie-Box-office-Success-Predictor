# create_lookup_tables.py
# Run this ONCE after training to create lookup tables

import pandas as pd
import numpy as np
import joblib

# Load the original data
df = pd.read_csv("movie_metadata.csv")

# Clean the data (same as training)
REQUIRED_COLS = ['gross', 'budget', 'director_name', 'actor_1_name', 'genres', 'imdb_score', 'duration', 'language']
df_cleaned = df.dropna(subset=REQUIRED_COLS).copy()
df_cleaned = df_cleaned[(df_cleaned['budget'] > 1000) & (df_cleaned['gross'] > 1000)].copy()

# Calculate director success rates
director_stats = df_cleaned.groupby('director_name').agg({
    'gross': ['mean', 'count']
}).reset_index()
director_stats.columns = ['director_name', 'avg_gross', 'movie_count']
director_stats['log_director_success'] = np.log1p(director_stats['avg_gross'])

# Determine director popularity
director_stats['director_popularity'] = pd.cut(
    director_stats['movie_count'],
    bins=[0, 2, 5, float('inf')],
    labels=['New', 'Experienced', 'Veteran']
)

# Calculate actor success rates
actor_stats = df_cleaned.groupby('actor_1_name').agg({
    'gross': ['mean', 'count']
}).reset_index()
actor_stats.columns = ['actor_1_name', 'avg_gross', 'movie_count']
actor_stats['log_actor_success'] = np.log1p(actor_stats['avg_gross'])

# Determine actor popularity
actor_stats['actor_popularity'] = pd.cut(
    actor_stats['movie_count'],
    bins=[0, 3, 8, float('inf')],
    labels=['New', 'Experienced', 'Star']
)

# Create lookup dictionaries
director_lookup = director_stats.set_index('director_name')[['log_director_success', 'director_popularity']].to_dict('index')
actor_lookup = actor_stats.set_index('actor_1_name')[['log_actor_success', 'actor_popularity']].to_dict('index')

# Calculate defaults (medians)
default_director_success = director_stats['log_director_success'].median()
default_actor_success = actor_stats['log_actor_success'].median()

# Save lookup tables
lookup_data = {
    'director_lookup': director_lookup,
    'actor_lookup': actor_lookup,
    'default_director_success': default_director_success,
    'default_actor_success': default_actor_success
}

joblib.dump(lookup_data, 'lookup_tables.joblib')

print("✅ Lookup tables created successfully!")
print(f"\nSample Directors:")
for director in ['Christopher Nolan', 'Steven Spielberg', 'James Cameron']:
    if director in director_lookup:
        stats = director_lookup[director]
        print(f"  {director}:")
        print(f"    Success Score: {stats['log_director_success']:.2f}")
        print(f"    Popularity: {stats['director_popularity']}")

print(f"\nSample Actors:")
for actor in ['Matthew McConaughey', 'Leonardo DiCaprio', 'Tom Cruise']:
    if actor in actor_lookup:
        stats = actor_lookup[actor]
        print(f"  {actor}:")
        print(f"    Success Score: {stats['log_actor_success']:.2f}")
        print(f"    Popularity: {stats['actor_popularity']}")

print(f"\nDefault values:")
print(f"  Director: {default_director_success:.2f}")
print(f"  Actor: {default_actor_success:.2f}")