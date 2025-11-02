import pandas as pd
import numpy as np
import joblib

success_model = joblib.load('success_classifier_model.joblib')
lookup_tables = joblib.load('lookup_tables.joblib')

def test_movie(budget_mil, rating, duration, genre, director, actor):
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
    log_director_success = director_stats.get('log_director_success', 17.0)
    director_popularity = str(director_stats.get('director_popularity', 'Experienced'))
    
    actor_stats = lookup_tables['actor_lookup'].get(actor, {})
    log_actor_success = actor_stats.get('log_actor_success', 17.0)
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
    print(f'\n{director} - \M - {rating}/10 - {genre}')
    print(f'  → {pred}', end='')
    for cls, p in zip(success_model.classes_, proba):
        print(f' | {cls}: {p*100:.0f}%', end='')
    print()

# Test various scenarios
print('='*60)
print('TESTING DIFFERENT SCENARIOS')
print('='*60)

# Lower budget Nolan
test_movie(50, 8.5, 120, 'Action', 'Christopher Nolan', 'Christian Bale')

# Interstellar as-is
test_movie(165, 8.6, 169, 'Sci-Fi', 'Christopher Nolan', 'Matthew McConaughey')

# Lower Interstellar budget (what if it was cheaper?)
test_movie(100, 8.6, 169, 'Sci-Fi', 'Christopher Nolan', 'Matthew McConaughey')

# Different genre
test_movie(165, 8.6, 169, 'Action', 'Christopher Nolan', 'Matthew McConaughey')

# Spielberg comparison
test_movie(165, 8.0, 169, 'Sci-Fi', 'Steven Spielberg', 'Tom Hanks')

# Unknown director with same specs
test_movie(165, 8.6, 169, 'Sci-Fi', 'Unknown', 'Unknown')
