# diagnose_prediction.py
# Let's see what the model is actually seeing

import pandas as pd
import numpy as np
import joblib

# Load everything
success_model = joblib.load('success_classifier_model.joblib')
lookup_tables = joblib.load('lookup_tables.joblib')

# Interstellar details
budget = 165_000_000
rating = 8.6
duration = 169
genre = 'Sci-Fi'
director = 'Christopher Nolan'
actor = 'Matthew McConaughey'
language = 'English'

# Feature engineering
budget_per_minute = budget / duration
log_budget = np.log1p(budget)
rating_budget_interaction = rating * log_budget
rating_tier = 'High'
budget_tier = 'Ultra'

# Get actual stats
if director in lookup_tables['director_lookup']:
    director_stats = lookup_tables['director_lookup'][director]
    log_director_success = director_stats['log_director_success']
    director_popularity = director_stats['director_popularity']
    print(f"✅ Found {director}:")
    print(f"   Success Score: {log_director_success:.2f}")
    print(f"   Popularity: {director_popularity}")
else:
    print(f"❌ {director} not found in lookup")
    log_director_success = 17.0
    director_popularity = 'Experienced'

if actor in lookup_tables['actor_lookup']:
    actor_stats = lookup_tables['actor_lookup'][actor]
    log_actor_success = actor_stats['log_actor_success']
    actor_popularity = actor_stats['actor_popularity']
    print(f"✅ Found {actor}:")
    print(f"   Success Score: {log_actor_success:.2f}")
    print(f"   Popularity: {actor_popularity}")
else:
    print(f"❌ {actor} not found in lookup")
    log_actor_success = 17.0
    actor_popularity = 'Experienced'

# Create input
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
    'director_name': [director],
    'actor_1_name': [actor],
    'language': [language],
    'rating_tier': [rating_tier],
    'budget_tier': [budget_tier],
    'director_popularity': [str(director_popularity)],
    'actor_popularity': [str(actor_popularity)]
})

print("\n" + "="*50)
print("INPUT DATA:")
print("="*50)
print(input_data.T)

# Predict
success_pred = success_model.predict(input_data)[0]
success_proba = success_model.predict_proba(input_data)[0]

print("\n" + "="*50)
print("PREDICTION:")
print("="*50)
print(f"Success Level: {success_pred}")
print(f"Probabilities:")
for cls, prob in zip(success_model.classes_, success_proba):
    print(f"  {cls}: {prob*100:.1f}%")

# Now let's try with AVERAGE director/actor to compare
print("\n" + "="*50)
print("COMPARISON WITH AVERAGE DIRECTOR/ACTOR:")
print("="*50)

input_data_avg = input_data.copy()
input_data_avg['log_director_success'] = [17.0]
input_data_avg['log_actor_success'] = [17.0]
input_data_avg['director_popularity'] = ['Experienced']
input_data_avg['actor_popularity'] = ['Experienced']

success_pred_avg = success_model.predict(input_data_avg)[0]
success_proba_avg = success_model.predict_proba(input_data_avg)[0]

print(f"With average values: {success_pred_avg}")
print(f"Probabilities:")
for cls, prob in zip(success_model.classes_, success_proba_avg):
    print(f"  {cls}: {prob*100:.1f}%")

print("\n" + "="*50)
print("ANALYSIS:")
print("="*50)

# Check what the training data actually used
print("\nChecking if model was trained with real success rates...")
print("(If predictions are the same, model wasn't trained with real data)")

if success_pred == success_pred_avg:
    print("\n⚠️ WARNING: Model gives same prediction with/without real director data!")
    print("This means the model was trained with AVERAGE values, not real ones.")
    print("\n💡 SOLUTION: You need to retrain the model!")
    print("The training script should calculate real director/actor success rates")
    print("during training, not use fixed 17.0 for everyone.")
else:
    print("\n✅ Model responds to director/actor quality!")
    print("Something else is wrong...")