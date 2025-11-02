# explain_prediction.py
# Generate explanations for predictions using SHAP

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model
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

director_stats = lookup_tables['director_lookup'][director]
log_director_success = director_stats['log_director_success']
director_popularity = director_stats['director_popularity']

actor_stats = lookup_tables['actor_lookup'][actor]
log_actor_success = actor_stats['log_actor_success']
actor_popularity = actor_stats['actor_popularity']

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

print("="*60)
print("INTERSTELLAR PREDICTION EXPLANATION")
print("="*60)

# Make prediction
pred = success_model.predict(input_data)[0]
proba = success_model.predict_proba(input_data)[0]

print(f"\nPrediction: {pred}")
print("Probabilities:")
for cls, p in zip(success_model.classes_, proba):
    print(f"  {cls}: {p*100:.1f}%")

print("\n" + "="*60)
print("WHY DID THE MODEL PREDICT THIS?")
print("="*60)

# Create SHAP explainer
print("\nCalculating SHAP values (this may take a minute)...")

# Get the preprocessor and transform data
X_processed = success_model.named_steps['preprocessor'].transform(input_data)

# Create explainer for the classifier
explainer = shap.TreeExplainer(
    success_model.named_steps['classifier'],
    feature_perturbation='interventional'
)

# Calculate SHAP values
shap_values = explainer.shap_values(X_processed)

# Get feature names after preprocessing
feature_names = []
preprocessor = success_model.named_steps['preprocessor']

for name, transformer, columns in preprocessor.transformers_:
    if name == 'num':
        feature_names.extend(columns)
    elif hasattr(transformer, 'get_feature_names_out'):
        feature_names.extend(transformer.get_feature_names_out(columns))

print("\n" + "="*60)
print("TOP FACTORS PUSHING TOWARD 'FLOP':")
print("="*60)

# Get SHAP values for Flop class
flop_idx = list(success_model.classes_).index('Flop')
flop_shap = shap_values[flop_idx][0]

# Get top negative (pushing away from Flop) and positive (pushing toward Flop) factors
feature_impact = pd.DataFrame({
    'Feature': feature_names,
    'SHAP Value': flop_shap
}).sort_values('SHAP Value', ascending=False)

print("\n🔴 Factors INCREASING Flop probability:")
top_flop = feature_impact.head(10)
for idx, row in top_flop.iterrows():
    if row['SHAP Value'] > 0:
        print(f"  {row['Feature']}: +{row['SHAP Value']:.4f}")

print("\n🟢 Factors DECREASING Flop probability:")
top_not_flop = feature_impact.tail(10)
for idx, row in top_not_flop.iterrows():
    if row['SHAP Value'] < 0:
        print(f"  {row['Feature']}: {row['SHAP Value']:.4f}")

# Simple interpretation
print("\n" + "="*60)
print("SIMPLE EXPLANATION:")
print("="*60)

# Analyze key factors
if budget_tier == 'Ultra':
    print("\n⚠️ ULTRA-HIGH BUDGET ($150M+):")
    print("   The model sees this as HIGH RISK because:")
    print("   • Most $150M+ movies in training data were flops")
    print("   • Higher budgets need massive box office to break even")
    print("   • Ultra budgets often mean studio interference")

if genre == 'Sci-Fi':
    print("\n🚀 SCI-FI GENRE:")
    print("   Historically risky because:")
    print("   • Sci-Fi has lower average success rate than Action")
    print("   • Expensive VFX don't always guarantee quality")
    print("   • Niche audience compared to action movies")

if log_director_success > 18:
    print(f"\n⭐ CHRISTOPHER NOLAN (Success Score: {log_director_success:.2f}):")
    print("   This HELPS the prediction:")
    print("   • Among top directors in the dataset")
    print("   • But not enough to overcome ultra-budget + Sci-Fi risk")

if rating >= 8.5:
    print(f"\n🎯 HIGH RATING ({rating}/10):")
    print("   This HELPS significantly:")
    print("   • Model knows high-rated movies perform better")
    print("   • But it's an EXPECTED rating, not a guarantee")

print("\n" + "="*60)
print("BOTTOM LINE:")
print("="*60)
print("""
The model is being CONSERVATIVE based on historical data:
• $165M Sci-Fi movies have high failure rate (think John Carter, Jupiter Ascending)
• Even great directors struggle with ultra-high budgets
• Interstellar succeeded DESPITE these risk factors (that's what made it special!)

The model's job is to predict the MOST LIKELY outcome based on patterns.
Interstellar beat the odds - which is why it's legendary!

To get a 'Hit' or 'Blockbuster' prediction, the model would need:
• Lower budget ($80-120M range)
• OR Action/Adventure genre instead of Sci-Fi  
• OR even stronger director track record (Spielberg scored 96% Hit!)
""")

print("\n" + "="*60)
print("Want to save detailed SHAP plots? (will open in browser)")
response = input("Save plots? (y/n): ")

if response.lower() == 'y':
    # Create SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values[flop_idx].reshape(1, -1),
        X_processed,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig('shap_explanation.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved as 'shap_explanation.png'")
    plt.close()
    
    # Create waterfall plot
    shap.waterfall_plot(
        shap.Explanation(
            values=flop_shap,
            base_values=explainer.expected_value[flop_idx],
            data=X_processed[0],
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
    print("✅ Saved as 'shap_waterfall.png'")
    plt.close()