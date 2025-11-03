# movie_predictor_train_v2.py - IMPROVED with realistic thresholds

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
FILE_PATH = "movie_metadata.csv"

try:
    df = pd.read_csv(FILE_PATH)
    print("Data loaded successfully.")
    print(f"Initial shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: {FILE_PATH} not found.")
    exit()

# --- 2. Enhanced Feature Engineering ---
REQUIRED_COLS = ['gross', 'budget', 'director_name', 'actor_1_name', 'genres', 'imdb_score', 'duration', 'language']
df_cleaned = df.dropna(subset=REQUIRED_COLS).copy()
df_cleaned = df_cleaned[(df_cleaned['budget'] > 1000) & (df_cleaned['gross'] > 1000)].copy()

print(f"After cleaning: {df_cleaned.shape}")

df_cleaned['movie_title'] = df_cleaned['movie_title'].str.strip()
df_cleaned['genre_main'] = df_cleaned['genres'].apply(lambda x: x.split('|')[0])

# Feature engineering
df_cleaned['budget_per_minute'] = df_cleaned['budget'] / df_cleaned['duration']
df_cleaned['log_budget'] = np.log1p(df_cleaned['budget'])
df_cleaned['rating_tier'] = pd.cut(df_cleaned['imdb_score'], 
                                    bins=[0, 6.0, 7.5, 10], 
                                    labels=['Low', 'Medium', 'High'])
df_cleaned['budget_tier'] = pd.cut(df_cleaned['budget'],
                                    bins=[0, 20e6, 60e6, 150e6, float('inf')],
                                    labels=['Low', 'Medium', 'High', 'Ultra'])
df_cleaned['rating_budget_interaction'] = df_cleaned['imdb_score'] * df_cleaned['log_budget']

director_counts = df_cleaned['director_name'].value_counts()
df_cleaned['director_movie_count'] = df_cleaned['director_name'].map(director_counts)
df_cleaned['director_popularity'] = pd.cut(df_cleaned['director_movie_count'],
                                           bins=[0, 2, 5, float('inf')],
                                           labels=['New', 'Experienced', 'Veteran'])

actor_counts = df_cleaned['actor_1_name'].value_counts()
df_cleaned['actor_movie_count'] = df_cleaned['actor_1_name'].map(actor_counts)
df_cleaned['actor_popularity'] = pd.cut(df_cleaned['actor_movie_count'],
                                        bins=[0, 3, 8, float('inf')],
                                        labels=['New', 'Experienced', 'Star'])

director_success = df_cleaned.groupby('director_name')['gross'].mean()
df_cleaned['director_avg_gross'] = df_cleaned['director_name'].map(director_success)

actor_success = df_cleaned.groupby('actor_1_name')['gross'].mean()
df_cleaned['actor_avg_gross'] = df_cleaned['actor_1_name'].map(actor_success)

df_cleaned['log_director_success'] = np.log1p(df_cleaned['director_avg_gross'])
df_cleaned['log_actor_success'] = np.log1p(df_cleaned['actor_avg_gross'])

# --- 3. IMPROVED Target Variable Creation ---
print("\n" + "="*70)
print("🎯 USING IMPROVED THRESHOLDS (Industry Standard)")
print("="*70)

def define_success_improved(row):
    """
    Industry standard: movies need ~2x budget to break even (covers marketing)
    Flop: gross < 2x budget (loses money)
    Hit: 2x ≤ gross < 4x budget (profitable)
    Blockbuster: gross ≥ 4x budget (huge success)
    """
    ratio = row['gross'] / row['budget']
    if ratio < 2.0:
        return 'Flop'
    elif ratio < 4.0:
        return 'Hit'
    else:
        return 'Blockbuster'

df_cleaned['success_class'] = df_cleaned.apply(define_success_improved, axis=1)

def define_collection_range(gross):
    if gross < 20_000_000:
        return 'Low (<$20M)'
    elif gross < 100_000_000:
        return 'Moderate ($20M-$100M)'
    elif gross < 300_000_000:
        return 'High ($100M-$300M)'
    else:
        return 'Very High (>$300M)'

df_cleaned['collection_range'] = df_cleaned['gross'].apply(define_collection_range)

# Print class distributions
print("\n=== NEW Class Distributions ===")
print("Success Classes:")
print(df_cleaned['success_class'].value_counts())
print("\nPercentages:")
print(df_cleaned['success_class'].value_counts(normalize=True) * 100)

print("\nCollection Range Classes:")
print(df_cleaned['collection_range'].value_counts())

# Compare with old thresholds
def define_success_old(row):
    profit_margin = row['gross'] / row['budget']
    if profit_margin >= 4.0:
        return 'Blockbuster'
    elif profit_margin >= 1.2:
        return 'Hit'
    else:
        return 'Flop'

df_cleaned['success_class_old'] = df_cleaned.apply(define_success_old, axis=1)

print("\n=== COMPARISON: Old vs New ===")
print("Old thresholds:")
print(df_cleaned['success_class_old'].value_counts(normalize=True) * 100)
print("\nNew thresholds:")
print(df_cleaned['success_class'].value_counts(normalize=True) * 100)

flop_reduction = ((df_cleaned['success_class_old'] == 'Flop').sum() - 
                  (df_cleaned['success_class'] == 'Flop').sum())
print(f"\n✅ Reduced Flops by {flop_reduction} movies ({flop_reduction/len(df_cleaned)*100:.1f}%)")

# --- 4. Enhanced Feature Set ---
NUMERICAL_FEATURES = [
    'budget', 'imdb_score', 'duration', 
    'budget_per_minute', 'log_budget', 
    'rating_budget_interaction',
    'log_director_success', 'log_actor_success'
]

CATEGORICAL_FEATURES = [
    'genre_main', 'director_name', 'actor_1_name', 'language',
    'rating_tier', 'budget_tier', 
    'director_popularity', 'actor_popularity'
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
X = df_cleaned[ALL_FEATURES]
y_success = df_cleaned['success_class']
y_collection_range = df_cleaned['collection_range']

# --- 5. Enhanced Preprocessing Pipeline ---
TOP_N_DIRECTORS = 100
TOP_N_ACTORS = 100
TOP_N_LANGUAGES = 15

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERICAL_FEATURES),
        ('genre', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['genre_main']),
        ('director', OneHotEncoder(handle_unknown='ignore', max_categories=TOP_N_DIRECTORS, sparse_output=False), ['director_name']),
        ('actor', OneHotEncoder(handle_unknown='ignore', max_categories=TOP_N_ACTORS, sparse_output=False), ['actor_1_name']),
        ('lang', OneHotEncoder(handle_unknown='ignore', max_categories=TOP_N_LANGUAGES, sparse_output=False), ['language']),
        ('rating_tier', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['rating_tier']),
        ('budget_tier', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['budget_tier']),
        ('dir_pop', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['director_popularity']),
        ('actor_pop', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['actor_popularity'])
    ],
    remainder='drop'
)

preprocessor.fit(X)
joblib.dump(preprocessor, 'preprocessor.joblib')
print("\nPreprocessor saved to preprocessor.joblib")

# --- 6. Model Training ---
print("\n=== Training Success Classifier ===")
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y_success, test_size=0.2, random_state=42, stratify=y_success
)

models_to_try = {
    'RandomForest': RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )
}

best_model_s = None
best_score_s = 0

for model_name, model in models_to_try.items():
    clf_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    clf_pipeline.fit(X_train_s, y_train_s)
    y_pred_s = clf_pipeline.predict(X_test_s)
    score = accuracy_score(y_test_s, y_pred_s)
    
    print(f"\n{model_name} Accuracy: {score:.4f}")
    print(classification_report(y_test_s, y_pred_s))
    
    if score > best_score_s:
        best_score_s = score
        best_model_s = clf_pipeline

print(f"\n✓ Best Success Classifier: {best_score_s:.4f} accuracy")
joblib.dump(best_model_s, 'success_classifier_model.joblib')
print("Success Classifier Model saved to success_classifier_model.joblib")

# --- 7. Collection Range Model ---
print("\n=== Training Collection Range Classifier ===")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_collection_range, test_size=0.2, random_state=42, stratify=y_collection_range
)

best_model_r = None
best_score_r = 0

for model_name, model in models_to_try.items():
    range_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    range_pipeline.fit(X_train_r, y_train_r)
    y_pred_r = range_pipeline.predict(X_test_r)
    score = accuracy_score(y_test_r, y_pred_r)
    
    print(f"\n{model_name} Accuracy: {score:.4f}")
    print(classification_report(y_test_r, y_pred_r))
    
    if score > best_score_r:
        best_score_r = score
        best_model_r = range_pipeline

print(f"\n✓ Best Collection Range Classifier: {best_score_r:.4f} accuracy")
joblib.dump(best_model_r, 'collection_range_model.joblib')
print("Collection Range Model saved to collection_range_model.joblib")

# --- 8. Save Metadata ---
unique_genres = sorted(df_cleaned['genre_main'].unique().tolist())
unique_directors = sorted(df_cleaned['director_name'].value_counts().head(TOP_N_DIRECTORS).index.tolist())
unique_actors = sorted(df_cleaned['actor_1_name'].value_counts().head(TOP_N_ACTORS).index.tolist())
unique_languages = sorted(df_cleaned['language'].value_counts().head(TOP_N_LANGUAGES).index.tolist())

metadata = {
    'genres': unique_genres,
    'directors': unique_directors,
    'actors': unique_actors,
    'languages': unique_languages
}
joblib.dump(metadata, 'metadata.joblib')
print("\nMetadata saved to metadata.joblib")

print("\n" + "="*50)
print("✅ TRAINING COMPLETE WITH IMPROVED THRESHOLDS")
print("="*50)
print(f"Success Classifier Accuracy: {best_score_s:.2%}")
print(f"Collection Range Accuracy: {best_score_r:.2%}")
print("\nNew thresholds:")
print("  Flop: gross < 2x budget")
print("  Hit: 2x ≤ gross < 4x budget")
print("  Blockbuster: gross ≥ 4x budget")
print("="*50)