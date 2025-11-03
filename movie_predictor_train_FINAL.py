# movie_predictor_train_FINAL.py - ROI-Based Classification
# Instead of arbitrary thresholds, predict actual ROI ranges

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

FILE_PATH = "movie_metadata.csv"

df = pd.read_csv(FILE_PATH)
print(f"Initial shape: {df.shape}")

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

print("\n" + "="*70)
print("💰 NEW APPROACH: ROI-BASED CLASSIFICATION")
print("="*70)

# Calculate ROI (Return on Investment)
df_cleaned['roi'] = (df_cleaned['gross'] - df_cleaned['budget']) / df_cleaned['budget']

def define_roi_class(roi):
    """
    Based on actual ROI, not arbitrary ratios
    Loss: ROI < -0.5 (lost more than half the budget)
    Break Even: -0.5 <= ROI < 0.5 (roughly broke even)  
    Profitable: 0.5 <= ROI < 2.0 (made profit)
    Very Profitable: 2.0 <= ROI < 5.0 (excellent return)
    Blockbuster: ROI >= 5.0 (massive success)
    """
    if roi < -0.5:
        return 'Major Loss'
    elif roi < 0.5:
        return 'Break Even'
    elif roi < 2.0:
        return 'Profitable'
    elif roi < 5.0:
        return 'Very Profitable'
    else:
        return 'Blockbuster'

df_cleaned['success_class'] = df_cleaned['roi'].apply(define_roi_class)

print("\n=== ROI-Based Distribution ===")
print(df_cleaned['success_class'].value_counts().sort_index())
print("\nPercentages:")
print(df_cleaned['success_class'].value_counts(normalize=True).sort_index() * 100)

# For user-friendly display, also create simple 3-class version
def simple_success(roi):
    if roi < 0:
        return 'Unprofitable'
    elif roi < 2.0:
        return 'Profitable'
    else:
        return 'Highly Profitable'

df_cleaned['simple_success'] = df_cleaned['roi'].apply(simple_success)

print("\n=== Simplified 3-Class Distribution ===")
print(df_cleaned['simple_success'].value_counts())
print("\nPercentages:")
print(df_cleaned['simple_success'].value_counts(normalize=True) * 100)

# Check 40-60M movies
mid_budget = df_cleaned[(df_cleaned['budget'] >= 40e6) & (df_cleaned['budget'] <= 60e6)]
print(f"\n📊 $40-60M Budget Movies Analysis:")
print(mid_budget['simple_success'].value_counts(normalize=True) * 100)

# Use simple 3-class for training (more balanced)
y_success = df_cleaned['simple_success']

def define_collection_range(gross):
    if gross < 50_000_000:
        return 'Low (<$50M)'
    elif gross < 150_000_000:
        return 'Medium ($50M-$150M)'
    elif gross < 400_000_000:
        return 'High ($150M-$400M)'
    else:
        return 'Blockbuster (>$400M)'

df_cleaned['collection_range'] = df_cleaned['gross'].apply(define_collection_range)

print("\n=== Collection Range Distribution ===")
print(df_cleaned['collection_range'].value_counts())

# Features
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
y_collection_range = df_cleaned['collection_range']

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

print("\n=== Training Profitability Classifier ===")
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y_success, test_size=0.2, random_state=42, stratify=y_success
)

clf_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    ))
])

clf_pipeline.fit(X_train_s, y_train_s)
y_pred_s = clf_pipeline.predict(X_test_s)
score_s = accuracy_score(y_test_s, y_pred_s)

print(f"\nAccuracy: {score_s:.4f}")
print(classification_report(y_test_s, y_pred_s))

joblib.dump(clf_pipeline, 'success_classifier_model.joblib')
print("✅ Profitability Model saved")

print("\n=== Training Collection Range Classifier ===")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_collection_range, test_size=0.2, random_state=42, stratify=y_collection_range
)

range_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    ))
])

range_pipeline.fit(X_train_r, y_train_r)
y_pred_r = range_pipeline.predict(X_test_r)
score_r = accuracy_score(y_test_r, y_pred_r)

print(f"\nAccuracy: {score_r:.4f}")
print(classification_report(y_test_r, y_pred_r))

joblib.dump(range_pipeline, 'collection_range_model.joblib')
print("✅ Collection Model saved")

# Save metadata
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

print("\n" + "="*70)
print("✅ TRAINING COMPLETE - ROI-BASED MODEL")
print("="*70)
print(f"Profitability Accuracy: {score_s:.2%}")
print(f"Collection Accuracy: {score_r:.2%}")
print("\nNew categories:")
print("  Unprofitable: ROI < 0 (loses money)")
print("  Profitable: 0 <= ROI < 2.0 (makes profit)")
print("  Highly Profitable: ROI >= 2.0 (big success)")
print("="*70)