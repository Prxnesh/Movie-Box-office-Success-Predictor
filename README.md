
# 🎬 Movie Box Office Success Predictor

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

An AI-powered web application that predicts movie box office success using machine learning. Built with Streamlit, scikit-learn, and Gradient Boosting models.


## Table of Contents
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Usage](#-usage)
- [Model artifacts & where to get them](#-model-artifacts--where-to-get-them)
- [Model Performance](#-model-performance)
- [How It Works](#-how-it-works)
- [Analytics](#-analytics)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [To-Do](#-to-do)
- [License](#-license)


## ✨ Features

- 🎯 **Single Movie Predictions** - Predict box office success for individual movies
- 📊 **Batch Processing** - Upload CSV files to predict multiple movies at once
- 📈 **Analytics Dashboard** - Visualize prediction history and insights
- 🔍 **Movie Comparison** - Compare different movie scenarios side-by-side
- 📥 **Export Results** - Download predictions as CSV files
- 🧪 **Built-in Testing** - Test with famous movies like Interstellar
- 🎨 **Interactive UI** - Modern, responsive design with real-time feedback

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Clone the repository

```bash
git clone https://github.com/Prxnesh/Movie-Box-office-Success-Predictor.git
cd Movie-Box-office-Success-Predictor
```

### Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.\.venv\Scripts\activate  # Windows PowerShell
pip install --upgrade pip
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Ensure model files exist

Required model/artifact files (expected to live in project root):
- `success_classifier_model.joblib`
- `collection_range_model.joblib`
- `metadata.joblib`
- `lookup_tables.joblib`

Where to get them:
- Preferred: download from the repository Releases page (large files stored there) or via Git LFS if configured for this repo.
- If Releases are not available: use `create_lookup_tables.py` to generate `lookup_tables.joblib` and follow training scripts (if provided) to regenerate the models. If you don't have training data, look for a `models-sample` or `demo` release in Releases.

If model files are missing the app will fall back to safer defaults (less accurate); see Troubleshooting below.

### Run the application

```bash
streamlit run app_v2_fixed.py
```

Open your browser at `http://localhost:8501`.

## 📦 Project Structure

```
movie-box-office-predictor/
├── app_v2_fixed.py
├── success_classifier_model.joblib
├── collection_range_model.joblib
├── metadata.joblib
├── lookup_tables.joblib
├── requirements.txt
├── README.md
├── create_lookup_tables.py
└── screenshots/
```

## 📋 Requirements

Create a `requirements.txt` file with:

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
plotly>=5.17.0
```

For reproducible installs consider adding a lockfile or pinning exact versions in a `constraints.txt` or using `pyproject.toml` + a lock tool.

## 🎯 Usage

Important: budgets are expected as raw USD integers (e.g. 165000000 for 165 million USD). Example CSV and examples below use raw USD values.

### Single Movie Prediction

1. Navigate to **"🎬 Single Prediction"** from the sidebar
2. Fill in movie details:
   - Budget (in USD, integer)
   - Expected IMDB rating
   - Duration (minutes)
   - Genre
   - Director name
   - Main actor name
   - Language
3. Click **"🚀 Predict"**
4. View results including:
   - Success level (Blockbuster/Hit/Flop)
   - Collection range prediction
   - Confidence score
   - Probability distribution

### Batch Predictions

1. Navigate to **"📊 Batch Predictions"**
2. Upload a CSV file with these columns (recommended types shown):
   - `budget` (int) - Movie budget in USD
   - `imdb_score` (float) - Expected rating (1-10)
   - `duration` (int) - Runtime in minutes
   - `genre_main` (str) - Primary genre
   - `director_name` (str) - Director's name
   - `actor_1_name` (str) - Lead actor's name
   - `language` (str) - Movie language
   - `movie_title` (str, optional)

3. Click **"🚀 Run Predictions"**
4. Download results as CSV

### Example CSV Format

```csv
movie_title,budget,imdb_score,duration,genre_main,director_name,actor_1_name,language
Interstellar,165000000,8.6,169,Sci-Fi,Christopher Nolan,Matthew McConaughey,English
The Dark Knight,185000000,9.0,152,Action,Christopher Nolan,Christian Bale,English
```

### Example output CSV row (columns produced by the app)

```
movie_title,budget,imdb_score,duration,genre_main,director_name,actor_1_name,language,predicted_success,collection_range,confidence,prob_blockbuster,prob_hit,prob_flop
Interstellar,165000000,8.6,169,Sci-Fi,Christopher Nolan,Matthew McConaughey,English,Blockbuster,>$300M,0.92,0.90,0.08,0.02
```

## 🧪 Testing

### Built-in Test

Use the **Interstellar Test** on the home page:
1. Go to **"🏠 Home"**
2. Expand **"🧪 Test with Interstellar"**
3. Click **"Run Interstellar Test"**
4. Verify prediction matches expected results

### Command Line Test (example)

```bash
python - <<'PY'
import joblib, pandas as pd, numpy as np

model = joblib.load('success_classifier_model.joblib')
lookup = joblib.load('lookup_tables.joblib')

# Interstellar test
budget = 165_000_000
rating = 8.6
duration = 169

director_stats = lookup['director_lookup'].get('Christopher Nolan', {'log_director_success': 0.0, 'director_popularity': 0})
actor_stats = lookup['actor_lookup'].get('Matthew McConaughey', {'log_actor_success': 0.0, 'actor_popularity': 0})

input_data = pd.DataFrame({
    'budget': [budget],
    'imdb_score': [rating],
    'duration': [duration],
    'budget_per_minute': [budget/duration],
    'log_budget': [np.log1p(budget)],
    'rating_budget_interaction': [rating * np.log1p(budget)],
    'log_director_success': [director_stats.get('log_director_success', 0.0)],
    'log_actor_success': [actor_stats.get('log_actor_success', 0.0)],
    'genre_main': ['Sci-Fi'],
    'director_name': ['Christopher Nolan'],
    'actor_1_name': ['Matthew McConaughey'],
    'language': ['English'],
    'rating_tier': ['High'],
    'budget_tier': ['Ultra'],
    'director_popularity': [str(director_stats.get('director_popularity', 0))],
    'actor_popularity': [str(actor_stats.get('actor_popularity', 0))]
})

pred = model.predict(input_data)[0]
print(f'Prediction: {pred}')
PY
```

## 📊 Model Performance

### Success Classifier
- **Accuracy**: 73.45% (reported on validation/test split)
- **Classes**: Blockbuster, Hit, Flop
- **Algorithm**: Gradient Boosting

Please consider adding class-wise precision/recall/F1 and the dataset size / split used to compute these metrics in future updates for reproducibility.

### Collection Range Predictor
- **Accuracy**: 79.12%
- **Predicts**: Revenue ranges (binned)
- **Algorithm**: Gradient Boosting

### Key Features Used
1. Budget (USD)
2. IMDB Score
3. Duration (minutes)
4. Director Success History
5. Actor Success History
6. Genre
7. Language
8. Budget-Rating Interaction
9. Budget Tier
10. Rating Tier

## 🛠️ How It Works

### Feature Engineering

The model uses several engineered features (copy-paste friendly):

```python
# Derived features
budget_per_minute = budget / duration
log_budget = np.log1p(budget)
rating_budget_interaction = rating * log_budget

# Categorical tiers
rating_tier = 'High' if rating >= 7.5 else 'Medium' if rating >= 6.0 else 'Low'
budget_tier = 'Ultra' if budget >= 150_000_000 else 'High' if budget >= 60_000_000 else 'Medium' if budget >= 20_000_000 else 'Low'
```

### Director/Actor Success Scores

The app uses historical success rates:
- **log_director_success**: Logarithmic transformation of director's past success
- **log_actor_success**: Logarithmic transformation of actor's past success
- **Popularity tiers**: Experienced vs. New based on filmography

### Prediction Pipeline

1. **Input Validation** - Ensures all required fields are present
2. **Feature Engineering** - Creates derived features
3. **Lookup Stats** - Fetches director/actor historical data
4. **Model Prediction** - Runs through trained ML models
5. **Result Formatting** - Presents human-readable predictions

## 🔧 Configuration

### Model Files

If you need to retrain models or create lookup tables:

```bash
# Generate lookup tables from your training data
python create_lookup_tables.py
```

This creates `lookup_tables.joblib` with director/actor statistics.

### Custom Thresholds

Modify budget/rating tiers in `app_v2_fixed.py` (canonical source):

```python
# Budget tiers (in USD)
LOW_BUDGET = 20_000_000      # < 20M
MEDIUM_BUDGET = 60_000_000   # 20M - 60M
HIGH_BUDGET = 150_000_000    # 60M - 150M
# ULTRA_BUDGET > 150M

# Rating tiers
HIGH_RATING = 7.5    # >= 7.5
MEDIUM_RATING = 6.0  # 6.0 - 7.5
# LOW_RATING < 6.0
```

## 📈 Analytics

The analytics dashboard provides:
- **Prediction Distribution** - Pie chart of success categories
- **Budget Distribution** - Histogram of movie budgets
- **Prediction History** - Table of all past predictions
- **Export Capability** - Download history as CSV

## 🐛 Troubleshooting

### Issue: "Lookup tables not found"

**Solution**: 

```bash
# Create lookup tables from your training data
python create_lookup_tables.py
```

If you do not have training data, download `lookup_tables.joblib` from Releases or contact the maintainer.

### Issue: "Model files not found"

**Solution**: Ensure these files exist in the project directory:
- `success_classifier_model.joblib`
- `collection_range_model.joblib`
- `metadata.joblib`

Recommended steps:
1. Check Releases for model artifacts and download them.
2. If the repo uses Git LFS for models, clone with Git LFS enabled: `git lfs install && git lfs pull`.
3. If you have training data, run training scripts (not included by default) or contact the maintainer for guidance.

### Issue: Predictions seem inaccurate

**Solution**: 
1. Check if lookup tables are loaded (sidebar indicator)
2. Verify director/actor names match training data
3. Use debug expander to see actual feature values

### Issue: CSV upload fails

**Solution**: Ensure CSV has required columns:
- `budget`, `imdb_score`, `duration`, `genre_main`, `director_name`, `actor_1_name`, `language`

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please also consider adding these files to help contributors:
- `CONTRIBUTING.md` (how to run tests, coding style, PR process)
- `CODE_OF_CONDUCT.md`

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Update README for new features
- Test thoroughly before submitting

## 📝 To-Do

- [ ] Add SHAP values for prediction explainability
- [ ] Implement movie comparison feature
- [ ] Add more visualizations (ROI calculator, budget optimizer)
- [ ] Support for multiple actors
- [ ] Historical trend analysis
- [ ] API endpoint for predictions
- [ ] Docker containerization (see suggested Dockerfile)
- [ ] Unit tests and CI/CD

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔐 Privacy & Data Handling

Uploaded CSVs are processed in-memory for predictions. By default the app does not persist uploaded CSV files to disk. If you plan to host a public demo, add a short privacy notice describing data retention and use.

## 🙏 Acknowledgments

- Dataset: [IMDb Movie Database](https://www.imdb.com/) 
- Inspiration: Box office analysis and prediction research
- Built with: Streamlit, scikit-learn, Plotly


## 🌟 Show Your Support

Give a ⭐️ if this project helped you!

---

**Made with ❤️ and Python**

*Last Updated: auto-generated from repo commit date*
```
