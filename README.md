# 🎬 Movie Box Office Success Predictor

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

An AI-powered web application that predicts movie box office success using machine learning. Built with Streamlit, scikit-learn, and Gradient Boosting models.



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

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/movie-box-office-predictor.git
cd movie-box-office-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure model files exist**
Make sure you have these files in your project directory:
- `success_classifier_model.joblib`
- `collection_range_model.joblib`
- `metadata.joblib`
- `lookup_tables.joblib`

4. **Run the application**
```bash
streamlit run app_v2_fixed.py
```

5. **Open your browser**
Navigate to `http://localhost:8501`

## 📦 Project Structure

```
movie-box-office-predictor/
├── app_v2_fixed.py              # Main Streamlit application
├── success_classifier_model.joblib   # Trained success prediction model
├── collection_range_model.joblib     # Trained collection prediction model
├── metadata.joblib                   # Genre, director, actor metadata
├── lookup_tables.joblib              # Director/actor success statistics
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── create_lookup_tables.py          # Script to generate lookup tables (optional)
└── screenshots/                      # Application screenshots (optional)
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

## 🎯 Usage

### Single Movie Prediction

1. Navigate to **"🎬 Single Prediction"** from the sidebar
2. Fill in movie details:
   - Budget (in millions USD)
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
2. Upload a CSV file with these columns:
   - `budget` - Movie budget in USD
   - `imdb_score` - Expected rating (1-10)
   - `duration` - Runtime in minutes
   - `genre_main` - Primary genre
   - `director_name` - Director's name
   - `actor_1_name` - Lead actor's name
   - `language` - Movie language
   - `movie_title` - (Optional) Movie title
3. Click **"🚀 Run Predictions"**
4. Download results as CSV

### Example CSV Format

```csv
movie_title,budget,imdb_score,duration,genre_main,director_name,actor_1_name,language
Interstellar,165000000,8.6,169,Sci-Fi,Christopher Nolan,Matthew McConaughey,English
The Dark Knight,185000000,9.0,152,Action,Christopher Nolan,Christian Bale,English
```

## 🧪 Testing

### Built-in Test

Use the **Interstellar Test** on the home page:
1. Go to **"🏠 Home"**
2. Expand **"🧪 Test with Interstellar"**
3. Click **"Run Interstellar Test"**
4. Verify prediction matches expected results

### Command Line Test

```bash
python -c "
import joblib, pandas as pd, numpy as np

model = joblib.load('success_classifier_model.joblib')
lookup = joblib.load('lookup_tables.joblib')

# Interstellar test
budget = 165_000_000
rating = 8.6
duration = 169

director_stats = lookup['director_lookup']['Christopher Nolan']
actor_stats = lookup['actor_lookup']['Matthew McConaughey']

input_data = pd.DataFrame({
    'budget': [budget],
    'imdb_score': [rating],
    'duration': [duration],
    'budget_per_minute': [budget/duration],
    'log_budget': [np.log1p(budget)],
    'rating_budget_interaction': [rating * np.log1p(budget)],
    'log_director_success': [director_stats['log_director_success']],
    'log_actor_success': [actor_stats['log_actor_success']],
    'genre_main': ['Sci-Fi'],
    'director_name': ['Christopher Nolan'],
    'actor_1_name': ['Matthew McConaughey'],
    'language': ['English'],
    'rating_tier': ['High'],
    'budget_tier': ['Ultra'],
    'director_popularity': [str(director_stats['director_popularity'])],
    'actor_popularity': [str(actor_stats['actor_popularity'])]
})

pred = model.predict(input_data)[0]
print(f'Prediction: {pred}')
"
```

## 📊 Model Performance

### Success Classifier
- **Accuracy**: 73.45%
- **Classes**: Blockbuster, Hit, Flop
- **Algorithm**: Gradient Boosting

### Collection Range Predictor
- **Accuracy**: 79.12%
- **Predicts**: Revenue ranges
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

The model uses several engineered features:

```python
# Derived features
budget_per_minute = budget / duration
log_budget = log(budget + 1)
rating_budget_interaction = rating × log_budget

# Categorical tiers
rating_tier = 'High' if rating >= 7.5 else 'Medium' if rating >= 6.0 else 'Low'
budget_tier = 'Ultra' if budget >= 150M else 'High' if budget >= 60M else 'Medium' if budget >= 20M else 'Low'
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

# This creates lookup_tables.joblib with director/actor statistics
```

### Custom Thresholds

Modify budget/rating tiers in `app_v2_fixed.py`:

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

Or the app will use default values (less accurate).

### Issue: "Model files not found"

**Solution**: Ensure these files exist in the project directory:
- `success_classifier_model.joblib`
- `collection_range_model.joblib`
- `metadata.joblib`

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
- [ ] Docker containerization
- [ ] Unit tests and CI/CD

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 🙏 Acknowledgments

- Dataset: [IMDb Movie Database](https://www.imdb.com/) 
- Inspiration: Box office analysis and prediction research
- Built with: Streamlit, scikit-learn, Plotly





## 🌟 Show Your Support

Give a ⭐️ if this project helped you!

---

**Made with ❤️ and Python**

*Last Updated: November 2024*
