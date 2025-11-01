# 🎬 Movie Box Office Success Predictor V2

A **Machine Learning–powered Streamlit web app** that predicts a movie’s **success potential** and **box office collection range** based on its attributes — like genre, budget, director, actors, and more.

Built with advanced ML models (**GradientBoosting**, **RandomForest**, and optional **XGBoost**), this project offers **real-time predictions**, **data analytics**, **batch processing**, and **insights visualization** — all in a beautiful multi-page UI.

---

## 🚀 Features

### 🧠 Machine Learning
- Predicts **movie success level** (`Flop`, `Hit`, `Blockbuster`)
- Predicts **box office collection range** (`Low`, `Moderate`, `High`, `Very High`)
- Built with **GradientBoosting**, **RandomForest**, and **SMOTE balancing**
- Achieved up to **73% accuracy (Success Classifier)** and **79% accuracy (Collection Range Classifier)**

### 🧩 Application Pages
| Page | Description |
|------|--------------|
| 🏠 **Home** | Overview and quick project summary |
| 🎬 **Single Prediction** | Enter movie details and get instant predictions |
| 📊 **Batch Predictions** | Upload a CSV with multiple movies and get predictions |
| 📈 **Analytics Dashboard** | Visualize model accuracy, feature importance, and trends |
| 🔍 **Compare** | Compare two movie scenarios side-by-side |
| ℹ️ **About** | Info about the model, dataset, and developers |

### 🎨 UI/UX Highlights
- Multi-page navigation (via sidebar)
- Interactive charts using **Plotly**
- Modern gradient design with custom themes
- Searchable dropdowns and autocomplete inputs
- CSV export & prediction history tracking

---

## 🧰 Tech Stack

**Frontend:** [Streamlit](https://streamlit.io)  
**Backend / ML:** scikit-learn, imbalanced-learn, joblib  
**Data Visualization:** Plotly  
**Language:** Python 3.11+  
**Deployment:** Streamlit Cloud / Local  

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Movie-Box-Office-Success-Predictor.git
cd Movie-Box-Office-Success-Predictor
Create and Activate Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate    # On Windows
# OR
source .venv/bin/activate   # On macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt


Or install manually if requirements.txt not yet added:

pip install streamlit pandas scikit-learn imbalanced-learn joblib plotly streamlit-option-menu

⚙️ Usage
🧠 Step 1: Train the Models

Run the training script to generate model artifacts (.joblib files):

python movie_predictor_train.py


You’ll see accuracy and model stats in your terminal.

💻 Step 2: Run the Streamlit App

V1 (basic app):

streamlit run app.py


V2 (enhanced multi-page app):

streamlit run app_v2.py


Then open your browser at 👉 http://localhost:8501

📊 Results
Model	Accuracy	Best Algorithm
Success Classifier	73.45%	GradientBoosting
Collection Range	79.12%	GradientBoosting
📁 Project Structure
Movie-Box-Office-Success-Predictor/
│
├── app.py                      # Original app
├── app_v2.py                   # Enhanced V2 app (multi-page)
├── movie_predictor_train.py    # Model training script
├── preprocessor.joblib         # Saved data preprocessor
├── success_classifier_model.joblib
├── collection_range_model.joblib
├── metadata.joblib
│
├── pages/
│   ├── __init__.py
│   ├── home.py
│   ├── single_prediction.py
│   ├── batch_predictions.py
│   ├── analytics.py
│   ├── compare.py
│   └── about.py
│
├── data/                       # (Optional) Dataset files
└── README.md

💡 Future Enhancements (Planned for V3)

✨ Add XGBoost & hyperparameter tuning

🧩 SHAP explainability (“why did the model predict this?”)

🧮 Integration with TMDB API for real movie metadata

☁️ Cloud deployment (Streamlit Cloud / Hugging Face Spaces)

📱 Mobile-responsive UI

👨‍💻 Author

Pranesh Dharani
🎓 Computer Science Engineering @ SRMIST Chennai
📧 [Add your email or portfolio link if you want]

🧠 Acknowledgements

Dataset inspired by IMDb, TMDB, and Kaggle movie datasets.

Built with ❤️ using Python and Streamlit.
