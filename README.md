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
