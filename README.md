# 🎬 Movie Box Office Success Predictor (V2)

A multi-page Streamlit web application that uses machine learning to predict a movie's potential box office success (Flop, Hit, or Blockbuster) and its estimated gross collection range.

This V2 build features a robust multi-page interface, batch prediction capabilities, and an analytics dashboard, all powered by a GradientBoosting model with ~73-79% accuracy.



## ✨ Key Features (V2)

* **🏠 Home Dashboard:** A quick overview of the project and model performance.
* **🎬 Single Prediction:** The core predictor. Enter a movie's details (budget, genre, director, actors, etc.) and get an instant prediction for its success and collection.
* **📊 Batch Predictions:** Upload a CSV file with multiple movies to get predictions for your entire slate.
* **📈 Analytics & Insights:** An interactive dashboard (using Plotly) to explore the training data and understand relationships between features like budget, rating, and gross.
* **🔍 Movie Comparisons:** A side-by-side tool to compare the predicted outcomes of two different movie scenarios.
* **🤖 High-Accuracy Models:** Utilizes `GradientBoosting` and `SMOTE` to handle class imbalance, achieving:
    * **73.5% Accuracy** for Success Classification (Flop/Hit/Blockbuster)
    * **79.1% Accuracy** for Collection Range Classification

## 💻 Tech Stack

* **Python**
* **Streamlit:** For the multi-page web interface.
* **Scikit-learn:** For machine learning pipelines, feature engineering, and models (GradientBoosting).
* **Pandas:** For data manipulation and processing.
* **Imbalanced-learn:** For using SMOTE to correct class imbalance.
* **Plotly:** For interactive data visualizations on the Analytics page.
* **Streamlit-Searchbox:** For user-friendly autocomplete dropdowns.

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites

* Python 3.8+
* Git

### 2. Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Prxnesh/Movie-Box-office-Success-Predictor.git
    cd Movie-Box-office-Success-Predictor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

3.  **Install the required libraries:**
    *First, create a `requirements.txt` file if you don't have one. You can add the libraries from our chat or run `pip freeze > requirements.txt` after installing them.*

    ```bash
    # Key requirements to install
    pip install streamlit pandas scikit-learn plotly imbalanced-learn streamlit-option-menu streamlit-searchbox
    
    # Or, if you have a requirements.txt file:
    pip install -r requirements.txt
    ```

### 3. ❗ Important: Train the Models

You must run the training script first. This will generate the `.joblib` files (`success_classifier_model.joblib`, `collection_range_model.joblib`, `preprocessor.joblib`, and `metadata.joblib`) that the Streamlit app needs to run.

**This step fixes the `NameError: name 'metadata' is not defined` and `FileNotFoundError`.**

```bash
python movie_predictor_train.py
```
You will see the accuracy metrics printed in your terminal as the models are trained and saved.

4.  **Run the Streamlit App**

    Now that the models and metadata are saved, you can launch the V2 application:
    ```bash
    streamlit run app_v2.py
    ```
    Your app will open automatically in your web browser at `http://localhost:8501`.

---

## 📈 Model Performance

The initial simple model had low accuracy. The current models were significantly improved using advanced feature engineering, `SMOTE` to handle imbalanced data, and by comparing `RandomForest` and `GradientBoosting` to select the best performer.

| Classifier | Baseline Accuracy | V2 Model Accuracy |
| :--- | :---: | :---: |
| **Success (Flop/Hit/Blockbuster)** | ~49.1% | **73.45%** |
| **Collection Range** | ~61.9% | **79.12%** |

**Key Improvements:**
* **Feature Engineering:** Added `log_budget`, `director_avg_gross`, `actor_avg_gross`, and interaction features.
* **Class Imbalance:** Implemented `SMOTE` to prevent the model from only predicting "Flop."
* **Algorithm:** Switched to `GradientBoostingClassifier`, which outperformed `RandomForest`.

## License

This project is licensed under the MIT License.
