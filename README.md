# 🌍 Global AI Salary Predictor

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/guralemre/predicting-global-ai-salaries-hybrid-k-means-xg)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)

An end-to-end machine learning project that predicts annual salaries for AI/ML roles worldwide. Features a complete data science pipeline (EDA, clustering, regression) and an interactive **Streamlit** web application.

---

## 📖 Project Overview

**Datamin** analyzes a dataset of ~90,000 global AI job listings across 35 attributes, then builds a regression model to predict `salary_usd`. The workflow combines **K-Means clustering** (as a feature engineering step) with **XGBoost regression** for accurate salary estimation.

**Key capabilities:**

- Data cleaning, exploratory analysis, and correlation analysis
- Optimal cluster discovery via the elbow method (**KElbowVisualizer** / Silhouette Score)
- Decision Tree and XGBoost regression models with side-by-side comparison
- Instant salary prediction through a fast, cached Streamlit web interface

---

## 📊 Dataset

**Source:** [Kaggle — Global AI Jobs Dataset](https://www.kaggle.com/code/guralemre/predicting-global-ai-salaries-hybrid-k-means-xg)  
**File:** `global_ai_jobs.csv` (~17 MB, 90,000 rows, 35 columns)

Features include:

| Category | Features |
|----------|----------|
| **Role & Industry** | `job_role`, `ai_specialization`, `industry`, `education_required` |
| **Experience** | `experience_level`, `experience_years` |
| **Compensation** | `salary_usd` (target), `bonus_usd` |
| **Company** | `company_size`, `company_funding_billion`, `company_rating` |
| **Work** | `work_mode`, `interview_rounds` |
| **Scores** | `skill_demand_score`, `automation_risk`, `job_security_score`, `career_growth_score`, `work_life_balance_score` |
| **Economy & Location** | `country`, `cost_of_living_index`, `economic_index`, `tax_rate_percent`, `vacation_days` |

---

## 🔧 Pipeline

### 1. Data Loading & Cleaning
- Drops non-predictive columns: `id`, `bonus_usd`, `salary_percentile`
- Computes summary statistics and generates a correlation heatmap

### 2. Clustering (Feature Engineering)
- One-hot encodes categorical features and standardizes all numeric features
- **Two methods available:**
  - `main.py` — uses **KElbowVisualizer** (yellowbrick) to determine `k` dynamically
  - `app.py` — uses precomputed **optimal_k=4** (found via Silhouette Score analysis) for instant loading
- Cluster labels are added as a new feature for the regressor

### 3. Model Training
- **Decision Tree Regressor** — baseline / interpretable model
- **XGBoost Regressor** — primary model (`learning_rate=0.1`, `max_depth=6`, `n_estimators=150`)

### 4. Evaluation
- Metrics: **RMSE** (lower is better) and **R²** (higher is better)
- Both models compared on the same train/test split

### 5. Live Prediction
- The Streamlit app exposes the trained XGBoost model for interactive use
- Users input experience, role, country, company details, and more to get a real-time salary estimate

---

## 🚀 How to Run

### Prerequisites

**Option A — Conda (recommended):**

```bash
conda env create -f environment.yaml
conda activate datam_env
```

**Option B — pip:**

```bash
pip install -r requirements.txt
```

> **Note:** Python 3.13 is recommended (the environment was built on Anaconda with Python 3.13).

### Data Science Pipeline

Run the full analysis script to see EDA, clustering, model training, and evaluation:

```bash
python main.py
```

This will display statistical summaries, correlation heatmaps, elbow plots, and print RMSE / R² scores for both models.

### Streamlit Web App

Launch the interactive salary prediction dashboard:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 📁 Project Structure

```
datamin/
├── app.py                  # Streamlit web application (cached pipeline + UI)
├── main.py                 # Full data science pipeline script (EDA → clustering → models)
├── global_ai_jobs.csv      # Dataset (~90k rows, 35 columns)
├── environment.yaml        # Conda environment specification
├── requirements.txt        # pip dependencies
└── README.md               # This file
```

---

## 📈 Models

| Model              | Type     | Role                      | Key Hyperparameters                          |
|--------------------|----------|---------------------------|----------------------------------------------|
| Decision Tree      | Baseline | Simple, interpretable     | Default scikit-learn settings                |
| XGBoost Regressor  | Primary  | High-accuracy prediction  | `learning_rate=0.1`, `max_depth=6`, `n_estimators=150` |

The **XGBoost** model serves as the primary predictor, used both in the pipeline evaluation and the Streamlit app. The Decision Tree model is kept for comparison and interpretability.

### Hybrid Approach: K-Means + XGBoost

1. K-Means clustering discovers latent job market segments (e.g., "high-salary senior roles", "entry-level positions")
2. Cluster labels are fed as an additional feature into XGBoost
3. This hybrid approach helps the regressor capture non-linear group patterns that raw features alone might miss

---

## 🛠️ Tech Stack

| Tool / Library    | Usage                           |
|-------------------|---------------------------------|
| Python 3.13       | Core language                   |
| Pandas            | Data manipulation & cleaning    |
| NumPy             | Numerical operations            |
| Matplotlib        | Static plots & visualizations   |
| Seaborn           | Statistical charting             |
| Scikit-learn      | Train/test split, scaling, DT   |
| XGBoost           | Gradient-boosted regression     |
| Yellowbrick       | Elbow method visualization      |
| Streamlit         | Interactive web UI              |

---

## 📄 License

This project is for educational and portfolio purposes. The dataset and code are provided as-is.

---

## 👤 Author

**Emre Gural** — [@guralemre](https://github.com/guralemre)

**Kaggle Notebook:** [Predicting Global AI Salaries — Hybrid K-Means + XGBoost](https://www.kaggle.com/code/guralemre/predicting-global-ai-salaries-hybrid-k-means-xg)
