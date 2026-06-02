# Global AI Salary Predictor

An end-to-end machine learning project that predicts annual salaries for AI/ML roles worldwide. Features a complete data science pipeline (EDA, clustering, regression) and an interactive **Streamlit** web application.

---

## Project Overview

Datamin analyzes a dataset of ~90,000 global AI job listings across 35 attributes, then builds a regression model to predict `salary_usd`. The workflow combines **K-Means clustering** (as a feature engineering step) with **XGBoost regression** for accurate salary estimation.

**Key capabilities:**

- Data cleaning, exploratory analysis, and correlation analysis
- Optimal cluster discovery via the elbow method
- Decision Tree and XGBoost regression models with side-by-side comparison
- Interactive salary prediction through a Streamlit web interface

---

## Dataset

**File:** `global_ai_jobs.csv` (~17 MB, 90,000 rows, 35 columns)

The dataset covers global AI/ML job postings with features including:

- **Role & Industry:** `job_role`, `ai_specialization`, `industry`, `education_required`
- **Experience:** `experience_level`, `experience_years`
- **Compensation:** `salary_usd`, `bonus_usd`
- **Company:** `company_size`, `company_funding_billion`, `company_rating`
- **Work:** `work_mode`, `weekly_hours`, `interview_rounds`
- **Scores:** `skill_demand_score`, `automation_risk`, `job_security_score`, `career_growth_score`, `work_life_balance_score`, `employee_satisfaction`
- **Economy & Location:** `country`, `cost_of_living_index`, `economic_index`, `tax_rate_percent`, `vacation_days`

---

## Pipeline

### 1. Data Loading & Cleaning
- Drops non-predictive columns: `id`, `bonus_usd`, `salary_percentile`
- Computes summary statistics and generates a correlation heatmap

### 2. Clustering (Feature Engineering)
- One-hot encodes categorical features and standardizes all numeric features
- Runs **K-Means** with `KElbowVisualizer` (yellowbrick) to determine the optimal number of clusters
- Cluster labels are added as a new feature for the regressor

### 3. Model Training
- **Decision Tree Regressor** — baseline model
- **XGBoost Regressor** — primary model (`learning_rate=0.1`, `max_depth=6`, `n_estimators=150`)

### 4. Evaluation
- Metrics: RMSE and R² score
- Models are compared on the same train/test split

### 5. Live Prediction
- The Streamlit app exposes the trained XGBoost model for interactive use
- Users input experience, hours, role, and country to get a real-time salary estimate

---

## How to Run

### Prerequisites

The project uses a **Conda** environment. Create it from the provided spec:

```bash
conda env create -f environment.yaml
conda activate datam_env
```

Alternatively, install the key dependencies manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost yellowbrick streamlit
```

> **Note:** Python 3.13 is recommended (the environment was built on Anaconda with Python 3.13).

### Data Science Pipeline

Run the full analysis script to see EDA, clustering, model training, and evaluation:

```bash
python main.py
```

This will display statistical summaries, correlation heatmaps, elbow plots, and print RMSE/R² scores for both models.

### Streamlit Web App

Launch the interactive salary prediction dashboard:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Project Structure

```
datamin/
├── app.py                  # Streamlit web application
├── main.py                 # Data science pipeline script
├── global_ai_jobs.csv      # Dataset (~90k rows, 35 columns)
├── environment.yaml        # Conda environment specification
└── README.md               # This file
```

---

## Models

| Model              | Type     | Role                      |
|--------------------|----------|---------------------------|
| Decision Tree      | Baseline | Simple interpretable model|
| XGBoost Regressor  | Primary  | High-accuracy prediction  |

The XGBoost model is the primary predictor, used both in the pipeline evaluation and the Streamlit app. Decision Tree is kept for comparison.

---

## Tech Stack

| Tool/Library    | Usage                           |
|-----------------|---------------------------------|
| Python 3.13     | Core language                   |
| Pandas          | Data manipulation & cleaning    |
| NumPy           | Numerical operations            |
| Matplotlib      | Static plots & visualizations   |
| Seaborn         | Statistical charting             |
| Scikit-learn    | Train/test split, scaling, DT   |
| XGBoost         | Gradient-boosted regression     |
| Yellowbrick     | Elbow method visualization      |
| Streamlit       | Interactive web UI              |

---

## License

This project is for educational and portfolio purposes. The dataset and code are provided as-is.

---

## Author

**Emre Gural** — [@guralemre](https://github.com/guralemre)
