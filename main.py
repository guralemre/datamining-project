import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# STEP 1: DATA LOADING, CLEANING & EDA (Exploratory Data Analysis)
# =====================================================================
df = pd.read_csv('global_ai_jobs.csv')

df.info()

df.describe()

# Dropping columns that don't contribute to the model or are unknown during prediction
columns_to_drop = ['id', 'bonus_usd', 'salary_percentile']
df_cleaned = df.drop(columns=columns_to_drop)

# Selecting only numerical variables for correlation and summary statistics
numeric_cols = df_cleaned.select_dtypes(include=[np.number])

def num_summary(data, numerical_col, plot=False):
    quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(f"\n--- Statistical Summary of {numerical_col} ---")
    print(data[numerical_col].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(8, 5))
        data[numerical_col].hist(bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        plt.xlabel(numerical_col.replace('_', ' ').title(), fontsize=12)
        plt.ylabel('Frequency (Number of Employees)', fontsize=12)
        plt.title(f'Distribution of {numerical_col.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show(block=True)

print("Step 1.1: Target and Key Variables Distribution")
key_columns = ['salary_usd', 'experience_years']
for col in key_columns:
    num_summary(numeric_cols, col, plot=True)

print("\nStep 1.2: Feature Correlation Heatmap")
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            linewidths=0.5,
            annot_kws={"size": 7})

plt.title("Correlation Matrix of Dataset Variables", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# =====================================================================
# STEP 2 & 3: K-MEANS CLUSTERING & ELBOW METHOD
# =====================================================================
print("\n--- Step 2: K-Means Clustering and Elbow Method ---")

# Preparing data for clustering (Encoding categorical variables)
X = df_cleaned.drop(['salary_usd'], axis=1)
X_encoded = pd.get_dummies(X, drop_first=True)

# Standardizing the data for distance-based K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)

# Elbow Method to find the optimal number of clusters
plt.figure(figsize=(10, 6))
kmeans = KMeans(random_state=17)
elbow = KElbowVisualizer(kmeans, k=(2, 15), timings=False)
elbow.fit(X_scaled_df)

plt.title("Determining Optimal Number of Clusters (Elbow Method)", fontsize=14, fontweight='bold')
elbow.show()

optimal_k = elbow.elbow_value_
print(f"\nAlgorithm determined the optimal number of clusters as: {optimal_k}")

# Fitting the final K-Means model
final_kmeans = KMeans(n_clusters=optimal_k, random_state=17).fit(X_scaled_df)
clusters = final_kmeans.labels_

# Adding cluster labels as a new feature to the original dataset
df_cleaned['cluster'] = clusters + 1

cluster_summary = df_cleaned.groupby('cluster')[['experience_years']].agg(['count', 'mean'])
print("\nCluster Profiles (Count and Average Experience Years):")
print(cluster_summary)

# =====================================================================
# STEP 4 & 5: MODEL TRAINING AND PERFORMANCE COMPARISON
# =====================================================================
print("\n--- Step 4: Model Training and Evaluation ---")

# Preparing final features including the newly created K-Means clusters
X_final = X_encoded.copy()
if 'cluster' in df_cleaned.columns:
    X_final['k_means_cluster'] = df_cleaned['cluster']

y = df_cleaned['salary_usd']

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# MODIFIED: Removed Linear Regression
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "XGBoost": XGBRegressor(learning_rate=0.1, max_depth=6, n_estimators=150, random_state=42)
}

results = []
best_model = None
best_y_pred = None

for name, model in models.items():
    # 1. Modeli eğitiyoruz
    model.fit(X_train, y_train)

    # 2. Hem Eğitim (Train) hem de Test verisi için tahmin alıyoruz
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 3. Metrikleri hesaplıyoruz
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))

    results.append({
        "Model": name,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Test RMSE": test_rmse
    })

    if name == "XGBoost":
        best_model = model
        best_y_pred = y_test_pred

results_df = pd.DataFrame(results)
print("\n--- Model Comparison Table (Overfitting) ---")
print(results_df.to_string(index=False))

# MODIFIED: Separated RMSE and R2 plots into two distinct subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. Subplot: RMSE Comparison (Bar Chart)
bars = ax1.bar(results_df['Model'], results_df['Test RMSE'], color=['#f39c12', '#27ae60'], alpha=0.85, width=0.4)
ax1.set_ylabel('RMSE (USD) - Lower is better', fontsize=12, fontweight='bold')
ax1.set_title('Test RMSE Comparison', fontsize=13, fontweight='bold')
ax1.set_ylim(0, results_df['Test RMSE'].max() * 1.2)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + (results_df['Test RMSE'].max() * 0.02),
             f'${int(yval):,}', ha='center', va='bottom', fontweight='bold')

# 2. Subplot: R² Score Comparison (Line/Point Chart)
ax2.plot(results_df['Model'], results_df['Test R2'], color='#c0392b', marker='o', markersize=10, linewidth=3, linestyle='--')
ax2.set_ylabel('R² Score - Higher is better', fontsize=12, fontweight='bold')
ax2.set_title('Test R² Score Comparison', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1.1)

for i, txt in enumerate(results_df['Test R2']):
    ax2.annotate(f"{txt:.4f}", (i, results_df['Test R2'].iloc[i] + 0.04), ha='center',
                 color='#c0392b', fontweight='bold', backgroundcolor='white')

plt.suptitle("Performance Comparison: Tree vs Ensemble Models", fontsize=16, fontweight='bold', y=0.98)
fig.tight_layout()
plt.show()

# Actual vs Predicted Salary Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_y_pred, alpha=0.5, color='#27ae60')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('XGBoost: Actual vs Predicted Salary')
plt.tight_layout()
plt.show()

# Feature Importance Plot (XGBoost)
feat_imp = pd.Series(best_model.feature_importances_, index=X_train.columns)
top_features = feat_imp.nlargest(15).sort_values()

plt.figure(figsize=(10, 7))
bars = plt.barh(top_features.index, top_features.values, color='#27ae60', alpha=0.85, edgecolor='black')

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
             f'{width:.4f}', va='center', fontsize=9, fontweight='bold')

plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 15 Feature Importances (XGBoost)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =====================================================================
# STEP 6: LIVE PREDICTION SIMULATION
# =====================================================================
print("\n==================================================")
print("  STEP 6: LIVE SIMULATION ")
print("==================================================\n")

new_candidate_scenario = {
    'experience_years': [2],
    'weekly_hours': [40],
    'country_Germany': [1],
    'job_role_Machine Learning Engineer': [1]
}

new_candidate_df = pd.DataFrame(new_candidate_scenario)
new_candidate_template = pd.DataFrame(0, index=[0], columns=X_train.columns)

for col in new_candidate_df.columns:
    if col in new_candidate_template.columns:
        new_candidate_template[col] = new_candidate_df[col]
    else:
        print(f"Warning: Feature '{col}' does not exist in the training data!")

candidate_for_cluster = new_candidate_template.drop(columns=['k_means_cluster'])
candidate_scaled = scaler.transform(candidate_for_cluster)
assigned_cluster = final_kmeans.predict(candidate_scaled)[0] + 1
new_candidate_template['k_means_cluster'] = assigned_cluster


live_prediction = best_model.predict(new_candidate_template)[0]

print(">>> SIMULATION RESULT <<<")
print(f"Input Criteria:")
for key, value in new_candidate_scenario.items():
    if value[0] == 1 and ('country_' in key or 'job_role_' in key):
        print(f" - {key.split('_', 1)[1]}")
    elif 'years' in key or 'hours' in key:
        print(f" - {key.replace('_', ' ').title()}: {value[0]}")

print(f" - Assigned Cluster: {assigned_cluster}")
print("\n" + "*"*40)
print(f"Predicted Salary (XGBoost): ${live_prediction:,.2f}")
print("*"*40 + "\n")