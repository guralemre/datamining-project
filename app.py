import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from yellowbrick.cluster import KElbowVisualizer

# Sayfa Ayarları
st.set_page_config(page_title="Global AI Salary Predictor", page_icon="🤖", layout="wide")


# =====================================================================
# VERİ VE MODEL HAZIRLAMA (Arka Planda Çalışan Altyapı)
# =====================================================================
@st.cache_resource
def load_and_train_pipeline():
    df = pd.read_csv('global_ai_jobs.csv')
    columns_to_drop = ['id', 'bonus_usd', 'salary_percentile']
    df_cleaned = df.drop(columns=columns_to_drop)

    X = df_cleaned.drop(['salary_usd'], axis=1)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # UI'da gösterilecek tek kategorik alanlar
    UI_CATEGORICAL_COLS = ['job_role', 'country']
    categorical_options = {col: sorted(X[col].unique().tolist()) for col in UI_CATEGORICAL_COLS if col in X.columns}

    # Gizli alanların varsayılan değerleri (Mod)
    hidden_defaults = {}
    for col in categorical_cols:
        if col not in UI_CATEGORICAL_COLS:
            hidden_defaults[col] = X[col].mode()[0]

    # One-Hot Encoding
    X_encoded = pd.get_dummies(X, drop_first=True)

    # K-Means Kümeleme (Model girdisi olarak kullanılmak üzere)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)

    kmeans = KMeans(random_state=17)
    elbow = KElbowVisualizer(kmeans, k=(2, 15), timings=False)
    elbow.fit(X_scaled_df)
    optimal_k = elbow.elbow_value_ if elbow.elbow_value_ is not None else 4

    final_kmeans = KMeans(n_clusters=optimal_k, random_state=17).fit(X_scaled_df)
    df_cleaned['cluster'] = final_kmeans.labels_ + 1

    # XGBoost Model Eğitimi
    X_final = X_encoded.copy()
    X_final['k_means_cluster'] = df_cleaned['cluster']
    y = df_cleaned['salary_usd']

    X_train, _, y_train, _ = train_test_split(X_final, y, test_size=0.2, random_state=42)

    xgb_model = XGBRegressor(learning_rate=0.1, max_depth=6, n_estimators=150, random_state=42)
    xgb_model.fit(X_train, y_train)

    return scaler, final_kmeans, xgb_model, X_train.columns, categorical_options, hidden_defaults


# Arka plan yükleme hazırlığı
with st.spinner("Veri modeli arka planda hazırlanıyor..."):
    scaler, final_kmeans, xgb_model, train_columns, categorical_options, hidden_defaults = load_and_train_pipeline()

# =====================================================================
# ARAYÜZ TASARIMI (Sadece Maaş Tahmini)
# =====================================================================
st.title("🤖 Global AI Jobs — Canlı Maaş Tahmin Paneli")
st.markdown(
    "Aday kriterlerini seçerek, yapay zeka modeline göre hesaplanan tahmini yıllık maaşı anlık olarak görebilirsiniz.")
st.write("---")

# Ekranı iki sütuna bölüyoruz
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Aday Kriterleri")

    # Sayısal Girdiler
    experience_years = st.slider("Deneyim Yılı (Experience Years)", min_value=0, max_value=30, value=2)
    weekly_hours = st.slider("Haftalık Çalışma Saati (Weekly Hours)", min_value=10, max_value=80, value=40, step=5)

    # Kategorik Girdiler
    user_selections = {}
    for col_name, options in categorical_options.items():
        display_label = col_name.replace('_', ' ').title()
        user_selections[col_name] = st.selectbox(f"{display_label} Seçin", options)

with col2:
    st.subheader("🔮 Simülasyon Çıktısı")

    # 1. Boş şablon oluşturma
    new_candidate_template = pd.DataFrame(0, index=[0], columns=train_columns)

    # 2. UI sayısal değerlerini yazma
    new_candidate_template['experience_years'] = experience_years
    new_candidate_template['weekly_hours'] = weekly_hours

    # 3. UI kategorik seçimlerini aktarma
    for col_name, selected_val in user_selections.items():
        dummy_col = f"{col_name}_{selected_val}"
        if dummy_col in new_candidate_template.columns:
            new_candidate_template[dummy_col] = 1

    # 4. Arka planda sabitlenen gizli değişkenleri ekleme
    for col_name, default_val in hidden_defaults.items():
        dummy_col = f"{col_name}_{default_val}"
        if dummy_col in new_candidate_template.columns:
            new_candidate_template[dummy_col] = 1

    # 5. K-Means Küme Tespiti (Gizli hesaplama)
    candidate_for_cluster = new_candidate_template.drop(columns=['k_means_cluster'], errors='ignore')
    candidate_scaled = scaler.transform(candidate_for_cluster)
    assigned_cluster = final_kmeans.predict(candidate_scaled)[0] + 1
    new_candidate_template['k_means_cluster'] = assigned_cluster

    # 6. XGBoost Maaş Tahmini
    live_prediction = xgb_model.predict(new_candidate_template)[0]

    # Sonuç kartının tek başına gösterilmesi
    st.write("##")
    st.metric(label="Öngörülen Yıllık Maaş (XGBoost)", value=f"${live_prediction:,.2f}")

    # Bilgilendirme Kutusu
    st.write("##")
    st.info(
        f"**Profil Özeti:** {user_selections.get('country', 'Seçili ülkede')} lokasyonunda, "
        f"{experience_years} yıl deneyim ile haftalık {weekly_hours} saat çalışan bir "
        f"**{user_selections.get('job_role', 'Yapay Zeka Çalışanı')}** profili simüle edilmiştir."
    )