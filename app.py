import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tsvr import TwinSVR
import matplotlib.pyplot as plt


# =============================
# 🎨 Tambahan CSS untuk tema pink
# =============================
def add_custom_css():
    st.markdown("""
    <style>
    .main { background-color: #fff5f7; }
    .stSidebar { background-color: #ffe6eb; }
    h1, h2, h3, h4 { color: #b03060; }
    .metric-card {
        background-color: #ffe6eb;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0px 2px 5px rgba(176, 48, 96, 0.2);
    }
    .metric-label { color: #800040; font-weight: bold; }
    .metric-value { color: #b03060; font-size: 22px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)


# =============================
# 📊 Fungsi Prediksi & Evaluasi
# =============================
def predict_january_2025(model, X_columns, scaler_X, scaler_y, df_test):
    start_date = pd.to_datetime("2025-01-01")
    end_date = pd.to_datetime("2025-01-07")
    date_range = pd.date_range(start=start_date, end=end_date)
    predictions = []

    for date in date_range:
        features = {
            'Hari': date.day,
            'Bulan': date.month,
            'Tahun': date.year,
            'HariKe': date.dayofyear,
            'HariMinggu': date.weekday()
        }
        
        features_df = pd.DataFrame([features])[X_columns]
        features_scaled = scaler_X.transform(features_df)
        
        prediction_scaled = model.predict(features_scaled)
        if prediction_scaled.ndim > 1:
            prediction_scaled = prediction_scaled.ravel()
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
        
        actual_value = None
        
        if "Tanggal" in df_test.columns and date in df_test["Tanggal"].values:
            actual_value = df_test.loc[df_test["Tanggal"] == date, "Harga"].values[0]
        predictions.append((date, prediction[0][0], actual_value))

    return pd.DataFrame(predictions, columns=["Tanggal", "Prediksi", "Aktual"])


def evaluate_forecast(model, feature_cols, scaler_X, scaler_y, df_test):
    df_pred = predict_january_2025(model, feature_cols, scaler_X, scaler_y, df_test)
    df_eval = df_pred.dropna(subset=["Aktual"])
    st.markdown("### 🎯 Hasil Evaluasi Model")

    if not df_eval.empty:
        errors = df_eval["Aktual"] - df_eval["Prediksi"]
        mape = mean_absolute_percentage_error(df_eval["Aktual"], df_eval["Prediksi"]) * 100
        mse = mean_squared_error(df_eval["Aktual"], df_eval["Prediksi"])
        r2 = r2_score(df_eval["Aktual"], df_eval["Prediksi"])

        # --- Card Metrik
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>MAPE</div><div class='metric-value'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>MSE</div><div class='metric-value'>{mse:.3f}</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>R²</div><div class='metric-value'>{r2:.3f}</div></div>", unsafe_allow_html=True)

        # --- Grafik di bawah metrik
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_pred["Tanggal"], df_pred["Prediksi"], label="Prediksi", marker="o", color="#b03060")
        if df_pred["Aktual"].notna().any():
            ax.plot(df_pred["Tanggal"], df_pred["Aktual"], label="Aktual", marker="x", color="#505fd4")
        ax.set_title("Grafik Prediksi vs Aktual (Januari 2025)", color="#800040", fontsize=14)
        ax.legend()
        st.pyplot(fig)

        st.markdown("#### 📅 Data Prediksi dan Aktual")
        st.dataframe(df_pred)
    else:
        st.info("Tidak ada data aktual Januari 2025 untuk evaluasi.")


# =============================
# 📦 MAIN APP
# =============================
def main():
    st.set_page_config(page_title="📈 Dashboard Prediksi Harga Jagung", layout="wide")
    add_custom_css()

    st.title("🌽 Prediksi Harga Konsumen Jagung - Kabupaten Sampang")
    st.markdown("Aplikasi ini membandingkan **TSVR, SVR, dan RFR** dalam memprediksi harga jagung berdasarkan data historis.")

    # --- Load Data
    @st.cache_data
    def load_data():
       # === 1️⃣ Path Data ===
        local_train = r"D:\PUTRI\COOLYEAH\8~\BAB IV\streamlit\data\data_train.csv"
        local_test  = r"D:\PUTRI\COOLYEAH\8~\BAB IV\streamlit\data\data_test.csv"
        
        repo_train = os.path.join("data", "data_train.csv")
        repo_test  = os.path.join("data", "data_test.csv")
        
        # === 2️⃣ Baca File Data ===
        if os.path.exists(local_train):
            df_train = pd.read_csv(local_train)
            df_test  = pd.read_csv(local_test)
            print("📂 Menggunakan data dari lokal")
        else:
            df_train = pd.read_csv(repo_train)
            df_test  = pd.read_csv(repo_test)
            print("☁️ Menggunakan data dari GitHub/Streamlit Cloud")

        drop_cols = ['Pasar Rongtengah', 'Pasar Srimangunan', ' Pasar Rongtengah', ' Pasar Srimangunan']
        df_train = df_train.drop(columns=drop_cols, errors="ignore").rename(columns={'Rata-rata': 'Harga'})
        df_test = df_test.drop(columns=drop_cols, errors="ignore").rename(columns={' Rata-rata': 'Harga'})

        for df in [df_train, df_test]:
            df["Harga"] = df["Harga"].astype(str).str.replace("Rp", "").str.replace(".", "").astype(float)
            df["Tanggal"] = pd.to_datetime(df["Tanggal"])
            df["Hari"] = df["Tanggal"].dt.day
            df["Bulan"] = df["Tanggal"].dt.month
            df["Tahun"] = df["Tanggal"].dt.year
            df["HariKe"] = df["Tanggal"].dt.dayofyear
            df["HariMinggu"] = df["Tanggal"].dt.dayofweek

        feature_cols = ['Hari', 'Bulan', 'Tahun', 'HariKe', 'HariMinggu']
        X_train = df_train[feature_cols]
        y_train = df_train["Harga"].values.reshape(-1, 1)
        X_test = df_test[feature_cols]
        y_test = df_test["Harga"].values.reshape(-1, 1)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train).ravel()

        return X_train_scaled, y_train_scaled, X_test, y_test, scaler_X, scaler_y, df_test, feature_cols

    X_train, y_train, X_test, y_test, scaler_X, scaler_y, df_test, feature_cols = load_data()

    # --- Pilih model
    regressor = st.radio("Pilih Metode Regresi:", ["Twin Support Vector Regression (TSVR)", "Support Vector Regression (SVR)", "Random Forest Regression (RFR)"], horizontal=True)

    # --- Pilihan kernel
    if regressor in ["Twin Support Vector Regression (TSVR)", "Support Vector Regression (SVR)"]:
        st.radio("Pilih Jenis Kernel:", ["RBF"], horizontal=True)

    # --- Skenario parameter
    param_grid_tsvr = [
        {"C1": 0.01, "C2": 0.01, "gamma": 0.001},
        {"C1": 0.1, "C2": 1.0, "gamma": 0.05},
        {"C1": 1.0, "C2": 1.0, "gamma": 0.1},
    ]
    param_grid_svr = [
        {"C": 0.1, "epsilon": 0.1, "gamma": 0.01},
        {"C": 1.0, "epsilon": 0.1, "gamma": 0.1},
        {"C": 10, "epsilon": 0.1, "gamma": 1.0},
    ]
    rf_scenarios = [
        {"n_estimators": 50, "max_depth": None, "random_state": 42},
        {"n_estimators": 100, "max_depth": 5, "random_state": 42},
        {"n_estimators": 200, "max_depth": 10, "random_state": 42},
    ]

    if regressor == "Twin Support Vector Regression (TSVR)":
        scenario_options = [f"C1={p['C1']}, C2={p['C2']}, γ={p['gamma']}" for p in param_grid_tsvr]
        selected = st.selectbox("Pilih Skenario Parameter TSVR:", scenario_options)
        params = param_grid_tsvr[scenario_options.index(selected)]
        model = TwinSVR(C1=params["C1"], C2=params["C2"], kernel="rbf", gamma=params["gamma"])

    elif regressor == "Support Vector Regression (SVR)":
        scenario_options = [f"C={p['C']}, ε={p['epsilon']}, γ={p['gamma']}" for p in param_grid_svr]
        selected = st.selectbox("Pilih Skenario Parameter SVR:", scenario_options)
        params = param_grid_svr[scenario_options.index(selected)]
        model = SVR(C=params["C"], epsilon=params["epsilon"], kernel="rbf", gamma=params["gamma"])

    else:
        scenario_options = [f"Trees={p['n_estimators']}, Depth={p['max_depth']}" for p in rf_scenarios]
        selected = st.selectbox("Pilih Skenario Parameter RFR:", scenario_options)
        params = rf_scenarios[scenario_options.index(selected)]
        model = RandomForestRegressor(**params)

    st.markdown(f"### 🔧 Parameter Aktif: `{params}`")

    if st.button("🚀 Jalankan Model dan Tampilkan Hasil"):
        model.fit(X_train, y_train)
        evaluate_forecast(model, feature_cols, scaler_X, scaler_y, df_test)


if __name__ == "__main__":
    main()
