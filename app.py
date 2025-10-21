import os
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
    p, label, div, span {
        font-size: 14px !important;
        color: #333333;
    }
    .metric-card {
        background-color: #ffe6eb;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0px 2px 5px rgba(176, 48, 96, 0.2);
    }
    .metric-label { color: #800040; font-weight: bold; font-size: 14px; }
    .metric-value { color: #b03060; font-size: 18px; font-weight: bold; }
    .stRadio > label, .stSelectbox > label, .stMarkdown p {
        font-size: 14px !important;
        color: #800040 !important;
        font-weight: 600;
    }
    .stMarkdown code {
        background-color: #ffe6eb;
        color: #800040;
        padding: 3px 6px;
        border-radius: 5px;
        font-size: 13px;
    }
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
       # --- Hitung metrik evaluasi
        mape = mean_absolute_percentage_error(df_eval["Aktual"], df_eval["Prediksi"]) * 100
        mse = mean_squared_error(df_eval["Aktual"], df_eval["Prediksi"])
        r2 = r2_score(df_eval["Aktual"], df_eval["Prediksi"])
        sse = np.sum(np.square(errors))
        error_var = np.var(errors, ddof=1)  # ddof=1 agar sesuai dengan var sampel

        # --- Card Metrik
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>MAPE</div><div class='metric-value'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>MSE</div><div class='metric-value'>{mse:,.3f}</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>R²</div><div class='metric-value'>{r2:.3f}</div></div>", unsafe_allow_html=True)

        col4, col5 = st.columns(2)
        with col4:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>SSE</div><div class='metric-value'>{sse:,.3f}</div></div>", unsafe_allow_html=True)
        with col5:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Error Variance</div><div class='metric-value'>{error_var:,.3f}</div></div>", unsafe_allow_html=True)

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
       # --- Tambahkan rata-rata prediksi dan aktual selama 1 minggu
    avg_pred = df_pred["Prediksi"].mean()
    avg_actual = df_pred["Aktual"].mean() if "Aktual" in df_pred.columns else None

# --- Tampilkan dua metrik berdampingan
    if avg_actual is not None:
        st.markdown(f"""
        <div style='display: flex; justify-content: center; gap: 40px; margin-top: 20px;'>
            <div class='metric-card' style='width: 45%;'>
                <div class='metric-label'>📊 Rata-rata Prediksi Harga (1–7 Januari 2025)</div>
                <div class='metric-value'>Rp {avg_pred:,.0f}</div>
            </div>
            <div class='metric-card' style='width: 45%;'>
                <div class='metric-label'>💰 Rata-rata Aktual Harga (1–7 Januari 2025)</div>
                <div class='metric-value'>Rp {avg_actual:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Jika tidak ada kolom 'Aktual', tampilkan hanya prediksi
        st.markdown(f"""
        <div class='metric-card' style='width: 50%; margin: 20px auto;'>
            <div class='metric-label'>📊 Rata-rata Prediksi Harga (1–7 Januari 2025)</div>
            <div class='metric-value'>Rp {avg_pred:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)


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

        # === 2️⃣ Baca File Data (prioritaskan lokal, kalau tidak ada pakai repo) ===
        if os.path.exists(local_train) and os.path.exists(local_test):
            df_train = pd.read_csv(local_train)
            df_test  = pd.read_csv(local_test)
            st.write("📂 Menggunakan data dari lokal")
        elif os.path.exists(repo_train) and os.path.exists(repo_test):
            df_train = pd.read_csv(repo_train)
            df_test  = pd.read_csv(repo_test)
            st.write("☁️ Menggunakan data dari folder `data/` (repo)")
        else:
            st.error(
                "File data_train.csv atau data_test.csv tidak ditemukan di path lokal maupun folder `data/`.\n"
                "Pastikan file tersedia dan path sudah benar."
            )
            # Kembalikan struktur kosong agar app tidak crash lebih lanjut
            return np.empty((0, 5)), np.empty((0,)), np.empty((0, 5)), np.empty((0,)), None, None, pd.DataFrame(), []

        # Hapus kolom pasar yang tidak diperlukan jika ada, lalu seragamkan nama kolom Harga
        drop_cols = ['Pasar Rongtengah', 'Pasar Srimangunan', ' Pasar Rongtengah', ' Pasar Srimangunan']
        df_train = df_train.drop(columns=drop_cols, errors="ignore").rename(columns={'Rata-rata': 'Harga', ' Rata-rata': 'Harga'})
        df_test  = df_test.drop(columns=drop_cols, errors="ignore").rename(columns={'Rata-rata': 'Harga', ' Rata-rata': 'Harga'})

        # === 3️⃣ Bersihkan kolom Harga dan Tanggal untuk tiap dataframe ===
        def clean_df(df):
            # pastikan kolom Harga dan Tanggal ada
            if "Harga" not in df.columns or "Tanggal" not in df.columns:
                raise ValueError("Kolom 'Harga' atau 'Tanggal' tidak ditemukan di file CSV.")

            # Ubah ke string, bersihkan karakter umum, hapus spasi tak terlihat
            df["Harga"] = df["Harga"].astype(str).str.replace("Rp", "", regex=False)
            # Hilangkan titik, koma, spasi, strip, dan karakter lain yang umum
            df["Harga"] = df["Harga"].str.replace(".", "", regex=False).str.replace(",", "", regex=False).str.replace("\u00A0", "", regex=True).str.replace("-", "", regex=False).str.strip()

            # Konversi aman ke numeric (nilai tidak valid -> NaN)
            df["Harga"] = pd.to_numeric(df["Harga"], errors="coerce")

            # Konversi Tanggal dengan coercion, hapus baris tanpa tanggal atau harga valid
            df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
            df = df.dropna(subset=["Harga", "Tanggal"]).copy()
            df.reset_index(drop=True, inplace=True)

            # Tambah fitur tanggal
            df["Hari"] = df["Tanggal"].dt.day
            df["Bulan"] = df["Tanggal"].dt.month
            df["Tahun"] = df["Tanggal"].dt.year
            df["HariKe"] = df["Tanggal"].dt.dayofyear
            df["HariMinggu"] = df["Tanggal"].dt.dayofweek

            return df

        try:
            df_train = clean_df(df_train)
            df_test  = clean_df(df_test)
        except Exception as e:
            st.error(f"Gagal membersihkan data: {e}")
            return np.empty((0, 5)), np.empty((0,)), np.empty((0, 5)), np.empty((0,)), None, None, pd.DataFrame(), []

        # === 4️⃣ Siapkan fitur dan target ===
        feature_cols = ['Hari', 'Bulan', 'Tahun', 'HariKe', 'HariMinggu']
        X_train = df_train[feature_cols].copy()
        y_train = df_train["Harga"].values.reshape(-1, 1)
        X_test  = df_test[feature_cols].copy()
        y_test  = df_test["Harga"].values.reshape(-1, 1)

        # === 5️⃣ Normalisasi (StandardScaler) ===
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        # fit_transform but tetap simpan scaler_y untuk inverse later
        y_train_scaled = scaler_y.fit_transform(y_train).ravel()

        # Kita juga sediakan X_test_scaled (opsional) jika mau digunakan nanti
        X_test_scaled = scaler_X.transform(X_test)

        return X_train_scaled, y_train_scaled, X_test_scaled, y_test, scaler_X, scaler_y, df_test, feature_cols
    X_train, y_train, X_test_scaled, y_test, scaler_X, scaler_y, df_test, feature_cols = load_data()


    # --- Pilih model
    regressor = st.radio("Pilih Metode Regresi:", ["Twin Support Vector Regression (TSVR)", "Support Vector Regression (SVR)", "Random Forest Regression (RFR)"], horizontal=True)

    # --- Pilihan kernel
    if regressor in ["Twin Support Vector Regression (TSVR)", "Support Vector Regression (SVR)"]:
        st.radio("Pilih Jenis Kernel:", ["RBF"], horizontal=True)

    # --- Skenario parameter
    param_grid_tsvr = [
        {"C1": 0.01, "C2": 0.01, "gamma": 0.001},
        {"C1": 0.1,  "C2": 1.0,  "gamma": 0.05},
        {"C1": 0.1, "C2": 0.1, "gamma": 0.01},
        {"C1": 0.1, "C2": 0.1, "gamma": 0.1},
        {"C1": 1.0, "C2": 1.0, "gamma": 0.1},
        {"C1": 1.0, "C2": 1.0, "gamma": 1.0},
        {"C1": 1.0, "C2": 10, "gamma": 0.1},
        {"C1": 10, "C2": 10, "gamma": 0.1},
        {"C1": 10, "C2": 10, "gamma": 1.0},
        {"C1": 10, "C2": 1.0, "gamma": 10},
        {"C1": 100,  "C2": 100,  "gamma": 0.01},
        {"C1": 100,  "C2": 10,   "gamma": 0.1},
    ]
    param_grid_svr = [
        {"C": 0.1, "epsilon": 0.1, "gamma": 0.01},
        {"C": 0.1, "epsilon": 0.1, "gamma": 0.1},
        {"C": 1.0, "epsilon": 0.1, "gamma": 0.1},
        {"C": 1.0, "epsilon": 0.1, "gamma": 1.0},
        {"C": 10,  "epsilon": 0.1, "gamma": 0.01},
        {"C": 10,  "epsilon": 0.1, "gamma": 0.1},
        {"C": 10,  "epsilon": 0.1, "gamma": 1.0},
        {"C": 10,  "epsilon": 0.1, "gamma": 10},
    ]
    rf_scenarios = [
        {"n_estimators": 50,  "max_depth": None, "random_state": 42},
        {"n_estimators": 100, "max_depth": None, "random_state": 42},
        {"n_estimators": 200, "max_depth": None, "random_state": 42},
        {"n_estimators": 50,  "max_depth": 5, "random_state": 42},
        {"n_estimators": 100, "max_depth": 5, "random_state": 42},
        {"n_estimators": 200, "max_depth": 5, "random_state": 42},
        {"n_estimators": 50,  "max_depth": 10, "random_state": 42},
        {"n_estimators": 100, "max_depth": 10, "random_state": 42},
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
