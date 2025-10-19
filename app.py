import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tsvr import TwinSVR
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler   

# =============================
# Tambahan CSS untuk tema pink + navbar
# =============================
def add_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #fff5f7; /* pale pink */
    }
    h1, h2, h3, h4 {
        color: #b03060; /* deep rose */
    }
    /* 🔧 NAVBAR */
    .navbar {
        display: flex;
        justify-content: center;
        background-color: #ffe6eb;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .navbar button {
        background-color: #f7c6d9;
        border: none;
        color: #b03060;
        padding: 10px 18px;
        margin: 0 6px;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: 0.3s;
    }
    .navbar button:hover {
        background-color: #f090b2;
        color: white;
    }
    .navbar .active {
        background-color: #b03060;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# Fungsi prediksi & evaluasi
# ==================================================
def predict_january_2025(model, X_columns, scaler_X, scaler_y, df_test):
    start_date = pd.to_datetime("2025-01-01")
    end_date   = pd.to_datetime("2025-01-07")
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
        for col in [col for col in X_columns if col.startswith('Pasar_')]:
            features[col] = 0

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

    df_pred = pd.DataFrame(predictions, columns=["Tanggal", "Prediksi", "Aktual"])
    return df_pred


def evaluate_forecast(model, feature_cols, scaler_X, scaler_y, df_test,
                      mape_val=None, sse_val=None, mse_val=None, r2_val=None):

    df_pred_jan = predict_january_2025(model, feature_cols, scaler_X, scaler_y, df_test)

    st.subheader("📅 Prediksi Januari 2025")
    st.dataframe(df_pred_jan)

    # --- Siapkan data aktual (df_eval)
    df_eval = df_pred_jan.dropna(subset=["Aktual"])
    if not df_eval.empty:
        # Hitung metrik evaluasi pada data aktual
        errors = df_eval["Aktual"] - df_eval["Prediksi"]
        mape = mean_absolute_percentage_error(df_eval["Aktual"], df_eval["Prediksi"]) * 100
        sse = np.sum(errors ** 2)
        mse = mean_squared_error(df_eval["Aktual"], df_eval["Prediksi"])
        r2 = r2_score(df_eval["Aktual"], df_eval["Prediksi"])
        error_variance = np.var(errors, ddof=1)  # Error variance

        st.subheader("🎯 Evaluasi Perbandingan Data Aktual dan Prediksi Januari 2025")
        st.markdown(f"""
        **MAPE**           : 🧮 {mape:.2f}%  
        **SSE**            : 🧮 {sse:.2f}  
        **MSE**            : 🧮 {mse:.3f}  
        **R²**             : 🧮 {r2:.3f}  
        **Error Variance** : 🧮 {error_variance:.3f}  
        """)

        # --- Visualisasi
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_pred_jan["Tanggal"], df_pred_jan["Prediksi"], label="Prediksi", marker="o", color="#505fd4")
        if df_pred_jan["Aktual"].notna().any():
            ax.plot(df_pred_jan["Tanggal"], df_pred_jan["Aktual"], label="Aktual", marker="x", color="#f95d6a")
        ax.set_title("Prediksi vs Aktual - Januari 2025", color="#3d0318")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("ℹ️ Tidak ada data aktual Januari 2025 di df_test.")


# ==================================================
# Main Function
# ==================================================
def main():
    st.set_page_config(page_title="📈 Dashboard Perbandingan Metode", layout="wide")
    add_custom_css()
    st.title("📑 Regression Web App")

    # 🔧 NAVBAR (gantikan sidebar)
    model_options = [
        "Twin Support Vector Regression (TSVR)",
        "Support Vector Regression (SVR)",
        "Random Forest Regression"
    ]
    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    cols = st.columns(len(model_options))
    selected_model = None
    for i, option in enumerate(model_options):
        if cols[i].button(option):
            selected_model = option
    st.markdown('</div>', unsafe_allow_html=True)

    # Jika belum ada yang dipilih
    if not selected_model:
        st.info("👆 Pilih salah satu model di navbar untuk memulai.")
        return

    # =============================
    # Load data
    # =============================
    @st.cache_data
    def load_data():
        local_train = r"D:\PUTRI\COOLYEAH\8~\BAB IV\streamlit\data\data_train.csv"
        local_test = r"D:\PUTRI\COOLYEAH\8~\BAB IV\streamlit\data\data_test.csv"
        repo_train = os.path.join("data", "data_train.csv")
        repo_test = os.path.join("data", "data_test.csv")

        if os.path.exists(local_train):
            df_train = pd.read_csv(local_train)
            df_test = pd.read_csv(local_test)
        else:
            df_train = pd.read_csv(repo_train)
            df_test = pd.read_csv(repo_test)

        drop_cols = ['Pasar Rongtengah', 'Pasar Srimangunan', ' Pasar Rongtengah', ' Pasar Srimangunan']
        df_train = df_train.drop(columns=drop_cols, errors="ignore").rename(columns={'Rata-rata': 'Harga'})
        df_test  = df_test.drop(columns=drop_cols, errors="ignore").rename(columns={' Rata-rata': 'Harga'})

        def clean_price(series):
            return (
                series.astype(str)
                .str.replace("Rp", "", regex=False)
                .str.replace(".", "", regex=False)
                .str.replace(",", "", regex=False)
                .astype(float)
            )

        df_train["Harga"] = clean_price(df_train["Harga"])
        df_test["Harga"]  = clean_price(df_test["Harga"])

        df_train['Tanggal'] = pd.to_datetime(df_train['Tanggal'])
        df_test['Tanggal']  = pd.to_datetime(df_test['Tanggal'])

        for df in [df_train, df_test]:
            df['Hari']       = df['Tanggal'].dt.day
            df['Bulan']      = df['Tanggal'].dt.month
            df['Tahun']      = df['Tanggal'].dt.year
            df['HariKe']     = df['Tanggal'].dt.dayofyear
            df['HariMinggu'] = df['Tanggal'].dt.dayofweek

        feature_cols = ['Hari', 'Bulan', 'Tahun', 'HariKe', 'HariMinggu'] + \
                       [col for col in df_train.columns if col.startswith('Pasar_')]

        X_train = df_train[feature_cols]
        y_train = df_train['Harga'].values.reshape(-1, 1)
        X_test = df_test[feature_cols]
        y_test = df_test['Harga'].values.reshape(-1, 1)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train).ravel()
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test).ravel()

        return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y, df_test, feature_cols

    X_train, y_train, X_test, y_test, scaler_X, scaler_y, df_test, feature_cols = load_data()

    # =============================
    # Pemilihan model
    # =============================
    if selected_model == "Support Vector Regression (SVR)":
        st.subheader("🔹 Support Vector Regression (SVR)")
        param = {"C": 1.0, "epsilon": 0.1, "gamma": 0.1}
        model = SVR(C=param["C"], epsilon=param["epsilon"], gamma=param["gamma"], kernel="rbf")
        if st.button("🚀 Train & Evaluate SVR"):
            model.fit(X_train, y_train)
            evaluate_forecast(model, feature_cols, scaler_X, scaler_y, df_test)

    elif selected_model == "Random Forest Regression":
        st.subheader("🔹 Random Forest Regression")
        param = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        model = RandomForestRegressor(**param)
        if st.button("🚀 Train & Evaluate RFR"):
            model.fit(X_train, y_train)
            evaluate_forecast(model, feature_cols, scaler_X, scaler_y, df_test)

    elif selected_model == "Twin Support Vector Regression (TSVR)":
        st.subheader("🔹 Twin Support Vector Regression (TSVR)")
        param = {"C1": 1.0, "C2": 1.0, "gamma": 0.1}
        model = TwinSVR(**param, kernel="rbf")
        if st.button("🚀 Train & Evaluate TSVR"):
            model.fit(X_train, y_train)
            evaluate_forecast(model, feature_cols, scaler_X, scaler_y, df_test)


if __name__ == '__main__':
    main()
