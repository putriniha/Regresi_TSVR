import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tsvr import TwinSVR
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# =============================
# Tampilan & CSS Tema Pale Orange + Hover Pink
# =============================
def add_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #fff3e0;
    }
    h1, h2, h3, h4 {
        color: #b74f1d;
    }
    .navbar {
        display: flex;
        justify-content: center;
        background-color: #ffe0b2;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 25px;
    }
    .navbar button {
        background-color: #ffd699;
        border: none;
        color: #b74f1d;
        padding: 10px 20px;
        margin: 0 6px;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: 0.3s;
    }
    .navbar button:hover {
        background-color: #f7c6d9;
        color: #b74f1d;
    }
    .navbar .active {
        background-color: #e67e22;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# Fungsi Prediksi & Evaluasi
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

    return pd.DataFrame(predictions, columns=["Tanggal", "Prediksi", "Aktual"])


def evaluate_model(model, feature_cols, scaler_X, scaler_y, df_test):
    df_pred = predict_january_2025(model, feature_cols, scaler_X, scaler_y, df_test)
    df_eval = df_pred.dropna(subset=["Aktual"])
    if df_eval.empty:
        return None
    errors = df_eval["Aktual"] - df_eval["Prediksi"]
    mape = mean_absolute_percentage_error(df_eval["Aktual"], df_eval["Prediksi"]) * 100
    mse = mean_squared_error(df_eval["Aktual"], df_eval["Prediksi"])
    r2 = r2_score(df_eval["Aktual"], df_eval["Prediksi"])
    sse = np.sum(errors ** 2)
    error_variance = np.var(errors, ddof=1)
    return {"MAPE": mape, "MSE": mse, "R2": r2, "SSE": sse, "Error Variance": error_variance, "Pred": df_pred}


# ==================================================
# Main Function
# ==================================================
def main():
    st.set_page_config(page_title="📈 Dashboard Perbandingan Metode", layout="wide")
    add_custom_css()
    st.title("📑 Regression Web App")

    # =============================
    # Navbar Horizontal
    # =============================
    model_options = [
        "Twin Support Vector Regression (TSVR)",
        "Support Vector Regression (SVR)",
        "Random Forest Regression"
    ]
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    cols = st.columns(len(model_options))
    for i, option in enumerate(model_options):
        if cols[i].button(option):
            st.session_state.selected_model = option
    st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state.selected_model:
        st.info("👆 Pilih salah satu model di navbar untuk memulai.")
        return

    # =============================
    # Load Data
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

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train).ravel()

        return X_train_scaled, y_train_scaled, scaler_X, scaler_y, df_test, feature_cols

    X_train, y_train, scaler_X, scaler_y, df_test, feature_cols = load_data()

    # =============================
    # Jalankan Semua Skenario
    # =============================
    selected_model = st.session_state.selected_model

    st.subheader(f"⚙️ Pengujian Skenario: {selected_model}")
    kernel_choice = None
    if selected_model in ["Support Vector Regression (SVR)", "Twin Support Vector Regression (TSVR)"]:
        kernel_choice = st.radio("Pilih Kernel:", ["rbf", "linear", "poly", "sigmoid"], index=0, horizontal=True)

    if st.button("🚀 Jalankan Semua Skenario"):
        results = []

        if selected_model == "Twin Support Vector Regression (TSVR)":
            param_grid = [
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
                {"C1": 100,  "C2": 10,   "gamma": 0.1}
            ]
            for params in param_grid:
                model = TwinSVR(**params, kernel=kernel_choice)
                model.fit(X_train, y_train)
                res = evaluate_model(model, feature_cols, scaler_X, scaler_y, df_test)
                if res:
                    res.update(params)
                    res["kernel"] = kernel_choice
                    results.append(res)

        elif selected_model == "Support Vector Regression (SVR)":
            scenarios = [
                {"C": 0.1, "epsilon": 0.1, "gamma": 0.01},
                {"C": 0.1, "epsilon": 0.1, "gamma": 0.1},
                {"C": 1.0, "epsilon": 0.1, "gamma": 0.1},
                {"C": 1.0, "epsilon": 0.1, "gamma": 1.0},
                {"C": 10,  "epsilon": 0.1, "gamma": 0.01},
                {"C": 10,  "epsilon": 0.1, "gamma": 0.1},
                {"C": 10,  "epsilon": 0.1, "gamma": 1.0},
                {"C": 10,  "epsilon": 0.1, "gamma": 10}
            ]
            for params in scenarios:
                model = SVR(**params, kernel=kernel_choice)
                model.fit(X_train, y_train)
                res = evaluate_model(model, feature_cols, scaler_X, scaler_y, df_test)
                if res:
                    res.update(params)
                    res["kernel"] = kernel_choice
                    results.append(res)

        else:  # Random Forest
            rf_scenarios = [
                {"n_estimators": 50,  "max_depth": None, "random_state": 42},
                {"n_estimators": 100, "max_depth": None, "random_state": 42},
                {"n_estimators": 200, "max_depth": None, "random_state": 42},
                {"n_estimators": 50,  "max_depth": 5, "random_state": 42},
                {"n_estimators": 100, "max_depth": 5, "random_state": 42},
                {"n_estimators": 200, "max_depth": 5, "random_state": 42},
                {"n_estimators": 50,  "max_depth": 10, "random_state": 42},
                {"n_estimators": 100, "max_depth": 10, "random_state": 42},
                {"n_estimators": 200, "max_depth": 10, "random_state": 42}
            ]
            for params in rf_scenarios:
                model = RandomForestRegressor(**params)
                model.fit(X_train, y_train)
                res = evaluate_model(model, feature_cols, scaler_X, scaler_y, df_test)
                if res:
                    res.update(params)
                    results.append(res)

        df_results = pd.DataFrame(results)
        df_results_sorted = df_results.sort_values("MAPE").reset_index(drop=True)
        st.subheader("📊 Hasil Evaluasi Semua Skenario")
        st.dataframe(df_results_sorted, use_container_width=True)

        best = df_results_sorted.iloc[0]
        st.success(f"🔥 Skenario Terbaik dengan MAPE {best['MAPE']:.2f}%")
        df_pred_best = best["Pred"]

        # Visualisasi hasil terbaik
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_pred_best["Tanggal"], df_pred_best["Prediksi"], label="Prediksi", marker="o", color="#e67e22")
        ax.plot(df_pred_best["Tanggal"], df_pred_best["Aktual"], label="Aktual", marker="x", color="#b74f1d")
        ax.set_title("Prediksi vs Aktual (Model Terbaik)")
        ax.legend()
        st.pyplot(fig)


if __name__ == '__main__':
    main()
