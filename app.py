import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from tsvr import TwinSVR  # pastikan file tsvr.py ada di folder yang sama


# =====================================================
# 🔹 FUNGSI PREDIKSI JANUARI 2025
# =====================================================
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


# =====================================================
# 🔹 FUNGSI EVALUASI (VERSI BARU)
# =====================================================
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

        # --- Evaluasi hasil prediksi aktual
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


# =====================================================
# 🔹 DATA SIMULASI (GANTI DENGAN DATA ASLI)
# =====================================================
st.title("📈 Prediksi Harga Konsumen Jagung")

np.random.seed(42)
tanggal = pd.date_range(start="2024-01-01", periods=30, freq="M")
harga = np.random.randint(6000, 9000, size=30)
df = pd.DataFrame({
    "Tanggal": tanggal,
    "Harga": harga,
    "Hari": tanggal.day,
    "Bulan": tanggal.month,
    "Tahun": tanggal.year,
    "HariKe": tanggal.dayofyear,
    "HariMinggu": tanggal.weekday
})

feature_cols = ["Hari", "Bulan", "Tahun", "HariKe", "HariMinggu"]
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(df[feature_cols])
y = scaler_y.fit_transform(df[["Harga"]])

train_size = int(0.8 * len(df))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
df_train, df_test = df.iloc[:train_size], df.iloc[train_size:]


# =====================================================
# 🔹 SIDEBAR MODEL
# =====================================================
st.sidebar.title("⚙️ Pengaturan Model")
regressor = st.sidebar.radio(
    "Pilih Metode Regresi",
    ("Twin Support Vector Regression (TSVR)", "Support Vector Regression (SVR)", "Random Forest Regression")
)


# =====================================================
# 🔹 TWIN SUPPORT VECTOR REGRESSION
# =====================================================
if regressor == "Twin Support Vector Regression (TSVR)":
    tsvr_params = [
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

    selected_params = st.sidebar.selectbox(
        "🚦 Pilih kombinasi parameter TSVR",
        tsvr_params,
        format_func=lambda x: f"C1={x['C1']}, C2={x['C2']}, γ={x['gamma']}"
    )

    kernel = st.sidebar.radio("🎯 Kernel", ("rbf",), index=0)

    if st.sidebar.button("🚀 Jalankan TSVR"):
        model = TwinSVR(
            C1=selected_params["C1"],
            C2=selected_params["C2"],
            kernel=kernel,
            gamma=selected_params["gamma"]
        )
        model.fit(X_train, y_train.ravel())
        st.subheader("📊 Hasil Twin SVR")
        evaluate_forecast(model, feature_cols, scaler_X, scaler_y, df_test)


# =====================================================
# 🔹 SUPPORT VECTOR REGRESSION
# =====================================================
elif regressor == "Support Vector Regression (SVR)":
    svr_params = [
        {"C": 0.1, "epsilon": 0.1, "gamma": 0.01},
        {"C": 0.1, "epsilon": 0.1, "gamma": 0.1},
        {"C": 1.0, "epsilon": 0.1, "gamma": 0.1},
        {"C": 1.0, "epsilon": 0.1, "gamma": 1.0},
        {"C": 10,  "epsilon": 0.1, "gamma": 0.01},
        {"C": 10,  "epsilon": 0.1, "gamma": 0.1},
        {"C": 10,  "epsilon": 0.1, "gamma": 1.0},
        {"C": 10,  "epsilon": 0.1, "gamma": 10},
    ]

    selected_params = st.sidebar.selectbox(
        "🚦 Pilih kombinasi parameter SVR",
        svr_params,
        format_func=lambda x: f"C={x['C']}, ε={x['epsilon']}, γ={x['gamma']}"
    )

    kernel = st.sidebar.radio("🎯 Kernel", ("rbf",), index=0)

    if st.sidebar.button("🚀 Jalankan SVR"):
        model = SVR(
            C=selected_params["C"],
            kernel=kernel,
            gamma=selected_params["gamma"],
            epsilon=selected_params["epsilon"]
        )
        model.fit(X_train, y_train.ravel())
        st.subheader("📊 Hasil SVR")
        evaluate_forecast(model, feature_cols, scaler_X, scaler_y, df_test)


# =====================================================
# 🔹 RANDOM FOREST REGRESSION
# =====================================================
elif regressor == "Random Forest Regression":
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

    selected_rf = st.sidebar.selectbox(
        "🌲 Pilih kombinasi parameter RF",
        rf_scenarios,
        format_func=lambda x: f"n={x['n_estimators']}, depth={x['max_depth']}"
    )

    if st.sidebar.button("🚀 Jalankan RF"):
        model = RandomForestRegressor(
            n_estimators=selected_rf["n_estimators"],
            max_depth=selected_rf["max_depth"],
            random_state=selected_rf["random_state"]
        )
        model.fit(X_train, y_train.ravel())
        st.subheader("📊 Hasil Random Forest")
        evaluate_forecast(model, feature_cols, scaler_X, scaler_y, df_test)
