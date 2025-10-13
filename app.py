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
# Tambahan CSS untuk tema pink
# =============================
def add_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #fff5f7; /* pale pink */
    }
    .stSidebar {
        background-color: #ffe6eb; /* sidebar pink */
    }
    h1, h2, h3, h4 {
        color: #b03060; /* deep rose */
    }
    .css-1d391kg {  /* Sidebar title */
        color: #b03060 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    add_custom_css()

    st.set_page_config(page_title="📈 Dashboard Perbandingan Metode", layout="wide")
    st.title("📑 Regression Web App")
    st.sidebar.title("🤖 Model Selection")

    st.markdown("Pilih model di sidebar dan tekan **Train & Evaluate** untuk melihat hasil 📊.")

    @st.cache_data
    def load_data():
        df_train = pd.read_csv("data/data_train.csv")
        df_test = pd.read_csv(r"data/data_test.csv")

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

        if "Rata-rata" in df_train.columns:
            df_train = df_train.rename(columns={"Rata-rata": "Harga"})
            df_test  = df_test.rename(columns={"Rata-rata": "Harga"})

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
    
    st.sidebar.subheader("⚙️ Pilih Model")
    regressor = st.sidebar.selectbox("📌 Regressor", (
        "Twin Support Vector Regression (TSVR)", 
        "Support Vector Regression (SVR)", 
        "Random Forest Regression"))

    # ===============================================
    # SVR
    # ===============================================
    if regressor == "Support Vector Regression (SVR)":
    # Daftar kombinasi parameter
        param_combinations = [
            {"C": 0.1, "epsilon": 0.1, "gamma": 0.01},
            {"C": 0.1, "epsilon": 0.1, "gamma": 0.1},
            {"C": 1.0, "epsilon": 0.1, "gamma": 0.1},
            {"C": 1.0, "epsilon": 0.1, "gamma": 1.0},
            {"C": 10,  "epsilon": 0.1, "gamma": 0.01},
            {"C": 10,  "epsilon": 0.1, "gamma": 0.1},
            {"C": 10,  "epsilon": 0.1, "gamma": 1.0},
            {"C": 10,  "epsilon": 0.1, "gamma": 10},
        ]

        # Pilihan user di sidebar
        selected_params = st.sidebar.selectbox(
            "🚦 Pilih kombinasi parameter",
            param_combinations,
            format_func=lambda x: f"C={x['C']}, ε={x['epsilon']}, γ={x['gamma']}"
        )
        # Pilih kernel
        kernel = st.sidebar.radio(
        "🎯 Kernel",
        ("rbf"), index=0
        )
        
        if st.sidebar.button("🚀 Train & Evaluate"):
            # Buat model dengan parameter terpilih
            model = SVR(
                C=selected_params["C"],
                kernel="rbf",  # tetap pakai rbf
                gamma=selected_params["gamma"],
                epsilon=selected_params["epsilon"]
            )
            model.fit(X_train, y_train)

            st.subheader("📊 SVR Results")
            # st.subheader("🔹 Hasil Evaluasi Parameter SVR")
            # svr_results = evaluate_hyperparams_cv(SVR, param_combinations, scaler_X, scaler_y, model_type="SVR")
            # st.dataframe(svr_results)
            evaluate_forecast(model, feature_cols, scaler_X, scaler_y, df_test)


    # ===============================================
    # Random Forest
    # ===============================================
    elif regressor == "Random Forest Regression":
    # ============================================
    # 1. Daftar kombinasi parameter
    # ============================================
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

        # ============================================
        # 2. Pilihan parameter di sidebar
        # ============================================
        selected_params = st.sidebar.selectbox(
            "🚦 Pilih kombinasi parameter",
            rf_scenarios,
            format_func=lambda x: f"Trees={x['n_estimators']}, Depth={x['max_depth']}, rs={x['random_state']}"
        )

        # ============================================
        # 3. Train dan evaluasi model
        # ============================================
        if st.sidebar.button("🚀 Train & Evaluate"):
            # Buat model dengan parameter terpilih
            model = RandomForestRegressor(
                n_estimators=selected_params["n_estimators"],
                max_depth=selected_params["max_depth"],
                random_state=selected_params["random_state"]
            )

            # Training
            model.fit(X_train, y_train)

            # ============================================
            # 4. Evaluasi hasil
            # ============================================
            st.subheader("📊 Random Forest Regression Results")

            # Jika ingin menampilkan tabel hasil cross-validation:
            # rfr_results = evaluate_hyperparams_cv(
            #     RandomForestRegressor, rf_scenarios, scaler_X, scaler_y, model_type="RFR"
            # )
            # st.dataframe(rfr_results)

            # Evaluasi prediksi terhadap data uji
            evaluate_forecast(model, feature_cols, scaler_X, scaler_y, df_test)


    # ===============================================
    # TSVR
    # ===============================================
    elif regressor == "Twin Support Vector Regression (TSVR)":
    # Daftar kombinasi parameter
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

        # Pilihan kombinasi
        selected_params = st.sidebar.selectbox(
            "🚦 Pilih kombinasi parameter",
            param_grid,
            format_func=lambda x: f"C1={x['C1']}, C2={x['C2']}, γ={x['gamma']}"
        )

        # Pilih kernel
        kernel = st.sidebar.radio(
            "🎯 Kernel",
            ("rbf"), index=0
        )

        if st.sidebar.button("🚀 Train & Evaluate"):
            model = TwinSVR(
                C1=selected_params["C1"],
                C2=selected_params["C2"],
                kernel=kernel,
                gamma=selected_params["gamma"]
            )
            model.fit(X_train, y_train)

            st.subheader("📊 TSVR Results")
            # st.subheader("🔹 Hasil Evaluasi Twin SVR (TSVR)")
            # tsvr_results = evaluate_hyperparams_cv(TwinSVR, param_grid, scaler_X, scaler_y, model_type="TSVR")
            # st.dataframe(tsvr_results)
            evaluate_forecast(model, feature_cols, scaler_X, scaler_y, df_test)



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

if __name__ == '__main__':
    main()
