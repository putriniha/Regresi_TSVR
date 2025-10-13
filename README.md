# 🌾 Dashboard Prediksi Harga Jagung Menggunakan SVR, Random Forest, dan TwinSVR

Aplikasi ini dikembangkan menggunakan **Streamlit** untuk menampilkan hasil prediksi harga jagung berdasarkan model **Twin Support Vector Regression (TSVR)**, **Support Vector Regression (SVR)**, dan **Random Forest Regression (RFR)**.  
Tujuan utama aplikasi ini adalah untuk membantu analisis dan visualisasi performa model prediksi harga komoditas secara interaktif.

---

## 🚀 **Fitur Utama**
- 📈 Prediksi harga jagung menggunakan tiga metode regresi:
  - Twin Support Vector Regression (TSVR)
  - Support Vector Regression (SVR)
  - Random Forest Regression (RFR)
- 📊 Visualisasi hasil prediksi dan error (MAPE, MSE, SSE, R², dan Error Variance)
- ⚙️ Normalisasi data otomatis menggunakan `StandardScaler`
- 🖥️ Tampilan interaktif berbasis **Streamlit**

---

## 🧩 **Struktur Folder**
📂 streamlit-tsvr-app/
┣ 📜 app.py ← File utama aplikasi Streamlit
┣ 📜 requirements.txt ← Daftar library yang dibutuhkan
┣ 📜 tsvr.py ← Implementasi metode TwinSVR
┣ 📂 data/ ← Dataset
┗ 📜 README.md ← Dokumentasi proyek
