import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

st.title("🔮 LSTM Encoder-Decoder Forecasting (Seq2Seq)")

# Load model dan scaler
model = load_model("my_model.keras")
scaler = joblib.load("scaler.joblib")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload CSV berisi kolom 'tag_value'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'tag_value' not in df.columns:
        st.error("❌ File harus memiliki kolom 'tag_value'")
    elif len(df) < 60:
        st.error("❌ Data harus memiliki minimal 60 baris")
    else:
        st.subheader("📄 Data Input (10 terakhir)")
        st.write(df.tail(10))

        # Ambil 60 data terakhir dan normalisasi
        last_60 = df['tag_value'].values[-60:].reshape(-1, 1)
        last_60_scaled = scaler.transform(last_60)
        encoder_input = np.reshape(last_60_scaled, (1, 60, 1))

        # Buat decoder input (semua nol)
        decoder_input = np.zeros((1, 60, 1))

        # Prediksi
        prediction_scaled = model.predict([encoder_input, decoder_input])
        prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

        st.subheader("📈 Hasil Prediksi 60 langkah ke depan")
        st.line_chart(prediction)

        # Tampilkan DataFrame
        pred_df = pd.DataFrame(prediction, columns=["forecast_tag_value"])
        st.write(pred_df)

        # Tombol download
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil Prediksi", csv, file_name="forecast_result.csv", mime="text/csv")
