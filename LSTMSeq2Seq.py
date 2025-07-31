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

# Upload data CSV
uploaded_file = st.file_uploader("📤 Upload CSV berisi data sensor", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Pastikan ada kolom 'tag_value'
    if 'tag_value' not in df.columns:
        st.error("File CSV harus memiliki kolom bernama 'tag_value'.")
    else:
        st.subheader("📄 Data Input (10 Terakhir)")
        st.write(df.tail(10))

        # Ambil 60 data terakhir
        if len(df) < 60:
            st.error("Data harus minimal memiliki 60 baris.")
        else:
            last_60 = df['tag_value'].values[-60:].reshape(-1, 1)
            last_60_scaled = scaler.transform(last_60)
            last_60_scaled = np.reshape(last_60_scaled, (1, 60, 1))

            # Prediksi 60 langkah ke depan
            prediction_scaled = model.predict(last_60_scaled)
            prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

            st.subheader("📈 Hasil Prediksi (60 langkah ke depan)")
            st.line_chart(prediction)

            # Download hasil prediksi
            pred_df = pd.DataFrame(prediction, columns=["forecast_tag_value"])
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Hasil Prediksi", csv, file_name="forecast_result.csv", mime="text/csv")
