import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load model dan scaler
model = load_model('my_model.keras')
scaler = joblib.load('scaler.joblib')

st.title("LSTM Time Series Forecasting")

# Upload file input
uploaded_file = st.file_uploader("Upload CSV file (data terbaru, minimal 60 data terakhir)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data yang Diupload")
    st.write(df.tail(10))

    # Cek minimal 60 data
    if df.shape[0] < 60:
        st.error("Data minimal harus memiliki 60 baris untuk prediksi.")
    else:
        # Ambil 60 data terakhir
        data_input = df.tail(60).values
        data_scaled = scaler.transform(data_input)

        # Reshape untuk model: (1, 60, n_features)
        data_reshaped = np.reshape(data_scaled, (1, data_scaled.shape[0], data_scaled.shape[1]))

        # Prediksi
        prediction_scaled = model.predict(data_reshaped)

        # Invers transform hasil prediksi
        prediction = scaler.inverse_transform(prediction_scaled)

        st.subheader("Hasil Prediksi")
        st.write(pd.DataFrame(prediction, columns=df.columns))

        # Download hasil prediksi
        csv = pd.DataFrame(prediction, columns=df.columns).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil Prediksi", csv, file_name="prediction.csv", mime='text/csv')
