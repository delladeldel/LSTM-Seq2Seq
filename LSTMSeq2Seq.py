import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load model dan scaler
MODEL_PATH = "my_model.keras"
model = load_model(MODEL_PATH)

st.title("LSTM Seq2Seq - Time Series Forecasting")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'ddate' not in df.columns or 'tag_value' not in df.columns:
        st.error("File harus memiliki kolom 'ddate' dan 'tag_value'")
    else:
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate')

        # Tampilkan data mentah
        st.subheader("Preview Data")
        st.dataframe(df.tail(10))

        # Ambil 60 data terakhir
        data = df['tag_value'].values[-60:]

        # Normalisasi
        scaler = MinMaxScaler()
        # Fit menggunakan seluruh data (atau bisa pakai training data yang sama dengan training model)
        scaler.fit(df['tag_value'].values.reshape(-1, 1))

        data = np.array(data).reshape(-1, 1)
        data_scaled = scaler.transform(data)
        data_scaled = data_scaled.reshape(1, 60, 1)

        # Dummy decoder input
        decoder_input = np.zeros((1, 60, 1))

        # Prediksi
        prediction_scaled = model.predict([data_scaled, decoder_input])
        prediction = scaler.inverse_transform(prediction_scaled[0])

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi 60 Langkah ke Depan")
        fig, ax = plt.subplots()
        ax.plot(range(1, 61), prediction, label='Predicted')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Tag Value")
        ax.legend()
        st.pyplot(fig)

        st.write("\nPrediksi (terakhir 10 nilai):")
        st.write(prediction[-10:])
