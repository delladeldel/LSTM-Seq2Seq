import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import timedelta
import joblib

# Load model dan scaler
MODEL_PATH = "my_model.keras"
SCALER_PATH = "scaler.joblib"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

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
        last_timestamp = df['ddate'].values[-1]

        # Normalisasi dan reshape
        data = np.array(data).reshape(-1, 1)
        data_scaled = scaler.transform(data)
        data_scaled = data_scaled.reshape(1, 60, 1)

        # Decoder input: autoregressive (mulai dari nol, isi dengan prediksi sebelumnya)
        decoder_input = np.zeros((1, 60, 1))
        predictions_scaled = []

        for i in range(60):
            pred_scaled = model.predict([data_scaled, decoder_input], verbose=0)
            pred_value = pred_scaled[0, i, 0]
            predictions_scaled.append(pred_value)
            if i < 59:
                decoder_input[0, i + 1, 0] = pred_value

        # Inverse transform hasil prediksi
        prediction = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

        # Buat rentang waktu untuk hasil prediksi
        last_ddate = df['ddate'].iloc[-1]
        time_interval = df['ddate'].diff().mode()[0] if df['ddate'].diff().mode().size > 0 else timedelta(seconds=10)
        future_dates = [last_ddate + (i+1)*time_interval for i in range(60)]
        pred_df = pd.DataFrame({"ddate": future_dates, "predicted_value": prediction.flatten()})

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi 60 Langkah ke Depan")
        fig, ax = plt.subplots()
        ax.plot(df['ddate'].iloc[-200:], df['tag_value'].iloc[-200:], label='Data Historis')
        ax.plot(pred_df['ddate'], pred_df['predicted_value'], label='Predicted', color='red')
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Tag Value")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.write("\nPrediksi (terakhir 10 nilai):")
        st.dataframe(pred_df.tail(10))
