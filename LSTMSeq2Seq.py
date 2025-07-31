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

        # Normalisasi dengan scaler yang dimuat dari file
        data = np.array(data).reshape(-1, 1)
        data_scaled = scaler.transform(data)
        data_scaled = data_scaled.reshape(1, 60, 1)

        # Dummy decoder input
        decoder_input = np.zeros((1, 60, 1))

        # Prediksi
        prediction_scaled = model.predict([data_scaled, decoder_input])
        prediction = scaler.inverse_transform(prediction_scaled[0])

        # Buat rentang waktu untuk hasil prediksi
        last_ddate = df['ddate'].iloc[-1]
        time_interval = df['ddate'].diff().mode()[0] if df['ddate'].diff().mode().size > 0 else timedelta(seconds=10)
        future_dates = [last_ddate + (i+1)*time_interval for i in range(60)]
        pred_df = pd.DataFrame({"ddate": future_dates, "predicted_value": prediction.flatten()})

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi 60 Langkah ke Depan")
        fig, ax = plt.subplots()
        ax.plot(pred_df['ddate'], pred_df['predicted_value'], label='Predicted')
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Tag Value")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.write("\nPrediksi (terakhir 10 nilai):")
        st.dataframe(pred_df.tail(10))
