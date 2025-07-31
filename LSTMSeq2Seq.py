# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model dan scaler
model = load_model("model_seq2seq.h5")
scaler = "scaler.joblib"

st.title("LSTM Seq2Seq Forecasting")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['ddate'] = pd.to_datetime(df['ddate'])

    # Preprocessing & scaling
    data = df['tag_value'].values[-60:]
    data = np.array(data).reshape(-1, 1)
    data_scaled = scaler.transform(data)
    data_scaled = data_scaled.reshape(1, 60, 1)

    # Prediksi
    decoder_input = np.zeros((1, 60, 1))  # decoder dummy
    prediction = model.predict([data_scaled, decoder_input])[0]

    # Inverse transform
    prediction_inv = scaler.inverse_transform(prediction)

    # Plot
    st.line_chart(prediction_inv)
