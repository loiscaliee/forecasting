import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

st.title("📈 Forecasting Saham AALI")

# --- Load Model ---
@st.cache_resource
def load_model():
    with open("best_model_AALI.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- Input Section ---
st.subheader("Input Data untuk Forecasting")

# Contoh input: misalnya model pakai fitur numerik tertentu
open_price = st.number_input("Open Price", min_value=0.0, value=1000.0)
high_price = st.number_input("High Price", min_value=0.0, value=1100.0)
low_price = st.number_input("Low Price", min_value=0.0, value=900.0)
volume = st.number_input("Volume", min_value=0.0, value=500000.0)

if st.button("Predict"):
    try:
        # Buat dataframe sesuai input model
        X = pd.DataFrame({
            "Open": [open_price],
            "High": [high_price],
            "Low": [low_price],
            "Volume": [volume]
        })

        # Forecast
        prediction = model.predict(X)

        st.success(f"📊 **Prediksi Harga Close:** {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Model error: {e}")

st.info("Pastikan fitur input sesuai dengan fitur yang dipakai model di Jupyter Notebook-mu.")
