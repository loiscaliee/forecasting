import streamlit as st
import pickle
import numpy as np

# Load pkl
with open("best_model_AALI.pkl", "rb") as f:
    model_data = pickle.load(f)

st.title("📈 Forecasting Saham AALI")
st.write("Model info:", model_data)

# Input harga hari ini (kamu bisa ganti bebas)
harga_hari_ini = st.number_input("Harga Close Hari Ini", value=0.0)

if st.button("Predict"):
    if model_data["model_type"] == "ES":
        alpha = model_data["alpha"]
        last_value = model_data["last_metric_value"]

        pred = alpha * harga_hari_ini + (1 - alpha) * last_value

        st.success(f"Prediksi harga besok (ES): {pred:.2f}")

    elif model_data["model_type"] == "MA":
        window_data = np.array(model_data["last_data_window"])
        pred = window_data.mean()

        st.success(f"Prediksi harga besok (MA): {pred:.2f}")

    else:
        st.error("Model tidak dikenal!")
