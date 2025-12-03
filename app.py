import streamlit as st
import pickle
import numpy as np

st.title("📈 IDX Forecasting — MA / ES Model")

# Load model dict
with open("best_model_AALI.pkl", "rb") as f:
    model_data = pickle.load(f)

st.subheader("🔍 Model Loaded dari .pkl")
st.json(model_data)

# Input harga terakhir (Close hari ini)
harga_hari_ini = st.number_input("Harga Close Hari Ini", value=0.0)

if st.button("PREDIKSI HARGA BESOK"):

    model_type = model_data["model_type"]

    # ============================
    #   MODEL EXPONENTIAL SMOOTHING
    # ============================
    if model_type == "ES":

        alpha = model_data["alpha"]
        last_value = model_data["last_metric_value"]

        # Preprocess yang sama dengan ipynb:
        # pred_next = α * harga_hari_ini + (1-α) * last_ewm_value
        prediksi = alpha * harga_hari_ini + (1 - alpha) * last_value

        st.success(f"📌 Model: Exponential Smoothing (ES)")
        st.write(f"Prediksi harga besok: **{prediksi:.2f}**")

    # ============================
    #   MODEL MOVING AVERAGE
    # ============================
    elif model_type == "MA":

        last_window = np.array(model_data["last_data_window"])
        window_size = model_data["window_size"]

        # Preprocess yang sama dengan ipynb:
        # pred_next = mean(last 20 data)
        prediksi = last_window.mean()

        st.success(f"📌 Model: Moving Average (MA)")
        st.write(f"Prediksi harga besok: **{prediksi:.2f}**")

    else:
        st.error("Model tidak dikenali.")
