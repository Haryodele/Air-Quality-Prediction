import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model using joblib
model = joblib.load("model.pkl")

# Define expected feature names (same order as during training!)
feature_names = ["Temperature", "Humidity", "Pressure", "Wind"]

# Title and description
st.title("🌍 Air Quality Prediction in Abuja")
st.write("Enter weather conditions to predict the Air Quality Index (AQI) category.")

# Sidebar info/help
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app predicts the **Air Quality Index (AQI)** in Abuja "
        "based on temperature, humidity, pressure, and wind speed."
    )
    st.write("Created with Streamlit + a trained ML model.")

# Input widgets
st.subheader("Input values (these must match the features used during training)")

temperature = st.number_input("🌡️ Temperature (°C)", 20.0, 45.0, 25.0)
st.caption("Typical Abuja range: ~20–35°C")

humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 50.0)
st.caption("Humidity as percentage (0–100).")

pressure = st.number_input("📊 Pressure (hPa)", 900.0, 1100.0, 1010.0)
st.caption("Typical sea-level pressure values around 1000–1020 hPa.")

wind = st.number_input("💨 Wind (m/s)", 0.0, 20.0, 3.0)
st.caption("Wind speed in m/s.")

# Example presets
st.write("")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Use example: Typical day"):
        st.session_state["Temperature"] = 30.0
        st.session_state["Humidity"] = 45.0
        st.session_state["Pressure"] = 1012.0
        st.session_state["Wind"] = 2.0
        st.experimental_rerun()
with col2:
    if st.button("Use example: Rainy day"):
        st.session_state["Temperature"] = 24.0
        st.session_state["Humidity"] = 85.0
        st.session_state["Pressure"] = 1008.0
        st.session_state["Wind"] = 5.0
        st.experimental_rerun()
with col3:
    if st.button("Use example: Dusty day"):
        st.session_state["Temperature"] = 35.0
        st.session_state["Humidity"] = 20.0
        st.session_state["Pressure"] = 1005.0
        st.session_state["Wind"] = 8.0
        st.experimental_rerun()

# Prepare input for prediction
input_data = pd.DataFrame(
    [[temperature, humidity, pressure, wind]],
    columns=feature_names
)

# Predict button
if st.button("🔮 Predict AQI"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"✅ Predicted AQI Category: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
