import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Load the trained model
# -------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Handle feature names robustly
# -------------------------------
try:
    if hasattr(model, "feature_names_in_"):
        feature_names = model.feature_names_in_
    elif hasattr(model, "feature_names"):
        feature_names = model.feature_names
    else:
        feature_names = None
except Exception:
    feature_names = None

# Final fallback: default feature names
if feature_names is None:
    feature_names = np.array(["temperature", "humidity", "pressure", "wind_speed"])

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Air Quality Prediction in Abuja", page_icon="🌍")

st.title("🌍 Air Quality Prediction in Abuja")
st.write("Enter weather conditions to predict the Air Quality Index (AQI) category.")

st.markdown("### Input values (these must match the features used during training)")

# Use Streamlit session_state to store inputs
if "temperature" not in st.session_state:
    st.session_state.temperature = 25.0
if "humidity" not in st.session_state:
    st.session_state.humidity = 50.0
if "pressure" not in st.session_state:
    st.session_state.pressure = 1010.0
if "wind_speed" not in st.session_state:
    st.session_state.wind_speed = 3.0

# Input sliders
st.session_state.temperature = st.number_input("🌡️ Temperature (°C)", value=st.session_state.temperature, step=0.5)
st.caption("Typical Abuja range: ~20–35°C")

st.session_state.humidity = st.number_input("💧 Humidity (%)", value=st.session_state.humidity, step=1.0)
st.caption("Humidity as percentage (0–100).")

st.session_state.pressure = st.number_input("📊 Pressure (hPa)", value=st.session_state.pressure, step=1.0)
st.caption("Typical sea-level pressure values around 1000–1020 hPa.")

st.session_state.wind_speed = st.number_input("💨 Wind (m/s)", value=st.session_state.wind_speed, step=0.5)
st.caption("Wind speed in m/s.")

# -------------------------------
# Preset Buttons
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Use example: Typical day"):
        st.session_state.temperature = 30.0
        st.session_state.humidity = 45.0
        st.session_state.pressure = 1012.0
        st.session_state.wind_speed = 2.0
        st.rerun()

with col2:
    if st.button("Use example: Rainy day"):
        st.session_state.temperature = 24.0
        st.session_state.humidity = 80.0
        st.session_state.pressure = 1008.0
        st.session_state.wind_speed = 4.0
        st.rerun()

with col3:
    if st.button("Use example: Dusty day"):
        st.session_state.temperature = 32.0
        st.session_state.humidity = 20.0
        st.session_state.pressure = 1005.0
        st.session_state.wind_speed = 1.5
        st.rerun()

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Air Quality"):
    input_data = np.array([
        st.session_state.temperature,
        st.session_state.humidity,
        st.session_state.pressure,
        st.session_state.wind_speed
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(input_data))
    else:
        confidence = 1.0

    st.subheader("Prediction Result:")
    categories = {
        0: ("Good 🙂", "Air quality is satisfactory, and air pollution poses little or no risk."),
        1: ("Moderate 😐", "Air quality is acceptable; however, some pollutants may be a concern for sensitive individuals."),
        2: ("Unhealthy for Sensitive Groups 😷", "Members of sensitive groups may experience health effects."),
        3: ("Unhealthy 🤒", "Everyone may begin to experience adverse health effects."),
        4: ("Very Unhealthy ☠️", "Avoid outdoor activities; sensitive individuals should stay indoors."),
        5: ("Hazardous 💀", "Health alert: everyone may experience serious health effects.")
    }

    category, description = categories.get(prediction, ("Unknown", "No description available."))
    st.markdown(
        f"### {category}\n{description}\n\n**Confidence:** {confidence:.2f}"
    )
