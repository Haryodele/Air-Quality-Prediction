import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Air Quality Predictor", page_icon="🌍", layout="centered")
st.title("🌍 Air Quality Prediction in Abuja")
st.write("Enter weather conditions to predict the Air Quality Index (AQI) category.")

MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Put model.pkl in the same folder as this script.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_names = getattr(model, "feature_names_in_", None)
if feature_names is None:
    feature_names = np.array(["temperature", "humidity", "pressure", "wind_speed"])

st.markdown("**Input values (these must match the features used during training)**")

# Container for inputs so we can update them via presets
with st.form(key="input_form"):
    inputs = {}
    for fname in feature_names:
        label = fname.replace("_", " ").capitalize()
        if "temp" in fname.lower():
            inputs[fname] = st.number_input(f"🌡️ {label} (°C)", value=25.0, step=0.1, key=fname)
            st.caption("Typical Abuja range: ~20–35°C")
        elif "humid" in fname.lower():
            inputs[fname] = st.number_input(f"💧 {label} (%)", value=50.0, step=0.1, key=fname)
            st.caption("Humidity as percentage (0–100).")
        elif "press" in fname.lower():
            inputs[fname] = st.number_input(f"📊 {label} (hPa)", value=1010.0, step=0.1, key=fname)
            st.caption("Typical sea-level pressure values around 1000–1020 hPa.")
        elif "wind" in fname.lower():
            inputs[fname] = st.number_input(f"🌬️ {label} (m/s)", value=3.0, step=0.1, key=fname)
            st.caption("Wind speed in m/s.")
        else:
            inputs[fname] = st.number_input(f"{label}", value=0.0, step=0.1, key=fname)

    # Preset buttons (example scenarios)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.form_submit_button("Use example: Typical day"):
            # set values — these keys match the inputs so Streamlit updates them
            st.session_state[feature_names[0]] = 30.0 if "temp" in feature_names[0].lower() else st.session_state.get(feature_names[0], 25.0)
            # fill humidity, pressure, wind if named
            for fn, val in zip(feature_names, [70.0, 1010.0, 2.0]):
                if fn in st.session_state:
                    st.session_state[fn] = val
    with col2:
        if st.form_submit_button("Use example: Rainy day"):
            for fn, val in zip(feature_names, [24.0, 90.0, 1008.0, 1.0]):
                if fn in st.session_state:
                    st.session_state[fn] = val
    with col3:
        if st.form_submit_button("Use example: Dusty day"):
            for fn, val in zip(feature_names, [35.0, 20.0, 1005.0, 5.0]):
                if fn in st.session_state:
                    st.session_state[fn] = val

    # The actual predict button within the form
    submit = st.form_submit_button("Predict Air Quality")

# Build DataFrame now (reads current session_state values where relevant)
input_vals = {fn: st.session_state.get(fn, inputs.get(fn, 0.0)) for fn in feature_names}
input_df = pd.DataFrame([input_vals], columns=feature_names)

if submit:
    try:
        pred = model.predict(input_df)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    raw_pred = pred[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(input_df)
            confidence = float(np.max(probs))
        except Exception:
            confidence = None

    label = str(raw_pred)
    advice = {
        "Good": ("Good 🌱 — Air quality is safe. Enjoy outdoor activities.", "success"),
        "Moderate": ("Moderate 🙂 — Acceptable, but sensitive people should take caution.", "info"),
        "Unhealthy_Sensitive": ("Unhealthy for Sensitive Groups 😷 — Children, elderly or asthmatics should limit outdoor time.", "warning"),
        "Unhealthy": ("Unhealthy 🚨 — Many people may experience health effects; limit outdoor activities.", "warning"),
        "Very_Unhealthy": ("Very Unhealthy ☠️ — Avoid outdoor activities; sensitive individuals should stay indoors.", "error"),
    }
    message, kind = advice.get(label, (f"{label}", "info"))
    msg = message if confidence is None else f"{message}  —  confidence {confidence:.2f}"

    st.subheader("Prediction Result:")
    if kind == "success":
        st.success(msg)
    elif kind == "info":
        st.info(msg)
    elif kind == "warning":
        st.warning(msg)
    elif kind == "error":
        st.error(msg)
    else:
        st.write(msg)

    # Provide CSV download of the input + prediction
    out_df = input_df.copy()
    out_df["prediction"] = label
    if confidence is not None:
        out_df["confidence"] = confidence

    csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download prediction (CSV)", csv, file_name="aqi_prediction.csv", mime="text/csv")

    # Optional debug expander
    with st.expander("Debug (raw model outputs)"):
        st.write("raw predict():", raw_pred)
        if hasattr(model, "classes_"):
            st.write("model.classes_:", model.classes_)
        if confidence is not None:
            st.write("predict_proba:", probs)
        st.write("Input dataframe:", input_df)

# Short user guide and privacy note
st.markdown("---")
with st.expander("How to use (quick)"):
    st.write("""
    1. Enter or pick a preset for Temperature, Humidity, Pressure and Wind.
    2. Click **Predict Air Quality**.
    3. Read the color-coded advice box (green → safe, red → avoid outside).
    4. Use the Download button to save the input + prediction as a CSV file.
    """)

st.caption("This tool gives guidance only — it is not an official health advisory. For medical advice contact a professional.")
