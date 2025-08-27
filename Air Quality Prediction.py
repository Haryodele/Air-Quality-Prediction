# --- assume model & feature_names already loaded above this block ---
import streamlit as st
import pandas as pd
import numpy as np

# fallback feature names (if model.feature_names_in_ not present)
feature_names = getattr(model, "feature_names_in_", None)
if feature_names is None:
    feature_names = np.array(["temperature", "humidity", "pressure", "wind_speed"])

# 1) Initialize session state keys with default values (run once)
defaults = {"temperature": 25.0, "humidity": 50.0, "pressure": 1010.0, "wind_speed": 3.0}
for fn in feature_names:
    if fn not in st.session_state:
        st.session_state[fn] = defaults.get(fn, 0.0)

st.markdown("**Input values (these must match the features used during training)**")

# 2) Preset buttons outside the form (set session_state, then rerun to refresh inputs)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Use example: Typical day"):
        st.session_state.update({"temperature": 30.0, "humidity": 70.0, "pressure": 1010.0, "wind_speed": 2.0})
        st.experimental_rerun()
with col2:
    if st.button("Use example: Rainy day"):
        st.session_state.update({"temperature": 24.0, "humidity": 90.0, "pressure": 1008.0, "wind_speed": 1.0})
        st.experimental_rerun()
with col3:
    if st.button("Use example: Dusty day"):
        st.session_state.update({"temperature": 35.0, "humidity": 20.0, "pressure": 1005.0, "wind_speed": 5.0})
        st.experimental_rerun()

# 3) Show inputs in a form (they are bound to st.session_state keys via key=)
with st.form(key="input_form"):
    inputs = {}
    for fname in feature_names:
        label = fname.replace("_", " ").capitalize()
        # number_input bound to the session_state key
        inputs[fname] = st.number_input(label, value=float(st.session_state[fname]), key=fname)
    submit = st.form_submit_button("Predict Air Quality")

# Build input_df as before
input_df = pd.DataFrame([{fn: st.session_state[fn] for fn in feature_names}])
if submit:
    # run prediction...
    pred = model.predict(input_df)
    ...
