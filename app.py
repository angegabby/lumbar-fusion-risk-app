import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Lumbar Fusion Reoperation Risk Predictor")
st.markdown("**MIMIC-IV AutoPrognosis • AUC 0.95**")

model = joblib.load('lumbar_model.pkl')

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 20, 90, 50)
    race_white = st.checkbox("White Race")
    insurance_private = st.checkbox("Private Insurance")
    discharge_home = st.checkbox("Discharge Home")
    los_days = st.number_input("Length of Stay (days)", 1, 30, 5)
    fusion_levels = st.number_input("Fusion Levels", 1, 5, 1)
with col2:
    charlson_score = st.number_input("Charlson Score", 0, 10, 0)
    chf = st.checkbox("Congestive Heart Failure")
    smoking = st.checkbox("Smoking")
    obesity = st.checkbox("Obesity")
    icu_stay = st.checkbox("ICU Stay")
    steroid_use = st.checkbox("Steroid Use")
    ssi = st.checkbox("Surgical Site Infection")

if st.button("Predict Risk"):
    data = pd.DataFrame({
        'age': [(age - 50)/15],
        'race_white': [int(race_white)],
        'insurance_private': [int(insurance_private)],
        'discharge_home': [int(discharge_home)],
        'los_days': [(los_days - 5)/3],
        'fusion_levels': [fusion_levels],
        'anterior_approach': [0],
        'charlson_score': [charlson_score],
        'chf': [int(chf)],
        'smoking': [int(smoking)],
        'obesity': [int(obesity)],
        'icu_stay': [int(icu_stay)],
        'steroid_use': [int(steroid_use)],
        'ssi': [int(ssi)]
    })
    model_features = model.feature_names if hasattr(model, 'feature_names') else data.columns
    data = data[[f for f in model_features if f in data.columns]]
    probs = model.predict_proba(data)
    if isinstance(probs, pd.DataFrame):
        risk = probs.iloc[:, 1][0] * 100
    else:
        risk = probs[:, 1][0] * 100
    st.metric("Reoperation Risk", f"{risk:.1f}%")
    if risk > 20:
        st.error("High Risk: Surgical consult recommended")
    elif risk > 10:
        st.warning("Moderate Risk: Close monitoring")
    else:
        st.success("Low Risk: Standard follow-up")
        
