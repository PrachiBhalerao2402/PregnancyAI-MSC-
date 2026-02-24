"""Streamlit app for pregnancy risk prediction."""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pregnancy Risk Predictor", page_icon="🩺", layout="centered")

st.markdown(
    """
    <style>
      .stApp {background: linear-gradient(180deg,#f4fbff 0%, #eef7ff 100%);} 
      .form-wrap {background: white; border-radius: 16px; padding: 1.25rem 1.25rem 0.75rem 1.25rem;
                  box-shadow: 0 6px 24px rgba(40,78,120,0.08); border: 1px solid #dceaf8;}
      h1, h2, h3 {color: #1f4b7a !important;}
      .small-muted {color:#5a7288; font-size:0.9rem; margin-top:-0.5rem; margin-bottom:1rem;}
      .stButton>button {width:100%; border-radius:10px; padding:0.55rem 0.9rem; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤰 Pregnancy Risk Prediction System")
st.caption("Machine Learning powered screening for Low Risk vs Medium Risk assessment")

try:
    artifact = joblib.load("models/best_pregnancy_risk_model.joblib")
except FileNotFoundError:
    st.error("Model file not found. Run `python train_models.py` first to generate the model artifact.")
    st.stop()

model = artifact["model"]
scaler = artifact["scaler"]
feature_columns = artifact["feature_columns"]

with st.container():
    st.markdown('<div class="form-wrap">', unsafe_allow_html=True)
    st.subheader("Health Details")
    st.markdown("<p class='small-muted'>Fill in maternal health details and click <b>Check Risk Level</b>.</p>", unsafe_allow_html=True)

    age = st.slider("Age", 18, 45, 27)
    blood_pressure = st.slider("Blood Pressure (mmHg)", 80, 180, 118)
    blood_sugar = st.slider("Blood Sugar (mg/dL)", 60, 250, 98)
    body_temperature = st.slider("Body Temperature (°F)", 96.0, 104.0, 98.4, 0.1)
    heart_rate = st.slider("Heart Rate (bpm)", 50, 150, 82)
    hemoglobin = st.slider("Hemoglobin Level (g/dL)", 7.0, 16.0, 11.8, 0.1)
    urine_protein = st.slider("Urine Protein Level (mg/dL)", 0, 500, 120)

    gravida = st.number_input("Number of Previous Pregnancies (Gravida)", min_value=0, max_value=12, value=1, step=1)
    para = st.number_input("Number of Previous Births (Para)", min_value=0, max_value=12, value=0, step=1)

    col1, col2 = st.columns(2)
    with col1:
        weight = st.slider("Weight (kg)", 35.0, 140.0, 68.0, 0.5)
    with col2:
        height = st.slider("Height (cm)", 135.0, 190.0, 158.0, 0.5)

    complications = st.selectbox("Previous Pregnancy Complications", ["No", "Yes"])
    stress_level = st.slider("Stress Level", 0, 10, 4)
    physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
    edema = st.radio("Edema", ["No", "Yes"], horizontal=True)
    smoking_alcohol = st.radio("Smoking / Alcohol History", ["No", "Yes"], horizontal=True)

    predict_btn = st.button("Check Risk Level", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

if predict_btn:
    input_payload = {
        "Age": age,
        "BloodPressure": float(blood_pressure),
        "BloodSugar": float(blood_sugar),
        "BodyTemperature": float(body_temperature),
        "HeartRate": int(heart_rate),
        "Hemoglobin": float(hemoglobin),
        "UrineProtein": float(urine_protein),
        "Gravida": int(gravida),
        "Para": int(para),
        "Weight": float(weight),
        "Height": float(height),
        "PreviousPregnancyComplications": 1 if complications == "Yes" else 0,
        "StressLevel": int(stress_level),
        "PhysicalActivityLevel": {"Low": 0, "Moderate": 1, "High": 2}[physical_activity],
        "Edema": 1 if edema == "Yes" else 0,
        "SmokingAlcoholHistory": 1 if smoking_alcohol == "Yes" else 0,
    }

    sample = pd.DataFrame([input_payload])[feature_columns]
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]

    if prediction == 0:
        st.success("✅ Predicted Risk Level: Low Risk")
    else:
        st.error("🚨 Predicted Risk Level: Medium Risk")

    st.info(f"Model used: {artifact['model_name']}")
