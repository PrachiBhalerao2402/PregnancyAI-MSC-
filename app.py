"""Streamlit app for pregnancy risk prediction."""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pregnancy Risk Predictor", page_icon="🩺", layout="centered")

st.markdown(
    """
    <style>
      .stApp {background: linear-gradient(180deg,#f7fbff 0%, #eef7ff 100%);} 
      .form-wrap {background: #ffffff; border-radius: 16px; padding: 1.2rem 1.2rem 0.8rem 1.2rem;
                  box-shadow: 0 8px 24px rgba(40,78,120,0.10); border: 1px solid #dceaf8;}
      h1, h2, h3 {color: #1f4b7a !important;}
      .small-muted {color:#5a7288; font-size:0.92rem; margin-top:-0.5rem; margin-bottom:1rem;}
      .stButton>button {width:100%; border-radius:10px; padding:0.58rem 0.9rem; font-weight:600;}
      .advice-card {background:#fff7f7; border:1px solid #ffd8df; border-radius:12px; padding:0.8rem 0.9rem;}
      .advice-title {color:#b32648; font-weight:700; margin-bottom:0.35rem;}
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


def parse_number(value: str, label: str, as_int: bool = False):
    value = value.strip()
    if value == "":
        raise ValueError(f"Please enter {label}.")
    try:
        return int(float(value)) if as_int else float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid value for {label}. Use numeric input.") from exc


def build_advice(payload, prediction):
    notes = []
    if payload["BloodPressure"] >= 130:
        notes.append("Monitor blood pressure daily and reduce salty foods.")
    if payload["BloodSugar"] >= 120:
        notes.append("Limit sugary foods and follow a low-glycemic diet plan.")
    if payload["Hemoglobin"] < 10.5:
        notes.append("Increase iron-rich foods and discuss iron supplements with your doctor.")
    if payload["StressLevel"] >= 7:
        notes.append("Use daily stress-reduction routines (sleep, breathing, light walking).")
    if payload["Edema"] == 1:
        notes.append("Track swelling and seek medical review if swelling worsens.")

    if prediction == 0:
        base = "Great signs overall. Continue prenatal vitamins, hydration, balanced nutrition, and regular antenatal checkups."
    else:
        base = "Please consult your doctor soon for targeted monitoring and a safer pregnancy plan."

    if not notes:
        notes = ["Keep routine antenatal visits and maintain healthy lifestyle habits."]

    return base, notes[:4]


with st.container():
    st.markdown('<div class="form-wrap">', unsafe_allow_html=True)
    st.subheader("Health Details")
    st.markdown(
        "<p class='small-muted'>Fill in maternal health details and click <b>Check Risk Level</b>.</p>",
        unsafe_allow_html=True,
    )

    age = st.text_input("Age", value="", placeholder="e.g. 28")
    blood_pressure = st.text_input("Blood Pressure (mmHg)", value="", placeholder="e.g. 120")
    blood_sugar = st.text_input("Blood Sugar (mg/dL)", value="", placeholder="e.g. 100")
    body_temperature = st.text_input("Body Temperature (°F)", value="", placeholder="e.g. 98.6")
    heart_rate = st.text_input("Heart Rate (bpm)", value="", placeholder="e.g. 80")
    hemoglobin = st.text_input("Hemoglobin Level (g/dL)", value="", placeholder="e.g. 12.5")
    urine_protein = st.text_input("Urine Protein Level (mg/dL)", value="", placeholder="e.g. 30")

    gravida = st.text_input("Number of Previous Pregnancies (Gravida)", value="", placeholder="e.g. 2")
    para = st.text_input("Number of Previous Births (Para)", value="", placeholder="e.g. 1")

    col1, col2 = st.columns(2)
    with col1:
        weight = st.text_input("Weight (kg)", value="", placeholder="e.g. 65")
    with col2:
        height = st.text_input("Height (cm)", value="", placeholder="e.g. 160")

    complications = st.selectbox("Previous Pregnancy Complications", ["No", "Yes"])
    stress_level = st.text_input("Stress Level (0-10)", value="", placeholder="e.g. 5")
    physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
    edema = st.radio("Edema", ["No", "Yes"], horizontal=True)
    smoking_alcohol = st.radio("Smoking / Alcohol History", ["No", "Yes"], horizontal=True)

    predict_btn = st.button("Check Risk Level", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

if predict_btn:
    try:
        input_payload = {
            "Age": parse_number(age, "Age", as_int=True),
            "BloodPressure": parse_number(blood_pressure, "Blood Pressure"),
            "BloodSugar": parse_number(blood_sugar, "Blood Sugar"),
            "BodyTemperature": parse_number(body_temperature, "Body Temperature"),
            "HeartRate": parse_number(heart_rate, "Heart Rate", as_int=True),
            "Hemoglobin": parse_number(hemoglobin, "Hemoglobin"),
            "UrineProtein": parse_number(urine_protein, "Urine Protein"),
            "Gravida": parse_number(gravida, "Gravida", as_int=True),
            "Para": parse_number(para, "Para", as_int=True),
            "Weight": parse_number(weight, "Weight"),
            "Height": parse_number(height, "Height"),
            "PreviousPregnancyComplications": 1 if complications == "Yes" else 0,
            "StressLevel": parse_number(stress_level, "Stress Level", as_int=True),
            "PhysicalActivityLevel": {"Low": 0, "Moderate": 1, "High": 2}[physical_activity],
            "Edema": 1 if edema == "Yes" else 0,
            "SmokingAlcoholHistory": 1 if smoking_alcohol == "Yes" else 0,
        }
    except ValueError as err:
        st.error(str(err))
        st.stop()

    sample = pd.DataFrame([input_payload])[feature_columns]
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]

    if prediction == 0:
        st.success("✅ Predicted Risk Level: Low Risk")
    else:
        st.error("🚨 Predicted Risk Level: Medium Risk")

    base_advice, notes = build_advice(input_payload, prediction)

    if st.button("Get Advice"):
        st.markdown('<div class="advice-card">', unsafe_allow_html=True)
        st.markdown('<div class="advice-title">Personalized Health Advice</div>', unsafe_allow_html=True)
        st.write(base_advice)
        for tip in notes:
            st.write(f"• {tip}")
        st.markdown('</div>', unsafe_allow_html=True)
