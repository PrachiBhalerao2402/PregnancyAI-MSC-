"""Streamlit app for pregnancy risk prediction with chatbot tab."""

from __future__ import annotations

import os
from typing import Dict, List

import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Pregnancy Risk Predictor", page_icon="🩺", layout="centered")

st.markdown(
    """
    <style>
      .stApp {background: linear-gradient(180deg,#fff8f9 0%, #fff0f5 100%);} 
      .form-wrap {background: #fffafb; border-radius: 16px; padding: 1.2rem 1.2rem 0.8rem 1.2rem;
                  box-shadow: 0 10px 26px rgba(184,74,104,0.10); border: 1px solid #f1d3db;}
      h1, h2, h3 {color: #7f2f47 !important;}
      .small-muted {color:#8c5670; font-size:0.92rem; margin-top:-0.5rem; margin-bottom:1rem;}
      .stButton>button {width:100%; border-radius:10px; padding:0.58rem 0.9rem; font-weight:600;}
      .advice-card {background:#fff1f5; border:1px solid #f6c8d5; border-radius:12px; padding:0.8rem 0.9rem;}
      .advice-title {color:#b32648; font-weight:700; margin-bottom:0.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤰 Pregnancy Risk Prediction System")
st.caption("Low / Medium risk screening + women-health assistant")

try:
    artifact = joblib.load("models/best_pregnancy_risk_model.joblib")
except FileNotFoundError:
    st.error("Model file not found. Run `python train_models.py` first to generate the model artifact.")
    st.stop()

model = artifact["model"]
scaler = artifact["scaler"]
feature_columns = artifact["feature_columns"]

for key, default in {
    "last_prediction": None,
    "last_payload": None,
    "risk_level_for_advice": None,
    "chat_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def parse_number(value: str, label: str, as_int: bool = False):
    value = value.strip()
    if value == "":
        raise ValueError(f"Please enter {label}.")
    try:
        return int(float(value)) if as_int else float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid value for {label}. Use numeric input.") from exc


def risk_tier_for_advice(payload, prediction):
    severity_points = 0
    severity_points += 1 if payload["BloodPressure"] >= 140 else 0
    severity_points += 1 if payload["BloodSugar"] >= 140 else 0
    severity_points += 1 if payload["Hemoglobin"] < 9.5 else 0
    severity_points += 1 if payload["StressLevel"] >= 8 else 0
    severity_points += 1 if payload["Edema"] == 1 else 0
    severity_points += 1 if payload["PreviousPregnancyComplications"] == 1 else 0

    if prediction == 0 and severity_points <= 1:
        return "low"
    if severity_points >= 3:
        return "high"
    return "medium"


def build_advice(payload, tier):
    if tier == "low":
        base = "Low Risk: Your current profile looks stable."
        tips = [
            "Continue balanced nutrition with iron, folate, calcium, and protein.",
            "Stay hydrated and keep regular sleep habits.",
            "Do light to moderate prenatal-safe activity after doctor approval.",
            "Continue routine antenatal visits.",
        ]
    elif tier == "medium":
        base = "Medium Risk: You need closer monitoring to avoid complications."
        tips = [
            "Track blood pressure and glucose at home regularly.",
            "Reduce salt, refined sugar, and fried foods.",
            "Improve stress management with breathing and walking.",
            "Plan more frequent review with your gynecologist.",
        ]
    else:
        base = "High Risk: Seek urgent specialist care for safer pregnancy management."
        tips = [
            "Contact your obstetric specialist immediately.",
            "Do not skip BP/sugar/fetal monitoring.",
            "Follow strict medication and diet instructions.",
            "Go to emergency care if severe symptoms appear.",
        ]

    if payload["Hemoglobin"] < 10.5:
        tips.append("Discuss iron supplementation with your doctor.")
    if payload["StressLevel"] >= 7:
        tips.append("Prioritize rest and emotional support daily.")

    return base, tips[:5]


def local_pregnancy_bot(question: str) -> str:
    q = question.lower().strip()
    if not q:
        return "Please type your question 🌸"

    rulebook = {
        "diet": "Diet plan: eat protein in each meal (eggs/paneer/fish/legumes), 2 fruits/day, green vegetables, nuts, and 2.5–3L water. Avoid unpasteurized/raw foods.",
        "exercise": "Exercise plan: 20–30 min brisk walk, prenatal stretches, and pelvic-floor training 4–5 days/week with doctor approval.",
        "bp": "For blood pressure: reduce salt, monitor BP daily, sleep on left side, and contact your obstetrician if BP stays high.",
        "pressure": "For blood pressure: reduce salt, monitor BP daily, sleep on left side, and contact your obstetrician if BP stays high.",
        "sugar": "For blood sugar: switch to low-GI carbs, split meals into smaller portions, include fiber/protein, and check glucose regularly.",
        "stress": "For stress: deep breathing 10 minutes twice daily, light walks, fixed sleep time, and emotional support from family.",
        "supplement": "Common supplements: folic acid, iron, calcium, vitamin D (only as prescribed by your doctor).",
        "trimester": "Trimester focus: 1st—nausea + folate; 2nd—anomaly scan + iron/calcium; 3rd—fetal movement + BP + delivery planning.",
        "warning": "Emergency signs: bleeding, severe headache, blurred vision, severe pain, reduced fetal movement, leaking fluid, breathlessness. Seek urgent care.",
        "nausea": "For nausea: take small frequent meals, keep dry snacks, ginger/lemon water, and avoid long empty stomach gaps.",
        "anemia": "For low Hb: iron-rich foods + vitamin C combination, avoid tea/coffee around meals, discuss iron therapy with doctor.",
        "sleep": "Sleep plan: 7–9 hrs/night, left lateral sleeping posture, avoid heavy dinner, and reduce screen time before bed.",
        "week": "If you share pregnancy week, I can give week-specific diet, exercise, scans, and warning-sign guidance.",
    }

    matches = [answer for key, answer in rulebook.items() if key in q]
    if matches:
        return "\n\n".join(matches[:2])

    return (
        "I can still help. Please ask in this format for best answers:\n"
        "• 'Diet plan for 5th month'\n"
        "• 'Exercise for back pain in pregnancy'\n"
        "• 'High BP/sugar what to do?'\n"
        "• 'What tests/scans this trimester?'\n"
        "• 'Emergency warning signs?'"
    )


def ask_gemini_if_configured(messages: List[Dict[str, str]]) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    convo = []
    for item in messages:
        role = "model" if item["role"] == "assistant" else "user"
        convo.append({"role": role, "parts": [{"text": item["content"]}]})

    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": "You are a compassionate prenatal health assistant. Give safe, practical guidance and remind users to seek a doctor for urgent symptoms."
                }
            ]
        },
        "contents": convo,
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 450},
    }

    try:
        resp = requests.post(url, json=payload, timeout=25)
        if resp.ok:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts).strip()
                if text:
                    return text
    except Exception:
        return None
    return None


tab_predict, tab_chatbot = st.tabs(["🩺 Risk Prediction", "💬 Women Health Chatbot"])

with tab_predict:
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
        prediction = int(model.predict(sample_scaled)[0])

        st.session_state.last_prediction = prediction
        st.session_state.last_payload = input_payload
        st.session_state.risk_level_for_advice = risk_tier_for_advice(input_payload, prediction)

    if st.session_state.last_prediction is not None:
        if st.session_state.last_prediction == 0:
            st.success("✅ Predicted Risk Level: Low Risk")
        else:
            st.error("🚨 Predicted Risk Level: Medium Risk")

        if st.button("Get Advice"):
            base, notes = build_advice(st.session_state.last_payload, st.session_state.risk_level_for_advice)
            st.markdown('<div class="advice-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="advice-title">Personalized Health Advice ({st.session_state.risk_level_for_advice.title()} Risk)</div>',
                unsafe_allow_html=True,
            )
            st.write(base)
            for tip in notes:
                st.write(f"• {tip}")
            st.markdown("</div>", unsafe_allow_html=True)

with tab_chatbot:
    st.markdown("### Ask Pregnancy Assistant")
    st.caption("Real-time style chat. If GEMINI_API_KEY is set, replies use Gemini API; otherwise safe local pregnancy bot responds.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.text_input(
        "Ask a pregnancy question",
        value="",
        placeholder="Ask about diet, exercise, supplements, trimester, stress, warning signs...",
        key="chat_prompt",
    )
    send_clicked = st.button("Send", key="send_chat", type="primary")

    if send_clicked:
        prompt = prompt.strip()
        if not prompt:
            st.warning("Please type a message before sending.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            llm_answer = ask_gemini_if_configured(st.session_state.chat_history[-8:])
            answer = llm_answer if llm_answer else local_pregnancy_bot(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
