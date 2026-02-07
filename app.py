import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="AI Symptom Checker", layout="centered")

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = joblib.load("model.pkl")

# ---------------------------
# Styling (Pretty UI)
# ---------------------------
st.markdown("""
<style>

.main {
    background-color:#F6F9FC;
}

.big-title {
    text-align:center;
    font-size:40px;
    font-weight:700;
    color:#2E86C1;
}

.card {
    padding:20px;
    border-radius:15px;
    background:white;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
    margin-bottom:15px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------
# Gemini helper
# ---------------------------
def gemini_explain(disease):
    model_ai = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Explain {disease} in simple language for a normal patient.
    Include:
    - what it is
    - precautions
    - diet
    - when to see doctor
    Keep it short and friendly.
    """

    response = model_ai.generate_content(prompt)
    return response.text


# ---------------------------
# UI
# ---------------------------
st.markdown('<p class="big-title">🩺 AI Symptom Checker</p>', unsafe_allow_html=True)

st.write("Type your symptoms separated by commas")

# 🔹 Symptom list (same as training)
symptoms = [
    "fever", "cough", "headache",
    "fatigue", "cold"
]

# ---------------------------
# 🔍 TEXT SEARCH INPUT
# ---------------------------
user_text = st.text_input(
    "Example: fever, cough, headache"
)

# ---------------------------
# Convert text → binary vector
# ---------------------------
def text_to_vector(text):
    tokens = [t.strip().lower() for t in text.split(",")]

    vector = []
    for s in symptoms:
        vector.append(1 if s in tokens else 0)

    return vector


# ---------------------------
# Prediction
# ---------------------------
if st.button("🔮 Predict Disease"):

    if user_text == "":
        st.warning("Please enter symptoms")
    else:

        vector = text_to_vector(user_text)
        df = pd.DataFrame([vector], columns=symptoms)

        probs = model.predict_proba(df)[0]
        classes = model.classes_

        results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

        st.subheader("Top Predictions")

        # ---------------------------
        # Pretty cards
        # ---------------------------
        for disease, prob in results[:3]:

            st.markdown(f"""
            <div class="card">
            <h4>{disease}</h4>
            <p>Probability: <b>{prob*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

        # ---------------------------
        # Gemini Explanation
        # ---------------------------
        st.subheader("🧠 AI Medical Guidance")

        with st.spinner("Generating advice using Gemini..."):
            explanation = gemini_explain(results[0][0])

        st.markdown(f"""
        <div class="card">
        {explanation}
        </div>
        """, unsafe_allow_html=True)
