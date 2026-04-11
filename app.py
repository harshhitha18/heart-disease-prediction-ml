import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="centered"
)

# ---------------- LOAD MODEL + SCALER ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ---------------- SESSION STATE INIT ----------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.probability = None

# ---------------- TITLE ----------------
st.title("Heart Disease Prediction System ❤️")
st.write("Enter patient medical details or upload a CSV file to predict heart disease risk.")

st.divider()

# ================== MANUAL INPUT ==================
st.subheader("🧑 Manual Patient Entry")

age = st.number_input("Age", 20, 100, value=40)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, value=120)
chol = st.number_input("Cholesterol Level", 100, 600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("ECG Result (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, value=1.0, step=0.1)
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0–3)", [0, 1, 2, 3])

# ---------------- BUTTONS ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Predict Heart Disease Risk"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])

        # 🔥 APPLY SCALING (IMPORTANT)
        input_scaled = scaler.transform(input_data)

        st.session_state.prediction = model.predict(input_scaled)[0]
        st.session_state.probability = model.predict_proba(input_scaled)[0][1]

with col2:
    if st.button("🧹 Clear Prediction"):
        st.session_state.prediction = None
        st.session_state.probability = None

# ---------------- DISPLAY RESULT ----------------
if st.session_state.prediction is not None:
    st.subheader("📊 Prediction Result")
    st.write(f"**Probability of Heart Disease:** {st.session_state.probability * 100:.2f}%")

    if st.session_state.prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.caption("⚠️ This system is intended for decision support and not for medical diagnosis.")

st.divider()

# ================== CSV UPLOAD ==================
st.subheader("📂 Upload Patient Data (CSV)")

uploaded_file = st.file_uploader(
    "Upload CSV with 13 input features (no target column)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:")
    st.dataframe(df)

    if st.button("📈 Predict for Uploaded Data"):

        # 🔥 APPLY SCALING TO FULL DATA
        df_scaled = scaler.transform(df)

        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)[:, 1]

        df["Prediction"] = predictions
        df["Risk_Probability"] = probabilities

        st.success("Predictions completed successfully!")
        st.dataframe(df)

        st.download_button(
            "⬇️ Download Results",
            df.to_csv(index=False),
            file_name="heart_disease_predictions.csv",
            mime="text/csv"
        )

