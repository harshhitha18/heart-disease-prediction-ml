import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CardioRisk AI",
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

# ---------------- SESSION STATE ----------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.probability = None

# ---------------- HEADER ----------------
st.title("❤️ CardioRisk AI")
st.markdown("### ML-Based Heart Disease Prediction System")
st.write("Enter patient details or upload a CSV file to assess heart disease risk.")

st.divider()

# ================== MANUAL INPUT ==================
st.subheader("🧑 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, value=40)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, value=120)
    chol = st.number_input("Cholesterol Level", 100, 600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    restecg = st.selectbox("ECG Result (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (0–3)", [0, 1, 2, 3])

# ---------------- BUTTONS ----------------
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🔍 Predict Risk"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak,
                                slope, ca, thal]])

        input_scaled = scaler.transform(input_data)

        st.session_state.prediction = model.predict(input_scaled)[0]
        st.session_state.probability = model.predict_proba(input_scaled)[0][1]

with col_btn2:
    if st.button("🧹 Clear"):
        st.session_state.prediction = None
        st.session_state.probability = None

# ---------------- RESULTS ----------------
if st.session_state.prediction is not None:
    st.divider()
    st.subheader("📊 Prediction Result")

    prob = st.session_state.probability
    st.write(f"### Risk Probability: **{prob * 100:.2f}%**")

    # 🔥 Progress Bar
    st.progress(int(prob * 100))

    # 🔥 Better Risk Interpretation
    if prob > 0.7:
        st.error("⚠️ High Risk of Heart Disease")
    elif prob > 0.4:
        st.warning("⚠️ Moderate Risk - Consider medical consultation")
    else:
        st.success("✅ Low Risk")

    # 🔥 Explanation Box
    st.info("💡 Prediction is based on clinical parameters like age, cholesterol, blood pressure, and heart rate.")

    st.caption("⚠️ This tool is for educational purposes and not a medical diagnosis.")

st.divider()

# ================== CSV UPLOAD ==================
st.subheader("📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload CSV with 13 input features (no target column)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Data Preview")
    st.dataframe(df.head())

    if st.button("📈 Run Batch Prediction"):

        df_scaled = scaler.transform(df)

        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)[:, 1]

        df["Prediction"] = predictions
        df["Risk_Probability"] = probabilities

        st.success("✅ Predictions completed successfully!")
        st.dataframe(df)

        st.download_button(
            "⬇️ Download Results",
            df.to_csv(index=False),
            file_name="heart_disease_predictions.csv",
            mime="text/csv"
        )