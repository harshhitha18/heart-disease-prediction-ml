
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
 
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="❤️",
    layout="centered"
)
 
# ---------------- LOAD MODEL ----------------
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
 
    sex_options = {"Female": 0, "Male": 1}
    sex = sex_options[st.selectbox("Sex", list(sex_options.keys()))]
 
    cp_options = {
        "0 — Asymptomatic (No pain)": 0,
        "1 — Typical Angina": 1,
        "2 — Atypical Angina": 2,
        "3 — Non-anginal Pain": 3
    }
    cp = cp_options[st.selectbox("Chest Pain Type", list(cp_options.keys()))]
 
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, value=120)
    chol = st.number_input("Cholesterol Level (mg/dl)", 100, 600, value=200)
 
    fbs_options = {"No (≤ 120 mg/dl)": 0, "Yes (> 120 mg/dl)": 1}
    fbs = fbs_options[st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(fbs_options.keys()))]
 
with col2:
    restecg_options = {
        "0 — Normal": 0,
        "1 — ST-T Wave Abnormality": 1,
        "2 — Left Ventricular Hypertrophy": 2
    }
    restecg = restecg_options[st.selectbox("Resting ECG Result", list(restecg_options.keys()))]
 
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, value=150)
 
    exang_options = {"No": 0, "Yes": 1}
    exang = exang_options[st.selectbox("Exercise Induced Angina", list(exang_options.keys()))]
 
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, value=1.0, step=0.1)
 
    slope_options = {
        "0 — Upsloping": 0,
        "1 — Flat": 1,
        "2 — Downsloping": 2
    }
    slope = slope_options[st.selectbox("ST Slope", list(slope_options.keys()))]
 
    ca = st.selectbox("Major Vessels Coloured by Fluoroscopy (0–3)", [0, 1, 2, 3])
 
    thal_options = {
        "0 — Normal": 0,
        "1 — Fixed Defect": 1,
        "2 — Reversible Defect": 2,
        "3 — Unknown": 3
    }
    thal = thal_options[st.selectbox("Thalassemia Type", list(thal_options.keys()))]
 
# ---------------- BUTTONS ----------------
col_btn1, col_btn2 = st.columns(2)
 
with col_btn1:
    if st.button("🔍 Predict Risk"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak,
                                slope, ca, thal]])
 
        # Random Forest does NOT need scaling — use raw input
        st.session_state.prediction  = model.predict(input_data)[0]
        st.session_state.probability = model.predict_proba(input_data)[0][1]
 
with col_btn2:
    if st.button("🧹 Clear"):
        st.session_state.prediction  = None
        st.session_state.probability = None
 
# ---------------- RESULTS ----------------
if st.session_state.prediction is not None:
    st.divider()
    st.subheader("📊 Prediction Result")
 
    prob = st.session_state.probability
    st.write(f"### Risk Probability: **{prob * 100:.2f}%**")
 
    st.progress(int(prob * 100))
 
    if prob > 0.7:
        st.error("⚠️ High Risk of Heart Disease — Please consult a doctor immediately.")
    elif prob > 0.4:
        st.warning("⚠️ Moderate Risk — Consider medical consultation.")
    else:
        st.success("✅ Low Risk — Keep maintaining a healthy lifestyle!")
 
    st.info("💡 Prediction is based on clinical parameters like age, cholesterol, blood pressure, and heart rate.")
    st.caption("⚠️ This tool is for educational purposes only and is NOT a medical diagnosis.")
 
    # ---------------- FEATURE IMPORTANCE ----------------
    st.divider()
    st.subheader("📊 What Factors Influence This Prediction?")
 
    feature_names = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol',
                     'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate',
                     'Exercise Angina', 'Oldpeak', 'ST Slope', 'Major Vessels', 'Thalassemia']
 
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
 
    fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale=["#bbdefb", "#c62828"])
    fig.update_layout(coloraxis_showscale=False, height=400,
                      paper_bgcolor='#fff', plot_bgcolor='#f8f9fa',
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
 
st.divider()
 
# ================== CSV UPLOAD ==================
st.subheader("📂 Batch Prediction (CSV Upload)")
st.caption("Upload a CSV with these 13 columns in order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal")
 
uploaded_file = st.file_uploader(
    "Upload CSV (no target column)",
    type=["csv"]
)
 
if uploaded_file:
    df = pd.read_csv(uploaded_file)
 
    st.write("### 📄 Data Preview")
    st.dataframe(df.head())
 
    if st.button("📈 Run Batch Prediction"):
        # Random Forest does NOT need scaling — use raw df
        predictions   = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
 
        df["Prediction"]       = predictions
        df["Risk_Probability"] = (probabilities * 100).round(2)
        df["Risk_Label"]       = df["Prediction"].map({0: "✅ Low Risk", 1: "⚠️ High Risk"})
 
        st.success("✅ Predictions completed successfully!")
        st.dataframe(df)
 
        st.download_button(
            "⬇️ Download Results",
            df.to_csv(index=False),
            file_name="heart_disease_predictions.csv",
            mime="text/csv"
        )
 
# ---------------- FOOTER ----------------
st.divider()
st.markdown("""
<div style="text-align:center;color:#90a4ae;font-size:0.82rem;padding:0.5rem">
    ❤️ <b>CardioRisk AI</b> · ML-Based Heart Disease Prediction<br>
    <span style="font-size:0.75rem;">Built with Streamlit · Scikit-learn · Random Forest · Plotly</span>
</div>
""", unsafe_allow_html=True)