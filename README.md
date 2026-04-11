# ❤️ CardioRisk AI – Heart Disease Prediction System

## 🚀 Overview
CardioRisk AI is an end-to-end Machine Learning system that predicts the risk of heart disease using clinical patient data. The project covers the complete ML lifecycle including data preprocessing, model training, hyperparameter tuning, evaluation, and deployment via an interactive Streamlit web application.

---

## 🌐 Live Demo
https://heart-disease-prediction-ml-lyey9gf2henc3ptaszc5vn.streamlit.app/

---

## 🧠 Features
- Multiple ML models (Logistic Regression, Decision Tree, Random Forest, Neural Network)
- Hyperparameter tuning using GridSearchCV
- Model evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Real-time prediction through an interactive UI
- Batch prediction using CSV upload
- Probability-based risk assessment for better interpretability

---

## 📊 Model Performance
- Accuracy: 83.6%
- Precision: 78%
- Recall: 97%
- F1 Score: 86.4%
- ROC-AUC: 91.6%

Best performing model: Random Forest Classifier (after hyperparameter tuning)

---

## 🔍 Key Insights
- Chest pain type (cp) and thalassemia (thal) are the most important features
- Maximum heart rate (thalach) and oldpeak significantly influence predictions
- Exercise-induced angina and number of vessels also contribute to risk assessment

---

## ⚙️ Tech Stack
- Programming Language: Python  
- Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn  
- Deployment/UI: Streamlit  
- Model Persistence: Joblib  

---

## 📁 Project Structure
heart-disease-prediction-ml/
│
├── app.py
├── heart_disease_ml.py
├── best_rf_model.pkl
├── scaler.pkl
├── heart.csv
├── requirements.txt
└── README.md

---

## ▶️ Run Locally

```bash
git clone https://github.com/harshhitha18/heart-disease-prediction-ml.git
cd heart-disease-prediction-ml
pip install -r requirements.txt
streamlit run app.py