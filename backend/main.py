from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("../best_rf_model.pkl")

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running 🚀"}

@app.post("/predict")
def predict(data: dict):
    try:
        values = np.array(list(data.values())).reshape(1, -1)

        pred = model.predict(values)[0]
        prob = model.predict_proba(values)[0][1]

        return {
            "prediction": int(pred),
            "probability": float(prob)
        }

    except Exception as e:
        return {"error": str(e)}