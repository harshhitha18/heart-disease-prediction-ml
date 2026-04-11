# =====================================================
# HEART DISEASE PREDICTION USING ML CLASSIFICATION MODELS
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

import warnings
warnings.filterwarnings("ignore")

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("heart.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# =========================
# 2. DATA UNDERSTANDING & CLEANING
# =========================
print("\nMissing Values:\n", df.isnull().sum())

# Fill missing values (if any)
df.fillna(df.median(), inplace=True)

# =========================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =========================
plt.figure(figsize=(6,4))
sns.countplot(x="target", data=df)
plt.title("Target Variable Distribution")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# =========================
# 4. FEATURE & TARGET SPLIT
# =========================
X = df.drop("target", axis=1)
y = df["target"]

# =========================
# 5. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 6. FEATURE SCALING
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 7. MODEL INITIALIZATION
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64,32),
                                    activation="relu",
                                    max_iter=500,
                                    random_state=42)
}

# =========================
# 8. MODEL TRAINING & EVALUATION
# =========================
results = []

for name, model in models.items():

    if name in ["Logistic Regression", "Neural Network"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    results.append([name, acc, prec, rec, f1, roc])

    print(f"\n{name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# =========================
# 9. MODEL COMPARISON
# =========================
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
)

print("\nMODEL PERFORMANCE COMPARISON:")
print(results_df)

# =========================
# 10. HYPERPARAMETER TUNING (RANDOM FOREST)
# =========================
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="roc_auc"
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_

# =========================
# 11. FEATURE IMPORTANCE
# =========================
importances = best_rf.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFEATURE IMPORTANCE:")
print(feature_importance_df)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance - Random Forest")
plt.show()

import joblib

joblib.dump(best_rf, "best_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")

