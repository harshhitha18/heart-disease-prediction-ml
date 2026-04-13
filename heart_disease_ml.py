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
from imblearn.over_sampling import SMOTE

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
# 2. DATA CLEANING
# =========================
print("\nMissing Values:\n", df.isnull().sum())
df.fillna(df.median(), inplace=True)

# =========================
# 3. EDA (BASIC)
# =========================
plt.figure(figsize=(6,4))
sns.countplot(x="target", data=df)
plt.title("Target Distribution")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# =========================
# 🔥 FEATURE ENGINEERING
# =========================

# 1. Age groups (categorical pattern)
df["age_group"] = pd.cut(df["age"],
                        bins=[20, 40, 60, 100],
                        labels=[0, 1, 2]).astype(int)

# 2. Cholesterol to BP ratio (health indicator)
df["chol_bp_ratio"] = df["chol"] / df["trestbps"]

# 3. Combined risk score (domain-based)
df["risk_score"] = df["cp"] + df["exang"] + df["oldpeak"]

print("\nNew Features Added:")
print(df[["age_group", "chol_bp_ratio", "risk_score"]].head())

# =========================
# 4. FEATURE / TARGET SPLIT
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
# 6. HANDLE IMBALANCE (SMOTE)
# =========================
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nBefore SMOTE:\n", y_train.value_counts())
print("After SMOTE:\n", pd.Series(y_train_resampled).value_counts())

# =========================
# 7. FEATURE SCALING
# =========================
scaler = StandardScaler()

X_train_res_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# =========================
# 8. MODEL INITIALIZATION
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
# 9. MODEL TRAINING & EVALUATION
# =========================
results = []

for name, model in models.items():

    # Models that need scaling
    if name in ["Logistic Regression", "Neural Network"]:
        model.fit(X_train_res_scaled, y_train_resampled)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Tree models
    else:
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

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
# 10. MODEL COMPARISON
# =========================
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
)

print("\nMODEL PERFORMANCE COMPARISON:")
print(results_df)

# =========================
# 11. HYPERPARAMETER TUNING
# =========================
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],              
    "min_samples_split": [5, 10],        
    "min_samples_leaf": [2, 4]           
}

rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

# Train on balanced data
grid.fit(X_train_resampled, y_train_resampled)

best_rf = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)

# =========================
# 🔥 STEP 2: CROSS VALIDATION (FIXED VERSION)
# =========================

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Create pipeline (SMOTE + model)
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(**grid.best_params_, random_state=42))
])

# Stratified K-Fold (FIX 1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross validation
cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)

print("\nCross Validation Results:")
print("Scores:", cv_scores)
print("Mean ROC-AUC:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

# Model stability check (FIX 2)
print("\nModel Stability Check:")

if cv_scores.std() < 0.05:
    print("Model is stable ✅")
else:
    print("Model is unstable ⚠️")

# =========================
# 12. FINAL MODEL EVALUATION
# =========================
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

print("\nFinal Random Forest Performance:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

import shap
import matplotlib.pyplot as plt

print("\nRunning SHAP Explainability...")

# Create explainer
explainer = shap.TreeExplainer(best_rf)

# Get SHAP values
shap_values = explainer.shap_values(X_test)

# 🔥 Create figure manually
plt.figure()

shap.summary_plot(shap_values, X_test, show=False)

# 🔥 SAVE IMAGE (IMPORTANT)
plt.savefig("shap_summary.png", bbox_inches='tight')

print("SHAP plot saved as shap_summary.png")

# Optional display
plt.show()




