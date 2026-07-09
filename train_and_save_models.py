import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Create models directory
os.makedirs('models', exist_ok=True)

# Helper function to convert numpy arrays/types to serializable python types
def make_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj

# ==========================================
# 1. SYMPTOM-BASED DISEASE PREDICTION
# ==========================================
print("Training Symptom-Based Disease Prediction models...")
df_symptom = pd.read_csv("Training(symptom-based).csv")
if "Unnamed: 133" in df_symptom.columns:
    df_symptom.drop("Unnamed: 133", axis=1, inplace=True)

X_sym = df_symptom.drop("prognosis", axis=1)
y_sym = df_symptom["prognosis"]

# Save feature list
joblib.dump(X_sym.columns.tolist(), "models/symptom_features.joblib")

X_sym_train, X_sym_test, y_sym_train, y_sym_test = train_test_split(
    X_sym, y_sym, test_size=0.2, random_state=42
)

symptom_models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

symptom_metrics = {}

for name, model in symptom_models.items():
    model.fit(X_sym_train, y_sym_train)
    y_pred = model.predict(X_sym_test)
    
    acc = accuracy_score(y_sym_test, y_pred)
    prec = precision_score(y_sym_test, y_pred, average="weighted")
    rec = recall_score(y_sym_test, y_pred, average="weighted")
    f1 = f1_score(y_sym_test, y_pred, average="weighted")
    cm = confusion_matrix(y_sym_test, y_pred)
    
    # Save the model
    model_key = name.lower().replace(" ", "_")
    joblib.dump(model, f"models/symptom_{model_key}.joblib")
    
    symptom_metrics[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classes": model.classes_.tolist()
    }

with open("models/symptom_metrics.json", "w") as f:
    json.dump(make_serializable(symptom_metrics), f, indent=4)

print("Symptom-based models trained successfully.")

# ==========================================
# 2. HEART DISEASE PREDICTION
# ==========================================
print("\nTraining Heart Disease Prediction models...")
df_heart = pd.read_csv("heart.csv")

# Fill zeros with mean of non-zeros
chol_mean = df_heart.loc[df_heart['Cholesterol'] != 0, 'Cholesterol'].mean()
df_heart['Cholesterol'] = df_heart['Cholesterol'].replace(0, chol_mean).round(2)

bp_mean = df_heart.loc[df_heart['RestingBP'] != 0, 'RestingBP'].mean()
df_heart['RestingBP'] = df_heart['RestingBP'].replace(0, bp_mean).round(2)

# One-hot encoding
df_heart_encoded = pd.get_dummies(df_heart, drop_first=True).astype(int)

X_heart = df_heart_encoded.drop('HeartDisease', axis=1)
y_heart = df_heart_encoded['HeartDisease']

# Save column names list
joblib.dump(X_heart.columns.tolist(), "models/heart_columns.joblib")

X_heart_train, X_heart_test, y_heart_train, y_heart_test = train_test_split(
    X_heart, y_heart, stratify=y_heart, test_size=0.2, random_state=42
)

# Scale features
scaler_heart = StandardScaler()
X_heart_train_scaled = scaler_heart.fit_transform(X_heart_train)
X_heart_test_scaled = scaler_heart.transform(X_heart_test)

# Save scaler
joblib.dump(scaler_heart, "models/heart_scaler.joblib")

heart_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True, random_state=42)
}

heart_metrics = {}

for name, model in heart_models.items():
    model.fit(X_heart_train_scaled, y_heart_train)
    y_pred = model.predict(X_heart_test_scaled)
    
    acc = accuracy_score(y_heart_test, y_pred)
    f1 = f1_score(y_heart_test, y_pred)
    prec = precision_score(y_heart_test, y_pred, average="weighted")
    rec = recall_score(y_heart_test, y_pred, average="weighted")
    cm = confusion_matrix(y_heart_test, y_pred)
    
    # ROC curve calculations
    y_probs = model.predict_proba(X_heart_test_scaled)[:, 1]
    auc = roc_auc_score(y_heart_test, y_probs)
    fpr, tpr, _ = roc_curve(y_heart_test, y_probs)
    
    model_key = name.lower().replace(" (rbf kernel)", "").replace(" ", "_")
    joblib.dump(model, f"models/heart_{model_key}.joblib")
    
    heart_metrics[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        },
        "confusion_matrix": cm.tolist()
    }

with open("models/heart_metrics.json", "w") as f:
    json.dump(make_serializable(heart_metrics), f, indent=4)

print("Heart Disease models trained successfully.")

# ==========================================
# 3. DIABETES PREDICTION
# ==========================================
print("\nTraining Diabetes Prediction models...")
df_diabetes = pd.read_csv("diabetes.csv")

# Replace 0 with NaN for specific columns
cols_to_nan = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_nan:
    df_diabetes[col] = df_diabetes[col].replace(0, np.nan)
    df_diabetes[col].fillna(df_diabetes[col].median(), inplace=True)

# Feature engineering
df_diabetes['AgeGroup'] = pd.cut(
    df_diabetes['Age'],
    bins=[20,30,40,50,60,100],
    labels=[1,2,3,4,5]
).astype(int)

def bmi_category(x):
    if x < 18.5:
        return 0
    elif x < 25:
        return 1
    elif x < 30:
        return 2
    else:
        return 3

df_diabetes['BMI_Category'] = df_diabetes['BMI'].apply(bmi_category)

X_diab = df_diabetes.drop('Outcome', axis=1)
y_diab = df_diabetes['Outcome']

# Save columns
joblib.dump(X_diab.columns.tolist(), "models/diabetes_columns.joblib")

X_diab_train, X_diab_test, y_diab_train, y_diab_test = train_test_split(
    X_diab, y_diab, test_size=0.2, random_state=160, stratify=y_diab
)

scaler_diab = StandardScaler()
X_diab_train_scaled = scaler_diab.fit_transform(X_diab_train)
X_diab_test_scaled = scaler_diab.transform(X_diab_test)

# Save scaler
joblib.dump(scaler_diab, "models/diabetes_scaler.joblib")

diabetes_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

diabetes_metrics = {}

for name, model in diabetes_models.items():
    model.fit(X_diab_train_scaled, y_diab_train)
    y_pred = model.predict(X_diab_test_scaled)
    
    acc = accuracy_score(y_diab_test, y_pred)
    f1 = f1_score(y_diab_test, y_pred)
    prec = precision_score(y_diab_test, y_pred, average="weighted")
    rec = recall_score(y_diab_test, y_pred, average="weighted")
    cm = confusion_matrix(y_diab_test, y_pred)
    
    y_probs = model.predict_proba(X_diab_test_scaled)[:, 1]
    auc = roc_auc_score(y_diab_test, y_probs)
    fpr, tpr, _ = roc_curve(y_diab_test, y_probs)
    
    model_key = name.lower().replace(" ", "_")
    joblib.dump(model, f"models/diabetes_{model_key}.joblib")
    
    diabetes_metrics[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        },
        "confusion_matrix": cm.tolist()
    }

with open("models/diabetes_metrics.json", "w") as f:
    json.dump(make_serializable(diabetes_metrics), f, indent=4)

print("Diabetes models trained successfully.")

# ==========================================
# 4. MENTAL HEALTH RISK PREDICTION
# ==========================================
print("\nTraining Mental Health Risk Prediction models...")
df_mental = pd.read_csv("mental_health_risk_dataset.csv")

cat_cols = [
    'gender', 'marital_status', 'education_level',
    'employment_status', 'panic_attack_history',
    'family_history_mental_illness',
    'previous_mental_health_diagnosis',
    'therapy_history', 'substance_use'
]

# Label encode and save encoders
mental_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_mental[col] = le.fit_transform(df_mental[col])
    mental_encoders[col] = le

# Save encoders
joblib.dump(mental_encoders, "models/mental_encoders.joblib")

X_mental = df_mental.drop("mental_health_risk", axis=1)
y_mental = df_mental["mental_health_risk"]

# Save columns
joblib.dump(X_mental.columns.tolist(), "models/mental_columns.joblib")

X_mental_train, X_mental_test, y_mental_train, y_mental_test = train_test_split(
    X_mental, y_mental, test_size=0.2, random_state=42
)

scaler_mental = StandardScaler()
X_mental_train_scaled = scaler_mental.fit_transform(X_mental_train)
X_mental_test_scaled = scaler_mental.transform(X_mental_test)

# Save scaler
joblib.dump(scaler_mental, "models/mental_scaler.joblib")

mental_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=42)
}

mental_metrics = {}

for name, model in mental_models.items():
    model.fit(X_mental_train_scaled, y_mental_train)
    y_pred = model.predict(X_mental_test_scaled)
    
    acc = accuracy_score(y_mental_test, y_pred)
    prec = precision_score(y_mental_test, y_pred, average="weighted")
    rec = recall_score(y_mental_test, y_pred, average="weighted")
    f1 = f1_score(y_mental_test, y_pred, average="weighted")
    cm = confusion_matrix(y_mental_test, y_pred)
    
    model_key = name.lower().replace(" ", "_")
    joblib.dump(model, f"models/mental_{model_key}.joblib")
    
    mental_metrics[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }

with open("models/mental_metrics.json", "w") as f:
    json.dump(make_serializable(mental_metrics), f, indent=4)

print("Mental Health Risk models trained successfully.")
print("\nAll models trained and saved to the models/ directory!")
