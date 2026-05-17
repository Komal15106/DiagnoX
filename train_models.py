"""
train_models.py
===============
Trains and serialises all three ML models for the Multiple Disease Prediction System.

Run once before launching the Streamlit app:
    python train_models.py

Outputs (saved to models/):
    diabetes_model.pkl   — SVC(kernel='rbf')  trained on diabetes.csv
    diabetes_scaler.pkl  — StandardScaler fitted on diabetes training set
    heart_model.pkl      — SVC(kernel='linear') trained on heart.csv
    heart_scaler.pkl     — StandardScaler fitted on heart training set
    symptom_model.pkl    — RandomForestClassifier trained on symptoms.csv
    symptom_encoder.pkl  — LabelEncoder for disease name decoding
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1.  DIABETES MODEL  (Pima Indians Diabetes Dataset)
# ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  TRAINING: Diabetes SVM (kernel=rbf)")
print("="*55)

try:
    df_diab = pd.read_csv("data/diabetes.csv")
    print(f"  Loaded diabetes.csv  — {df_diab.shape[0]} rows, {df_diab.shape[1]} cols")

    # Replace biologically-impossible zeros with column mean
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        df_diab[col] = df_diab[col].replace(0, np.nan)
        df_diab[col].fillna(df_diab[col].mean(), inplace=True)

    X_d = df_diab.drop("Outcome", axis=1).values
    y_d = df_diab["Outcome"].values

    X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
        X_d, y_d, test_size=0.2, random_state=42, stratify=y_d
    )

    scaler_d = StandardScaler()
    X_d_train = scaler_d.fit_transform(X_d_train)
    X_d_test  = scaler_d.transform(X_d_test)

    svm_d = SVC(kernel="rbf", probability=True, random_state=42)
    svm_d.fit(X_d_train, y_d_train)

    acc_d = accuracy_score(y_d_test, svm_d.predict(X_d_test))
    print(f"  ✔  Test Accuracy : {acc_d*100:.2f}%")

    with open("models/diabetes_model.pkl",  "wb") as f: pickle.dump(svm_d,     f)
    with open("models/diabetes_scaler.pkl", "wb") as f: pickle.dump(scaler_d,  f)
    print("  ✔  Saved: models/diabetes_model.pkl & models/diabetes_scaler.pkl")

except FileNotFoundError:
    print("  ⚠  data/diabetes.csv not found — skipping diabetes model training.")
    print("     Download: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")


# ─────────────────────────────────────────────────────────
# 2.  HEART DISEASE MODEL  (Cleveland Heart Disease Dataset)
# ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  TRAINING: Heart Disease SVM (kernel=linear)")
print("="*55)

try:
    df_heart = pd.read_csv("data/heart.csv")
    print(f"  Loaded heart.csv  — {df_heart.shape[0]} rows, {df_heart.shape[1]} cols")

    # Drop rows with missing values (Cleveland dataset has '?' entries)
    df_heart.replace("?", np.nan, inplace=True)
    df_heart.dropna(inplace=True)

    # Ensure all columns are numeric
    df_heart = df_heart.apply(pd.to_numeric)

    # The target column may be named 'target' or 'condition'
    target_col = "target" if "target" in df_heart.columns else df_heart.columns[-1]
    # Binarise: 0 = no disease, 1 = disease present
    df_heart[target_col] = (df_heart[target_col] > 0).astype(int)

    X_h = df_heart.drop(target_col, axis=1).values
    y_h = df_heart[target_col].values

    X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
        X_h, y_h, test_size=0.2, random_state=42, stratify=y_h
    )

    scaler_h = StandardScaler()
    X_h_train = scaler_h.fit_transform(X_h_train)
    X_h_test  = scaler_h.transform(X_h_test)

    svm_h = SVC(kernel="linear", probability=True, random_state=42)
    svm_h.fit(X_h_train, y_h_train)

    acc_h = accuracy_score(y_h_test, svm_h.predict(X_h_test))
    print(f"  ✔  Test Accuracy : {acc_h*100:.2f}%")

    with open("models/heart_model.pkl",  "wb") as f: pickle.dump(svm_h,    f)
    with open("models/heart_scaler.pkl", "wb") as f: pickle.dump(scaler_h, f)
    print("  ✔  Saved: models/heart_model.pkl & models/heart_scaler.pkl")

except FileNotFoundError:
    print("  ⚠  data/heart.csv not found — skipping heart disease model training.")
    print("     Download: https://www.kaggle.com/datasets/ronitf/heart-disease-uci")


# ─────────────────────────────────────────────────────────
# 3.  GENERAL SYMPTOM CHECKER  (130+ symptoms / 40+ diseases)
# ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  TRAINING: Symptom Checker Random Forest")
print("="*55)

try:
    df_sym = pd.read_csv("data/symptoms.csv")
    print(f"  Loaded symptoms.csv  — {df_sym.shape[0]} rows, {df_sym.shape[1]} cols")

    # Expected format: first column = disease label, remaining = symptom binary columns
    disease_col = df_sym.columns[0]
    X_s = df_sym.drop(disease_col, axis=1).values.astype(int)
    y_s_raw = df_sym[disease_col].values

    le = LabelEncoder()
    y_s = le.fit_transform(y_s_raw)

    X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
        X_s, y_s, test_size=0.2, random_state=42, stratify=y_s
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_s_train, y_s_train)

    acc_s = accuracy_score(y_s_test, rf.predict(X_s_test))
    print(f"  ✔  Test Accuracy : {acc_s*100:.2f}%")

    with open("models/symptom_model.pkl",   "wb") as f: pickle.dump(rf, f)
    with open("models/symptom_encoder.pkl", "wb") as f: pickle.dump(le, f)
    print("  ✔  Saved: models/symptom_model.pkl & models/symptom_encoder.pkl")

    # Save the ordered list of symptom feature names for runtime lookup
    symptom_cols = list(df_sym.drop(disease_col, axis=1).columns)
    with open("models/symptom_columns.pkl", "wb") as f: pickle.dump(symptom_cols, f)
    print(f"  ✔  Saved symptom column order ({len(symptom_cols)} features)")

except FileNotFoundError:
    print("  ⚠  data/symptoms.csv not found — skipping symptom model training.")
    print("     Dataset: Kaggle 'Disease Symptom Prediction' by itachi9604")

print("\n" + "="*55)
print("  Training complete. Run:  streamlit run main.py")
print("="*55 + "\n")