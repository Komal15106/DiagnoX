import pickle
import numpy as np
import os

with open("models/heart_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/heart_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

for i in range(5):
    # random inputs based on typical ranges
    age = np.random.uniform(29, 77)
    sex = np.random.randint(0, 2)
    cp = np.random.randint(0, 4)
    trestbps = np.random.uniform(94, 200)
    chol = np.random.uniform(126, 564)
    fbs = 0
    restecg = 0
    thalach = np.random.uniform(71, 202)
    exang = 0
    oldpeak = np.random.uniform(0.0, 6.2)
    slope = 1
    ca = 0
    thal = np.random.randint(1, 4)
    
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    scaled = scaler.transform(user_input)
    prediction = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]
    print(f"Sample {i}: Pred={prediction}, Probas={proba}, Confidence={proba[prediction]}")
