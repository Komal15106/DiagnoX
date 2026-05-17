"""
app.py — Flask backend for DiagnoX
Run with: python app.py
Opens at: http://localhost:5000
"""
from flask import Flask, request, jsonify, send_from_directory
from disease_data import DISEASE_DETAILS
import pickle, os, numpy as np

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
print("--- Starting DiagnoX Server ---")

# ── MODEL LOADING ──────────────────────────────────────────
def load_models():
    m = {}
    files = {
        "diabetes_model":  "models/diabetes_model.pkl",
        "diabetes_scaler": "models/diabetes_scaler.pkl",
        "heart_model":     "models/heart_model.pkl",
        "heart_scaler":    "models/heart_scaler.pkl",
        "symptom_model":   "models/symptom_model.pkl",
        "symptom_encoder": "models/symptom_encoder.pkl",
        "symptom_columns": "models/symptom_columns.pkl",
    }
    for key, path in files.items():
        fp = os.path.join(BASE, path)
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                m[key] = pickle.load(f)
        else:
            m[key] = None
    return m

models = load_models()

ALL_SYMPTOMS = [
    "itching","skin_rash","nodal_skin_eruptions","continuous_sneezing","shivering",
    "chills","joint_pain","stomach_pain","acidity","ulcers_on_tongue",
    "muscle_wasting","vomiting","burning_micturition","spotting_urination","fatigue",
    "weight_gain","anxiety","cold_hands_and_feets","mood_swings","weight_loss",
    "restlessness","lethargy","patches_in_throat","irregular_sugar_level","cough",
    "high_fever","sunken_eyes","breathlessness","sweating","dehydration",
    "indigestion","headache","yellowish_skin","dark_urine","nausea",
    "loss_of_appetite","pain_behind_the_eyes","back_pain","constipation","abdominal_pain",
    "diarrhoea","mild_fever","yellow_urine","yellowing_of_eyes","acute_liver_failure",
    "fluid_overload","swelling_of_stomach","swelled_lymph_nodes","malaise","blurred_and_distorted_vision",
    "phlegm","throat_irritation","redness_of_eyes","sinus_pressure","runny_nose",
    "congestion","chest_pain","weakness_in_limbs","fast_heart_rate","pain_during_bowel_movements",
    "pain_in_anal_region","bloody_stool","irritation_in_anus","neck_pain","dizziness",
    "cramps","bruising","obesity","swollen_legs","swollen_blood_vessels",
    "puffy_face_and_eyes","enlarged_thyroid","brittle_nails","swollen_extremeties","excessive_hunger",
    "extra_marital_contacts","drying_and_tingling_lips","slurred_speech","knee_pain","hip_joint_pain",
    "muscle_weakness","stiff_neck","swelling_joints","movement_stiffness","spinning_movements",
    "loss_of_balance","unsteadiness","weakness_of_one_body_side","loss_of_smell","bladder_discomfort",
    "foul_smell_of_urine","continuous_feel_of_urine","passage_of_gases","internal_itching","toxic_look_(typhos)",
    "depression","irritability","muscle_pain","altered_sensorium","red_spots_over_body",
    "belly_pain","abnormal_menstruation","dischromic_patches","watering_from_eyes","increased_appetite",
    "polyuria","family_history","mucoid_sputum","rusty_sputum","lack_of_concentration",
    "visual_disturbances","receiving_blood_transfusion","receiving_unsterile_injections","coma","stomach_bleeding",
    "distention_of_abdomen","history_of_alcohol_consumption","fluid_overload.1","blood_in_sputum","prominent_veins_on_calf",
    "palpitations","painful_walking","pus_filled_pimples","blackheads","scurring",
    "skin_peeling","silver_like_dusting","small_dents_in_nails","inflammatory_nails","blister",
    "red_sore_around_nose","yellow_crust_ooze",
]

SYMPTOM_PRECAUTIONS = {
    "Fungal infection":["Keep skin clean and dry.","Use antifungal powder/cream.","Avoid sharing towels.","Wear breathable cotton clothing."],
    "Allergy":["Identify and avoid allergens.","Keep antihistamines handy.","Use air purifiers indoors.","Consult an allergist."],
    "GERD":["Avoid spicy/acidic foods.","Eat smaller meals.","Do not lie down within 2 hours of eating.","Elevate head while sleeping."],
    "Chronic cholestasis":["Avoid fatty foods.","Take prescribed bile acid supplements.","Monitor liver enzymes regularly.","Stay well-hydrated."],
    "Drug Reaction":["Stop the suspected medication immediately.","Consult your prescribing doctor.","Drink plenty of fluids.","Monitor for worsening symptoms."],
    "Peptic ulcer disease":["Avoid NSAIDs and alcohol.","Eat small frequent meals.","Reduce stress levels.","Complete full antibiotic course if H. pylori positive."],
    "AIDS":["Adhere to ART medication schedule.","Practice safe sex.","Regular CD4 / viral load monitoring.","Maintain nutrition and mental health support."],
    "Diabetes":["Reduce refined sugar intake.","Aim for 30 minutes of exercise daily.","Drink 8-10 glasses of water per day.","Monitor blood glucose regularly."],
    "Gastroenteritis":["Stay hydrated with oral rehydration solution.","Eat bland foods (BRAT diet).","Avoid dairy temporarily.","Rest and monitor for dehydration signs."],
    "Bronchial Asthma":["Identify and avoid triggers.","Keep rescue inhaler accessible.","Monitor peak flow readings.","Follow asthma action plan."],
    "Hypertension":["Reduce sodium intake to <2g/day.","Engage in regular aerobic exercise.","Limit alcohol and caffeine.","Take BP medications as prescribed."],
    "Migraine":["Track triggers in a headache diary.","Stay hydrated and maintain sleep schedule.","Reduce screen time.","Consult neurologist for prophylactic treatment."],
    "Cervical spondylosis":["Practice neck-stretching exercises.","Use an ergonomic pillow.","Avoid prolonged screen time.","Physiotherapy sessions recommended."],
    "Paralysis (brain hemorrhage)":["Immediate emergency care required.","Monitor consciousness level.","Rehabilitation therapy post-stabilisation.","Control blood pressure strictly."],
    "Jaundice":["Rest and avoid hepatotoxic substances.","Stay well-hydrated.","Eat light, easily digestible foods.","Monitor bilirubin levels regularly."],
    "Malaria":["Complete full anti-malarial course.","Use mosquito nets and repellents.","Stay hydrated and rest.","Monitor for severe symptoms."],
    "Chicken pox":["Avoid scratching blisters.","Use calamine lotion for relief.","Isolate to prevent spread.","Monitor for secondary bacterial infection."],
    "Dengue":["Hydrate aggressively with fluids.","Monitor platelet count.","Avoid aspirin/NSAIDs.","Seek hospital care if platelet count drops sharply."],
    "Typhoid":["Complete the full antibiotic course.","Drink only boiled or bottled water.","Eat freshly cooked, hygienic food.","Rest and monitor temperature."],
    "Hepatitis A":["Rest and avoid alcohol.","Eat small, nutritious meals.","Maintain strict hand hygiene.","Monitor liver function tests."],
    "Hepatitis B":["Get vaccinated if not already.","Avoid sharing needles or personal care items.","Use protection during sex.","Regular HBsAg monitoring."],
    "Hepatitis C":["Direct-acting antiviral treatment available.","Avoid alcohol completely.","Regular viral load monitoring.","Consult a hepatologist."],
    "Hepatitis D":["Only occurs with Hepatitis B co-infection.","Hepatitis B vaccination provides indirect protection.","Interferon therapy may be advised.","Regular specialist follow-up."],
    "Hepatitis E":["Drink safe, treated water.","Avoid uncooked shellfish.","Rest and stay hydrated.","Especially critical to manage if pregnant."],
    "Alcoholic hepatitis":["Stop alcohol consumption immediately.","Nutritional support is essential.","Monitor liver enzymes.","Seek addiction counselling."],
    "Tuberculosis":["Complete full 6-month DOTS regimen.","Isolate during infectious phase.","Ensure good ventilation at home.","Notify contacts for screening."],
    "Common Cold":["Rest and stay hydrated.","Use saline nasal spray.","Take vitamin C supplements.","Avoid close contact to prevent spreading."],
    "Pneumonia":["Complete full antibiotic course.","Rest and drink plenty of fluids.","Use a humidifier.","Seek emergency care if breathing worsens."],
    "Varicose veins":["Elevate legs when resting.","Wear compression stockings.","Exercise regularly to improve circulation.","Avoid prolonged standing."],
    "Hypothyroidism":["Take levothyroxine as prescribed.","Monitor TSH levels every 6 months.","Maintain healthy weight.","Avoid calcium-rich foods near medication time."],
    "Hyperthyroidism":["Take anti-thyroid medications as prescribed.","Avoid high-iodine foods.","Monitor heart rate and weight.","Radioiodine therapy may be an option."],
    "Hypoglycemia":["Carry fast-acting glucose (sugar tablets/juice).","Eat regular, balanced meals.","Avoid skipping meals.","Monitor blood sugar regularly."],
    "Osteoarthritis":["Low-impact exercise (swimming, cycling).","Maintain healthy weight.","Use joint-protective devices.","Physiotherapy and pain management."],
    "Arthritis":["Anti-inflammatory medication as prescribed.","Joint-friendly exercise.","Hot/cold therapy for pain.","Occupational therapy for daily activities."],
    "(Vertigo) Paroxysmal Positional Vertigo":["Epley manoeuvre (performed by physiotherapist).","Avoid sudden head movements.","Use handrails on stairs.","Vestibular rehabilitation therapy."],
    "Acne":["Gentle cleansing twice daily.","Avoid touching or squeezing pimples.","Use non-comedogenic products.","Consult dermatologist for persistent acne."],
    "Urinary tract infection":["Drink 8+ glasses of water daily.","Complete full antibiotic course.","Urinate after intercourse.","Avoid irritants like bubble baths."],
    "Psoriasis":["Moisturise daily to reduce flaking.","Avoid skin trauma and sunburn.","Use prescribed topical corticosteroids.","Consult dermatologist for biologic therapy."],
    "Impetigo":["Keep affected area clean and covered.","Complete full antibiotic course.","Avoid touching or scratching lesions.","Maintain strict hand hygiene."],
}
DEFAULT_PRECAUTIONS = ["Consult a certified healthcare professional promptly.","Rest and stay well-hydrated.","Monitor symptoms and seek emergency care if they worsen."]
DIABETES_PRECAUTIONS = ["🥗 Reduce refined sugar and high-glycaemic foods.","🚶 Aim for at least 30 minutes of moderate exercise daily.","💧 Drink 8-10 glasses of water per day.","📊 Monitor your blood glucose regularly.","⚖️ Maintain a healthy BMI.","🩺 Consult an endocrinologist for personalised guidance."]
HEART_PRECAUTIONS = ["🚭 Avoid smoking and limit alcohol.","🧂 Follow a low-sodium, low-saturated-fat diet.","❤️ Monitor blood pressure and cholesterol weekly.","🏃 Engage in 150+ minutes of aerobic activity per week.","😴 Ensure 7-8 hours of quality sleep.","🩺 Schedule a cardiology consultation promptly."]
LOW_RISK_TIPS = ["✅ Continue your current healthy lifestyle habits.","📅 Get routine annual health check-ups.","🥦 Maintain a balanced, whole-foods diet.","🏋️ Stay physically active — at least 150 min/week."]

# ── ROUTES ─────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(BASE, 'index.html')

@app.route('/how-it-works')
def how_it_works():
    return send_from_directory(BASE, 'how_it_works.html')

@app.route('/symptom')
def symptom_page():
    return send_from_directory(BASE, 'symptom.html')

@app.route('/diabetes')
def diabetes_page():
    return send_from_directory(BASE, 'diabetes.html')

@app.route('/heart')
def heart_page():
    return send_from_directory(BASE, 'heart.html')

@app.route('/history')
def history_page():
    return send_from_directory(BASE, 'history.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(BASE, 'static'), filename)

@app.route('/predict/symptom', methods=['POST'])
def predict_symptom():
    try:
        data = request.get_json()
        selected = data.get('symptoms', [])
        if len(selected) < 5:
            return jsonify({"error": "Please select at least 5 symptoms for a reliable calculation."}), 400
        sym_cols = (models.get("symptom_columns") or ALL_SYMPTOMS)[:38]
        feature_vec = np.array([[1 if col in selected else 0 for col in sym_cols]])
        pred_idx = int(models["symptom_model"].predict(feature_vec)[0])
        disease = models["symptom_encoder"].inverse_transform([pred_idx])[0]
        proba_arr = models["symptom_model"].predict_proba(feature_vec)[0]
        confidence = float(proba_arr[pred_idx])
        top3_idx = np.argsort(proba_arr)[::-1][:3]
        top3 = [{"disease": models["symptom_encoder"].inverse_transform([int(i)])[0], "prob": float(proba_arr[i])} for i in top3_idx]
        precautions = SYMPTOM_PRECAUTIONS.get(disease, DEFAULT_PRECAUTIONS)
        info = DISEASE_DETAILS.get(disease, {})
        return jsonify({"disease": disease, "confidence": confidence, "precautions": precautions, "top3": top3, "risk": "detected", "info": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        d = request.get_json()
        user_input = np.array([[float(d['pregnancies']), float(d['glucose']), float(d['bp']),
                                float(d['skin']), float(d['insulin']), float(d['bmi']),
                                float(d['dpf']), float(d['age'])]])
        scaled = models["diabetes_scaler"].transform(user_input)
        prediction = int(models["diabetes_model"].predict(scaled)[0])
        probability = float(models["diabetes_model"].predict_proba(scaled)[0][prediction])
        risk = "high-risk" if prediction == 1 else "low-risk"
        label = "⚠️ High Risk of Diabetes" if prediction == 1 else "✅ Low Risk of Diabetes"
        precautions = DIABETES_PRECAUTIONS if prediction == 1 else LOW_RISK_TIPS
        info = DISEASE_DETAILS.get("High Risk of Diabetes", {}) if prediction == 1 else {}
        return jsonify({"label": label, "confidence": probability, "precautions": precautions, "risk": risk, "info": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        d = request.get_json()
        user_input = np.array([[float(d['age']), float(d['sex']), float(d['cp']),
                                float(d['trestbps']), float(d['chol']), float(d['fbs']),
                                float(d['restecg']), float(d['thalach']), float(d['exang']),
                                float(d['oldpeak']), float(d['slope']), float(d['ca']),
                                float(d['thal'])]])
        scaled = models["heart_scaler"].transform(user_input)
        prediction = int(models["heart_model"].predict(scaled)[0])
        probability = float(models["heart_model"].predict_proba(scaled)[0][prediction])
        risk = "high-risk" if prediction == 1 else "low-risk"
        label = "⚠️ Suspected Coronary Artery Disease" if prediction == 1 else "✅ Low Risk of Heart Disease"
        precautions = HEART_PRECAUTIONS if prediction == 1 else LOW_RISK_TIPS
        info = DISEASE_DETAILS.get("Suspected Coronary Artery Disease", {}) if prediction == 1 else {}
        return jsonify({"label": label, "confidence": probability, "precautions": precautions, "risk": risk, "info": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
