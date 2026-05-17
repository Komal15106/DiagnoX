"""
main.py
=======
Entry point for the Multiple Disease Prediction System Streamlit app.
Run with:  streamlit run main.py
"""

import pickle
import os
import numpy as np
import pandas as pd
import streamlit as st
from disease_data import DISEASE_DETAILS

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiagnoX — Multiple Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS  (matches DiagnoX brand colours from style guide)
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Nunito:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background: linear-gradient(160deg, #0C2340 0%, #123A5E 100%);
  }
  [data-testid="stSidebar"] * { color: #E0F7FA !important; }
  [data-testid="stSidebar"] .stSelectbox label { font-weight: 700 !important; }

  /* Primary buttons */
  .stButton > button {
      background: linear-gradient(135deg, #17C3CE, #0EA5B0);
      color: white !important;
      border: none;
      border-radius: 100px;
      padding: 10px 28px;
      font-weight: 800;
      font-size: .95rem;
      font-family: 'Nunito', sans-serif;
      transition: transform .2s, box-shadow .2s;
      box-shadow: 0 4px 18px rgba(23,195,206,.35);
  }
  .stButton > button:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 28px rgba(23,195,206,.5);
  }

  /* Cards / containers */
  .result-card {
      background: #fff;
      border-radius: 22px;
      padding: 28px 32px;
      border: 1.5px solid #CBE9EF;
      box-shadow: 0 8px 32px rgba(23,195,206,.12);
      margin-top: 20px;
  }
  .low-risk  { border-left: 6px solid #2DD4B0; }
  .high-risk { border-left: 6px solid #FF6F61; }
  .detected  { border-left: 6px solid #FFB547; }

  h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: #0C2340; }

  .disclaimer-box {
      background: #FFF8EC;
      border: 1.5px solid #FFB547;
      border-radius: 14px;
      padding: 14px 20px;
      font-size: .85rem;
      color: #7A5C00;
      margin-top: 16px;
  }

  .tip-list li { margin-bottom: 6px; line-height: 1.65; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTS — Precautions & Tips
# ─────────────────────────────────────────────────────────
DIABETES_PRECAUTIONS = [
    "🥗  Reduce refined sugar and high-glycaemic foods from your diet.",
    "🚶  Aim for at least 30 minutes of moderate exercise daily.",
    "💧  Drink 8–10 glasses of water per day to support kidney function.",
    "📊  Monitor your blood glucose regularly and log the readings.",
    "⚖️  Maintain a healthy BMI through balanced nutrition.",
    "🩺  Schedule a consultation with an endocrinologist for personalised guidance.",
]

HEART_PRECAUTIONS = [
    "🚭  Avoid smoking and limit alcohol to reduce cardiovascular strain.",
    "🧂  Follow a low-sodium, low-saturated-fat diet.",
    "❤️  Monitor your blood pressure and cholesterol levels weekly.",
    "🏃  Engage in 150+ minutes of aerobic activity per week.",
    "😴  Ensure 7–8 hours of quality sleep to reduce cardiac stress.",
    "🩺  Schedule a cardiology consultation promptly.",
]

LOW_RISK_TIPS = [
    "✅  Continue your current healthy lifestyle habits.",
    "📅  Get routine annual health check-ups.",
    "🥦  Maintain a balanced, whole-foods diet.",
    "🏋️  Stay physically active — at least 150 min/week.",
]

DISCLAIMER = (
    "⚠️  **Medical Disclaimer:** This tool is for informational and educational purposes only. "
    "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider with any health concerns."
)

# Symptom-level precautions for common predicted diseases
SYMPTOM_PRECAUTIONS: dict = {
    "Fungal infection":       ["Keep skin clean and dry.", "Use antifungal powder/cream.", "Avoid sharing towels.", "Wear breathable cotton clothing."],
    "Allergy":                ["Identify and avoid allergens.", "Keep antihistamines handy.", "Use air purifiers indoors.", "Consult an allergist."],
    "GERD":                   ["Avoid spicy/acidic foods.", "Eat smaller meals.", "Do not lie down within 2 hours of eating.", "Elevate head while sleeping."],
    "Chronic cholestasis":    ["Avoid fatty foods.", "Take prescribed bile acid supplements.", "Monitor liver enzymes regularly.", "Stay well-hydrated."],
    "Drug Reaction":          ["Stop the suspected medication immediately.", "Consult your prescribing doctor.", "Drink plenty of fluids.", "Monitor for worsening rash or breathing difficulty."],
    "Peptic ulcer disease":   ["Avoid NSAIDs and alcohol.", "Eat small frequent meals.", "Reduce stress levels.", "Complete full antibiotic course if H. pylori positive."],
    "AIDS":                   ["Adhere to ART medication schedule.", "Practice safe sex.", "Regular CD4 / viral load monitoring.", "Maintain nutrition and mental health support."],
    "Diabetes":               DIABETES_PRECAUTIONS,
    "Gastroenteritis":        ["Stay hydrated with oral rehydration solution.", "Eat bland foods (BRAT diet).", "Avoid dairy temporarily.", "Rest and monitor for dehydration signs."],
    "Bronchial Asthma":       ["Identify and avoid triggers.", "Keep rescue inhaler accessible.", "Monitor peak flow readings.", "Follow asthma action plan."],
    "Hypertension":           ["Reduce sodium intake to <2g/day.", "Engage in regular aerobic exercise.", "Limit alcohol and caffeine.", "Take BP medications as prescribed."],
    "Migraine":               ["Track triggers in a headache diary.", "Stay hydrated and maintain sleep schedule.", "Reduce screen time.", "Consult neurologist for prophylactic treatment."],
    "Cervical spondylosis":   ["Practice neck-stretching exercises.", "Use an ergonomic pillow.", "Avoid prolonged screen time.", "Physiotherapy sessions recommended."],
    "Paralysis (brain hemorrhage)": ["Immediate emergency care required.", "Monitor consciousness level.", "Rehabilitation therapy post-stabilisation.", "Control blood pressure strictly."],
    "Jaundice":               ["Rest and avoid hepatotoxic substances.", "Stay well-hydrated.", "Eat light, easily digestible foods.", "Monitor bilirubin levels regularly."],
    "Malaria":                ["Complete full anti-malarial course.", "Use mosquito nets and repellents.", "Stay hydrated and rest.", "Monitor for severe symptoms."],
    "Chicken pox":            ["Avoid scratching blisters.", "Use calamine lotion for relief.", "Isolate to prevent spread.", "Monitor for secondary bacterial infection."],
    "Dengue":                 ["Hydrate aggressively with fluids.", "Monitor platelet count.", "Avoid aspirin/NSAIDs.", "Seek hospital care if platelet count drops sharply."],
    "Typhoid":                ["Complete the full antibiotic course.", "Drink only boiled or bottled water.", "Eat freshly cooked, hygienic food.", "Rest and monitor temperature."],
    "Hepatitis A":            ["Rest and avoid alcohol.", "Eat small, nutritious meals.", "Maintain strict hand hygiene.", "Monitor liver function tests."],
    "Hepatitis B":            ["Get vaccinated if not already.", "Avoid sharing needles or personal care items.", "Use protection during sex.", "Regular HBsAg monitoring."],
    "Hepatitis C":            ["Direct-acting antiviral treatment available.", "Avoid alcohol completely.", "Regular viral load monitoring.", "Consult a hepatologist."],
    "Hepatitis D":            ["Only occurs with Hepatitis B co-infection.", "Hepatitis B vaccination provides indirect protection.", "Interferon therapy may be advised.", "Regular specialist follow-up."],
    "Hepatitis E":            ["Drink safe, treated water.", "Avoid uncooked shellfish.", "Rest and stay hydrated.", "Especially critical to manage if pregnant."],
    "Alcoholic hepatitis":    ["Stop alcohol consumption immediately.", "Nutritional support is essential.", "Monitor liver enzymes.", "Seek addiction counselling."],
    "Tuberculosis":           ["Complete full 6-month DOTS regimen.", "Isolate during infectious phase.", "Ensure good ventilation at home.", "Notify contacts for screening."],
    "Common Cold":            ["Rest and stay hydrated.", "Use saline nasal spray.", "Take vitamin C supplements.", "Avoid close contact to prevent spreading."],
    "Pneumonia":              ["Complete full antibiotic course.", "Rest and drink plenty of fluids.", "Use a humidifier.", "Seek emergency care if breathing worsens."],
    "Heart attack":           HEART_PRECAUTIONS,
    "Varicose veins":         ["Elevate legs when resting.", "Wear compression stockings.", "Exercise regularly to improve circulation.", "Avoid prolonged standing."],
    "Hypothyroidism":         ["Take levothyroxine as prescribed.", "Monitor TSH levels every 6 months.", "Maintain healthy weight.", "Avoid calcium-rich foods near medication time."],
    "Hyperthyroidism":        ["Take anti-thyroid medications as prescribed.", "Avoid high-iodine foods.", "Monitor heart rate and weight.", "Radioiodine therapy may be an option."],
    "Hypoglycemia":           ["Carry fast-acting glucose (sugar tablets/juice).", "Eat regular, balanced meals.", "Avoid skipping meals.", "Monitor blood sugar regularly."],
    "Osteoarthritis":         ["Low-impact exercise (swimming, cycling).", "Maintain healthy weight.", "Use joint-protective devices.", "Physiotherapy and pain management."],
    "Arthritis":              ["Anti-inflammatory medication as prescribed.", "Joint-friendly exercise.", "Hot/cold therapy for pain.", "Occupational therapy for daily activities."],
    "(Vertigo) Paroxysmal Positional Vertigo": ["Epley manoeuvre (performed by physiotherapist).", "Avoid sudden head movements.", "Use handrails on stairs.", "Vestibular rehabilitation therapy."],
    "Acne":                   ["Gentle cleansing twice daily.", "Avoid touching or squeezing pimples.", "Use non-comedogenic products.", "Consult dermatologist for persistent acne."],
    "Urinary tract infection": ["Drink 8+ glasses of water daily.", "Complete full antibiotic course.", "Urinate after intercourse.", "Avoid irritants like bubble baths."],
    "Psoriasis":              ["Moisturise daily to reduce flaking.", "Avoid skin trauma and sunburn.", "Use prescribed topical corticosteroids.", "Consult dermatologist for biologic therapy."],
    "Impetigo":               ["Keep affected area clean and covered.", "Complete full antibiotic course.", "Avoid touching or scratching lesions.", "Maintain strict hand hygiene."],
}
DEFAULT_PRECAUTIONS = ["Consult a certified healthcare professional promptly.", "Rest and stay well-hydrated.", "Monitor symptoms and seek emergency care if they worsen."]

ALL_SYMPTOMS = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
    "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue",
    "muscle_wasting", "vomiting", "burning_micturition", "spotting_urination", "fatigue",
    "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss",
    "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough",
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration",
    "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea",
    "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain",
    "diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure",
    "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision",
    "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose",
    "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements",
    "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness",
    "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels",
    "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "excessive_hunger",
    "extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain",
    "muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements",
    "loss_of_balance", "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort",
    "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_(typhos)",
    "depression", "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body",
    "belly_pain", "abnormal_menstruation", "dischromic_patches", "watering_from_eyes", "increased_appetite",
    "polyuria", "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration",
    "visual_disturbances", "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", "stomach_bleeding",
    "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload.1", "blood_in_sputum", "prominent_veins_on_calf",
    "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring",
    "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister",
    "red_sore_around_nose", "yellow_crust_ooze",
]

# ─────────────────────────────────────────────────────────
# MODEL LOADING (cached — loads once per session)
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all serialised .pkl models from the models/ directory."""
    models = {}
    model_files = {
        "diabetes_model":  "models/diabetes_model.pkl",
        "diabetes_scaler": "models/diabetes_scaler.pkl",
        "heart_model":     "models/heart_model.pkl",
        "heart_scaler":    "models/heart_scaler.pkl",
        "symptom_model":   "models/symptom_model.pkl",
        "symptom_encoder": "models/symptom_encoder.pkl",
        "symptom_columns": "models/symptom_columns.pkl",
    }
    for key, path in model_files.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
        else:
            models[key] = None
    return models

models = load_models()

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def model_ready(name: str) -> bool:
    return models.get(name) is not None

def render_result_card(label: str, probability: float, precautions: list, risk_class: str, disease_name: str = None):
    """Render a styled result card with probability bar and precaution list."""
    color_map = {"low-risk": "#2DD4B0", "high-risk": "#FF6F61", "detected": "#17C3CE"}
    icon_map  = {"low-risk": "✅", "high-risk": "🔴", "detected": "🟡"}
    color = color_map.get(risk_class, "#17C3CE")
    icon  = icon_map.get(risk_class, "ℹ️")

    st.markdown(f"""
    <div class="result-card {risk_class}">
      <h2 style="color:{color}; margin-bottom:4px;">{icon} {label}</h2>
      <p style="color:#5A7A8A; font-size:.9rem; margin-bottom:12px;">ML Prediction Confidence</p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(int(probability * 100), 100))
    st.caption(f"Model confidence: **{probability*100:.1f}%**")

    # Enhanced Disease Info
    info = DISEASE_DETAILS.get(disease_name, {})
    if info:
        st.subheader("💡 Condition Overview")
        st.write(info.get("desc", ""))
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**💊 Suggested Medication**\n\n{info.get('meds')}")
        with c2:
            st.warning(f"**🥗 Lifestyle Changes**\n\n{info.get('lifestyle')}")
            
        st.success(f"**👨‍⚕️ Recommended Specialist:** {info.get('doctor')}")

    st.subheader("📋 Recommended Precautions")
    html_items = "".join(f"<li>{p}</li>" for p in precautions)
    st.markdown(f"<ul class='tip-list'>{html_items}</ul>", unsafe_allow_html=True)

    st.markdown(f"<div class='disclaimer-box'>{DISCLAIMER}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
      <div style='font-family:Syne,sans-serif; font-size:1.7rem; font-weight:800; color:#17C3CE;'>
        🩺 DiagnoX
      </div>
      <div style='font-size:.8rem; color:#A0C4CC; margin-top:4px;'>
        AI-Powered Health Assistant
      </div>
    </div>
    <hr style='border:1px solid #1E4D7B; margin: 10px 0 20px;'>
    """, unsafe_allow_html=True)

    selected = st.selectbox(
        "🔍 Select Prediction Mode",
        options=[
            "🏠  Home",
            "🩸  Diabetes Prediction",
            "❤️  Heart Disease Prediction",
            "🌡️  Symptom Checker",
            "ℹ️  How It Works",
        ],
        index=0,
    )

    st.markdown("""
    <hr style='border:1px solid #1E4D7B; margin: 20px 0 14px;'>
    <div style='font-size:.75rem; color:#5A7A8A; text-align:center; line-height:1.6;'>
      Powered by Scikit-Learn SVM &amp; Random Forest<br>
      Models trained on validated medical datasets<br><br>
      <b style='color:#FF6F61;'>Not a substitute for medical advice.</b>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────
if "Home" in selected:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#E8FAFB,#D6F5F7); border-radius:24px; padding:48px 40px 36px; margin-bottom:28px;'>
      <div style='font-size:.75rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#0EA5B0;margin-bottom:10px;'>
        ● AI-Powered Health Screening
      </div>
      <h1 style='font-size:2.6rem; line-height:1.15; margin-bottom:14px; color:#0C2340;'>
        Your Smart <em style='color:#0EA5B0;font-style:normal;'>Health Assistant</em><br>All in One Place
      </h1>
      <p style='color:#5A7A8A; font-size:1.05rem; line-height:1.75; max-width:560px;'>
        DiagnoX uses machine-learning models trained on validated medical datasets to give you a 
        personalised risk assessment — whether you have everyday symptoms or lab-report values.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='result-card' style='border-left:6px solid #FF6F61;'>
          <div style='font-size:2rem;margin-bottom:10px;'>🩸</div>
          <h3 style='color:#FF6F61;margin-bottom:6px;'>Diabetes Check</h3>
          <p style='color:#5A7A8A;font-size:.85rem;line-height:1.6;'>
            Enter 8 clinical values (Glucose, BMI, Blood Pressure…) for an SVM-powered 
            diabetes risk assessment with probability.
          </p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='result-card' style='border-left:6px solid #17C3CE;'>
          <div style='font-size:2rem;margin-bottom:10px;'>❤️</div>
          <h3 style='color:#17C3CE;margin-bottom:6px;'>Heart Disease Check</h3>
          <p style='color:#5A7A8A;font-size:.85rem;line-height:1.6;'>
            Provide 13 cardiovascular indicators (Cholesterol, ECG, Angina…) for a 
            cardiac risk prediction with tailored advice.
          </p>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='result-card' style='border-left:6px solid #FFB547;'>
          <div style='font-size:2rem;margin-bottom:10px;'>🌡️</div>
          <h3 style='color:#FFB547;margin-bottom:6px;'>Symptom Checker</h3>
          <p style='color:#5A7A8A;font-size:.85rem;line-height:1.6;'>
            Select 1–10 symptoms from 130+ options. Our Random Forest model identifies 
            the most likely condition from 40+ diseases.
          </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Symptom Accuracy",  "~92%",  "Random Forest")
    m2.metric("🔬 Diseases Covered",   "40+",   "Symptom mode")
    m3.metric("📋 Symptoms Supported", "130+",  "Multi-select")
    m4.metric("⚡ Prediction Time",    "<1 sec", "In-memory models")

    st.markdown(f"<div class='disclaimer-box'>{DISCLAIMER}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# PAGE: DIABETES PREDICTION
# ─────────────────────────────────────────────────────────
elif "Diabetes" in selected:
    st.title("🩸 Diabetes Risk Prediction")
    st.markdown("Enter your clinical values below. All fields are required. "
                "Hover over labels for normal reference ranges.")

    if not model_ready("diabetes_model"):
        st.error("⚠️ Diabetes model not found. Please run `python train_models.py` first.")
        st.stop()

    with st.form("diabetes_form"):
        st.subheader("Clinical Input Values")

        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1,
                                          help="Number of times pregnant (0 for males — set to 0)")
            glucose     = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=110,
                                          help="Plasma glucose concentration (fasting normal: 70–100 mg/dL)")
            bp          = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=72,
                                          help="Diastolic blood pressure (normal: 60–80 mm Hg)")
            skin        = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20,
                                          help="Triceps skin fold thickness (typical: 15–35 mm)")
        with col2:
            insulin     = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=80,
                                          help="2-hour serum insulin (normal fasting: 2–25 μU/mL)")
            bmi         = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=26.5,
                                          help="Body Mass Index (healthy range: 18.5–24.9)")
            dpf         = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.35,
                                          help="Likelihood of diabetes based on family history (typical: 0.08–2.42)")
            age         = st.number_input("Age (years)", min_value=1, max_value=120, value=30,
                                          help="Patient's age in years")

        submitted = st.form_submit_button("🔍 Predict Diabetes Risk", use_container_width=True)

    if submitted:
        with st.spinner("Running SVM inference…"):
            try:
                user_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
                scaled     = models["diabetes_scaler"].transform(user_input)
                prediction = models["diabetes_model"].predict(scaled)[0]
                probability = models["diabetes_model"].predict_proba(scaled)[0][prediction]

                if prediction == 1:
                    render_result_card(
                        "⚠️  High Risk of Diabetes", probability,
                        DIABETES_PRECAUTIONS, "high-risk", "High Risk of Diabetes"
                    )
                else:
                    render_result_card(
                        "✅  Low Risk of Diabetes", probability,
                        LOW_RISK_TIPS, "low-risk"
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ─────────────────────────────────────────────────────────
# PAGE: HEART DISEASE PREDICTION
# ─────────────────────────────────────────────────────────
elif "Heart" in selected:
    st.title("❤️ Heart Disease Risk Prediction")
    st.markdown("Fill in all 13 cardiovascular indicators. Dropdown fields have "
                "labelled options to guide your selection.")

    if not model_ready("heart_model"):
        st.error("⚠️ Heart disease model not found. Please run `python train_models.py` first.")
        st.stop()

    with st.form("heart_form"):
        st.subheader("Cardiovascular Input Values")

        col1, col2 = st.columns(2)
        with col1:
            age      = st.number_input("Age (years)", min_value=1, max_value=120, value=50,
                                       help="Patient's age in years")
            sex      = st.selectbox("Sex", options=["Male (1)", "Female (0)"],
                                    help="Biological sex (Male=1, Female=0)")
            cp       = st.selectbox("Chest Pain Type",
                                    options=["0 — Typical Angina", "1 — Atypical Angina",
                                             "2 — Non-Anginal Pain", "3 — Asymptomatic"],
                                    help="Type of chest pain experienced")
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120,
                                       help="Resting BP on admission (normal: <120 mm Hg)")
            chol     = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=220,
                                       help="Serum cholesterol in mg/dL (desirable: <200)")
            fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL",
                                    options=["No (0)", "Yes (1)"],
                                    help="1 = True if fasting blood sugar > 120 mg/dL")
            restecg  = st.selectbox("Resting ECG Results",
                                    options=["0 — Normal", "1 — ST-T Wave Abnormality",
                                             "2 — Left Ventricular Hypertrophy"],
                                    help="Resting electrocardiographic results")

        with col2:
            thalach  = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=250, value=150,
                                       help="Maximum heart rate during exercise test")
            exang    = st.selectbox("Exercise-Induced Angina",
                                    options=["No (0)", "Yes (1)"],
                                    help="Chest pain during exercise?")
            oldpeak  = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0,
                                       help="ST depression induced by exercise relative to rest")
            slope    = st.selectbox("Slope of Peak Exercise ST Segment",
                                    options=["0 — Upsloping", "1 — Flat", "2 — Downsloping"],
                                    help="Slope of the peak exercise ST segment")
            ca       = st.selectbox("Major Vessels Coloured by Fluoroscopy",
                                    options=["0", "1", "2", "3"],
                                    help="Number of major vessels coloured by fluoroscopy (0–3)")
            thal     = st.selectbox("Thalassemia",
                                    options=["1 — Normal", "2 — Fixed Defect", "3 — Reversible Defect"],
                                    help="Thalassemia type")

        submitted_h = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)

    if submitted_h:
        with st.spinner("Running SVM inference…"):
            try:
                sex_val    = 1 if "Male" in sex else 0
                cp_val     = int(cp.split(" ")[0])
                fbs_val    = 1 if "Yes" in fbs else 0
                ecg_val    = int(restecg.split(" ")[0])
                exang_val  = 1 if "Yes" in exang else 0
                slope_val  = int(slope.split(" ")[0])
                ca_val     = int(ca)
                thal_val   = int(thal.split(" ")[0])

                user_input = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                                        ecg_val, thalach, exang_val, oldpeak,
                                        slope_val, ca_val, thal_val]])
                scaled     = models["heart_scaler"].transform(user_input)
                prediction = models["heart_model"].predict(scaled)[0]
                probability = models["heart_model"].predict_proba(scaled)[0][prediction]

                if prediction == 1:
                    render_result_card(
                        "⚠️  High Risk of Heart Disease", probability,
                        HEART_PRECAUTIONS, "high-risk", "High Risk of Heart Disease"
                    )
                else:
                    render_result_card(
                        "✅  Low Risk of Heart Disease", probability,
                        LOW_RISK_TIPS, "low-risk"
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ─────────────────────────────────────────────────────────
# PAGE: SYMPTOM CHECKER
# ─────────────────────────────────────────────────────────
elif "Symptom" in selected:
    st.title("🌡️ General Symptom Checker")
    st.markdown("Select 1–10 symptoms that best describe what you are currently experiencing. "
                "The model will identify the most likely condition from 40+ diseases.")

    if not model_ready("symptom_model"):
        st.error("⚠️ Symptom model not found. Please run `python train_models.py` first.")
        st.stop()

    symptom_display = [s.replace("_", " ").title() for s in ALL_SYMPTOMS]
    symptom_map     = dict(zip(symptom_display, ALL_SYMPTOMS))

    selected_display = st.multiselect(
        "🔍 Search and select your symptoms:",
        options=symptom_display,
        max_selections=10,
        help="Start typing to search. Select up to 10 symptoms.",
    )

    col_btn, col_clear = st.columns([1, 4])
    predict_clicked = col_btn.button("🔍 Check Symptoms", use_container_width=True)

    if predict_clicked:
        if len(selected_display) < 5:
            st.warning("⚠️ Please select at least 5 symptoms for analysis.")
        else:
            with st.spinner("Running Random Forest inference…"):
                try:
                    # Build binary feature vector in correct column order
                    sym_cols = (models.get("symptom_columns") or ALL_SYMPTOMS)[:38]
                    selected_raw = [symptom_map[s] for s in selected_display]
                    feature_vec = np.array([[1 if col in selected_raw else 0
                                             for col in sym_cols]])

                    pred_idx   = models["symptom_model"].predict(feature_vec)[0]
                    disease    = models["symptom_encoder"].inverse_transform([pred_idx])[0]
                    proba_arr  = models["symptom_model"].predict_proba(feature_vec)[0]
                    confidence = proba_arr[pred_idx]

                    precautions = SYMPTOM_PRECAUTIONS.get(disease, DEFAULT_PRECAUTIONS)

                    render_result_card(
                        f"Detected Condition: {disease}",
                        confidence,
                        precautions,
                        "detected",
                        disease
                    )

                    # Show top 3 alternate predictions
                    with st.expander("🔎 See top 3 alternate predictions"):
                        top3_idx = np.argsort(proba_arr)[::-1][:3]
                        for rank, idx in enumerate(top3_idx, 1):
                            dis = models["symptom_encoder"].inverse_transform([idx])[0]
                            prob = proba_arr[idx]
                            st.markdown(f"**{rank}.** {dis} — {prob*100:.1f}%")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# ─────────────────────────────────────────────────────────
# PAGE: HOW IT WORKS
# ─────────────────────────────────────────────────────────
elif "How" in selected:
    st.title("ℹ️ How DiagnoX Works")

    st.markdown("""
    DiagnoX is a machine-learning-powered health assistant. It uses pre-trained models to assess
    disease risk based on your inputs. Here's a plain-language explanation of the engine under the hood.
    """)

    with st.expander("🔬 The Machine Learning Models", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.markdown("**🩸 Diabetes**\n\nAlgorithm: SVM (RBF kernel)\n\nAccuracy: 78–85%\n\nFeatures: 8 numerical clinical values")
        col2.markdown("**❤️ Heart Disease**\n\nAlgorithm: SVM (Linear kernel)\n\nAccuracy: 78–85%\n\nFeatures: 13 cardiovascular indicators")
        col3.markdown("**🌡️ Symptom Checker**\n\nAlgorithm: Random Forest (100 trees)\n\nAccuracy: ~92%\n\nFeatures: 130+ binary symptom flags")

    with st.expander("⚙️ Data Flow — From Input to Prediction"):
        st.markdown("""
        1. **User enters inputs** via form (numbers, dropdowns, or symptom selection).
        2. **Input validation** — ranges checked client-side.
        3. **Preprocessing** — numerical inputs are scaled using the same `StandardScaler` fitted during training.
        4. **Inference** — `model.predict()` returns the class label; `model.predict_proba()` returns confidence.
        5. **Result rendering** — risk label, confidence bar, and tailored precautions displayed.
        """)

    with st.expander("🧬 Training Datasets"):
        st.markdown("""
        | Module | Dataset | Records | Source |
        |--------|---------|---------|--------|
        | Diabetes | Pima Indians Diabetes | 768 | UCI / Kaggle |
        | Heart Disease | Cleveland Heart Disease | 303 | UCI / Kaggle |
        | Symptom Checker | Disease-Symptom Mapping | 4,920 | Kaggle (itachi9604) |
        """)

    with st.expander("⚠️ Limitations & Disclaimer"):
        st.markdown(f"""
        {DISCLAIMER}

        - Model accuracy is 78–92% — errors are possible.
        - Models are trained on specific population datasets and may not generalise universally.
        - Results should never replace a physical examination or lab tests.
        - This tool does not collect, store, or transmit any personal data.
        """)