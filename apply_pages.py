"""
apply_pages.py
==============
Utility script to scaffold the Streamlit multi-page directory structure
and write individual page files that import and delegate to main.py helpers.

Run once during project setup:
    python apply_pages.py

This creates:
    pages/1_diabetes.py
    pages/2_heart_disease.py
    pages/3_symptom_checker.py
    pages/4_how_it_works.py
    .streamlit/config.toml
"""

import os

# ─────────────────────────────────────────────────────────
# Ensure directories exist
# ─────────────────────────────────────────────────────────
os.makedirs("pages", exist_ok=True)
os.makedirs(".streamlit", exist_ok=True)

# ─────────────────────────────────────────────────────────
# .streamlit/config.toml  — DiagnoX brand theme
# ─────────────────────────────────────────────────────────
CONFIG_TOML = """\
[theme]
primaryColor         = "#17C3CE"
backgroundColor      = "#F5FDFF"
secondaryBackgroundColor = "#EAF9FB"
textColor            = "#0C2340"
font                 = "sans serif"

[server]
headless            = true
enableCORS          = false
enableXsrfProtection = true
"""

with open(".streamlit/config.toml", "w") as f:
    f.write(CONFIG_TOML)
print("✔  Written: .streamlit/config.toml")


# ─────────────────────────────────────────────────────────
# Page files  (each page simply imports its show_ function)
# ─────────────────────────────────────────────────────────
PAGES = {
    "pages/1_🩸_Diabetes_Prediction.py": """\
\"\"\"Diabetes Prediction page — delegates to main.py show_diabetes_page().\"\"\"
import streamlit as st
import pickle, os, numpy as np

st.set_page_config(page_title="Diabetes Prediction | DiagnoX", page_icon="🩸", layout="wide")

# Re-use shared helpers from main
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import (
    load_models, render_result_card,
    DIABETES_PRECAUTIONS, LOW_RISK_TIPS, DISCLAIMER
)

st.markdown(\"\"\"
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Nunito:wght@600;700;800&display=swap');
  html,[class*='css']{font-family:'Nunito',sans-serif;}
  .stButton>button{background:linear-gradient(135deg,#17C3CE,#0EA5B0);color:#fff!important;border:none;
    border-radius:100px;padding:10px 28px;font-weight:800;}
  h1,h2,h3{font-family:'Syne',sans-serif!important;font-weight:800!important;color:#0C2340;}
  .result-card{background:#fff;border-radius:22px;padding:28px 32px;border:1.5px solid #CBE9EF;
    box-shadow:0 8px 32px rgba(23,195,206,.12);margin-top:20px;}
  .low-risk{border-left:6px solid #2DD4B0;} .high-risk{border-left:6px solid #FF6F61;}
  .disclaimer-box{background:#FFF8EC;border:1.5px solid #FFB547;border-radius:14px;
    padding:14px 20px;font-size:.85rem;color:#7A5C00;margin-top:16px;}
  .tip-list li{margin-bottom:6px;line-height:1.65;}
</style>\"\"\", unsafe_allow_html=True)

models = load_models()

st.title("🩸 Diabetes Risk Prediction")
st.markdown("Enter your clinical values. All 8 fields are required.")

if models.get("diabetes_model") is None:
    st.error("⚠️ Diabetes model not found. Run `python train_models.py` first.")
    st.stop()

with st.form("diab_form"):
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1, help="Set 0 if male")
        glucose     = st.number_input("Glucose (mg/dL)", 0, 300, 110, help="Fasting normal: 70–100")
        bp          = st.number_input("Blood Pressure (mm Hg)", 0, 200, 72, help="Diastolic normal: 60–80")
        skin        = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with c2:
        insulin     = st.number_input("Insulin (μU/mL)", 0, 900, 80, help="Fasting normal: 2–25")
        bmi         = st.number_input("BMI (kg/m²)", 0.0, 70.0, 26.5, help="Healthy: 18.5–24.9")
        dpf         = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.35)
        age         = st.number_input("Age (years)", 1, 120, 30)
    sub = st.form_submit_button("🔍 Predict Diabetes Risk", use_container_width=True)

if sub:
    with st.spinner("Running SVM inference…"):
        try:
            inp = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            sc  = models["diabetes_scaler"].transform(inp)
            pr  = models["diabetes_model"].predict(sc)[0]
            pb  = models["diabetes_model"].predict_proba(sc)[0][pr]
            render_result_card(
                ("⚠️  High Risk of Diabetes" if pr==1 else "✅  Low Risk of Diabetes"), pb,
                (DIABETES_PRECAUTIONS if pr==1 else LOW_RISK_TIPS),
                ("high-risk" if pr==1 else "low-risk")
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")
""",

    "pages/2_❤️_Heart_Disease_Prediction.py": """\
\"\"\"Heart Disease Prediction page.\"\"\"
import streamlit as st, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import load_models, render_result_card, HEART_PRECAUTIONS, LOW_RISK_TIPS

st.set_page_config(page_title="Heart Disease | DiagnoX", page_icon="❤️", layout="wide")
st.markdown(\"\"\"<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Nunito:wght@600;700;800&display=swap');
  html,[class*='css']{font-family:'Nunito',sans-serif;}
  .stButton>button{background:linear-gradient(135deg,#17C3CE,#0EA5B0);color:#fff!important;
    border:none;border-radius:100px;padding:10px 28px;font-weight:800;}
  h1,h2,h3{font-family:'Syne',sans-serif!important;font-weight:800!important;color:#0C2340;}
  .result-card{background:#fff;border-radius:22px;padding:28px 32px;border:1.5px solid #CBE9EF;
    box-shadow:0 8px 32px rgba(23,195,206,.12);margin-top:20px;}
  .low-risk{border-left:6px solid #2DD4B0;} .high-risk{border-left:6px solid #FF6F61;}
  .disclaimer-box{background:#FFF8EC;border:1.5px solid #FFB547;border-radius:14px;
    padding:14px 20px;font-size:.85rem;color:#7A5C00;margin-top:16px;}
  .tip-list li{margin-bottom:6px;line-height:1.65;}
\"\"\", unsafe_allow_html=True)

models = load_models()
st.title("❤️ Heart Disease Risk Prediction")
if models.get("heart_model") is None:
    st.error("⚠️ Heart model not found. Run `python train_models.py` first.")
    st.stop()

with st.form("heart_form"):
    c1, c2 = st.columns(2)
    with c1:
        age      = st.number_input("Age", 1, 120, 50)
        sex      = st.selectbox("Sex", ["Male (1)", "Female (0)"])
        cp       = st.selectbox("Chest Pain Type", ["0 — Typical Angina","1 — Atypical Angina","2 — Non-Anginal Pain","3 — Asymptomatic"])
        trestbps = st.number_input("Resting BP (mm Hg)", 80, 250, 120)
        chol     = st.number_input("Cholesterol (mg/dL)", 100, 600, 220)
        fbs      = st.selectbox("Fasting Blood Sugar >120", ["No (0)","Yes (1)"])
        restecg  = st.selectbox("Resting ECG", ["0 — Normal","1 — ST-T Abnormality","2 — LV Hypertrophy"])
    with c2:
        thalach  = st.number_input("Max Heart Rate (bpm)", 60, 250, 150)
        exang    = st.selectbox("Exercise Angina", ["No (0)","Yes (1)"])
        oldpeak  = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0)
        slope    = st.selectbox("ST Slope", ["0 — Upsloping","1 — Flat","2 — Downsloping"])
        ca       = st.selectbox("Major Vessels (0–3)", ["0","1","2","3"])
        thal     = st.selectbox("Thalassemia", ["1 — Normal","2 — Fixed Defect","3 — Reversible Defect"])
    sub = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)

if sub:
    with st.spinner("Running SVM inference…"):
        try:
            vals = [age, 1 if "Male" in sex else 0, int(cp[0]),
                    trestbps, chol, 1 if "Yes" in fbs else 0,
                    int(restecg[0]), thalach, 1 if "Yes" in exang else 0,
                    oldpeak, int(slope[0]), int(ca), int(thal[0])]
            inp = np.array([vals])
            sc  = models["heart_scaler"].transform(inp)
            pr  = models["heart_model"].predict(sc)[0]
            pb  = models["heart_model"].predict_proba(sc)[0][pr]
            render_result_card(
                ("⚠️  High Risk of Heart Disease" if pr==1 else "✅  Low Risk of Heart Disease"), pb,
                (HEART_PRECAUTIONS if pr==1 else LOW_RISK_TIPS),
                ("high-risk" if pr==1 else "low-risk")
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")
""",

    "pages/3_🌡️_Symptom_Checker.py": """\
\"\"\"General Symptom Checker page.\"\"\"
import streamlit as st, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import (load_models, render_result_card, ALL_SYMPTOMS,
                  SYMPTOM_PRECAUTIONS, DEFAULT_PRECAUTIONS)

st.set_page_config(page_title="Symptom Checker | DiagnoX", page_icon="🌡️", layout="wide")
st.markdown(\"\"\"<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Nunito:wght@600;700;800&display=swap');
  html,[class*='css']{font-family:'Nunito',sans-serif;}
  .stButton>button{background:linear-gradient(135deg,#17C3CE,#0EA5B0);color:#fff!important;
    border:none;border-radius:100px;padding:10px 28px;font-weight:800;}
  h1,h2,h3{font-family:'Syne',sans-serif!important;font-weight:800!important;color:#0C2340;}
  .result-card{background:#fff;border-radius:22px;padding:28px 32px;border:1.5px solid #CBE9EF;
    box-shadow:0 8px 32px rgba(23,195,206,.12);margin-top:20px;}
  .detected{border-left:6px solid #FFB547;}
  .disclaimer-box{background:#FFF8EC;border:1.5px solid #FFB547;border-radius:14px;
    padding:14px 20px;font-size:.85rem;color:#7A5C00;margin-top:16px;}
  .tip-list li{margin-bottom:6px;line-height:1.65;}
\"\"\", unsafe_allow_html=True)

models = load_models()
st.title("🌡️ General Symptom Checker")
if models.get("symptom_model") is None:
    st.error("⚠️ Symptom model not found. Run `python train_models.py` first.")
    st.stop()

symptom_display = [s.replace("_", " ").title() for s in ALL_SYMPTOMS]
symptom_map     = dict(zip(symptom_display, ALL_SYMPTOMS))

selected_display = st.multiselect("🔍 Select your symptoms (1–10):", symptom_display, max_selections=10)

if st.button("🔍 Check Symptoms", use_container_width=False):
    if not selected_display:
        st.warning("Please select at least one symptom.")
    else:
        with st.spinner("Running Random Forest inference…"):
            try:
                sym_cols    = (models.get("symptom_columns") or ALL_SYMPTOMS)[:38]
                selected_raw= [symptom_map[s] for s in selected_display]
                fvec        = np.array([[1 if c in selected_raw else 0 for c in sym_cols]])
                pred_idx    = models["symptom_model"].predict(fvec)[0]
                disease     = models["symptom_encoder"].inverse_transform([pred_idx])[0]
                proba_arr   = models["symptom_model"].predict_proba(fvec)[0]
                confidence  = proba_arr[pred_idx]
                precautions = SYMPTOM_PRECAUTIONS.get(disease, DEFAULT_PRECAUTIONS)
                render_result_card(f"Detected Condition: {disease}", confidence, precautions, "detected")
                with st.expander("🔎 Top 3 alternate predictions"):
                    for rank, idx in enumerate(np.argsort(proba_arr)[::-1][:3], 1):
                        d = models["symptom_encoder"].inverse_transform([idx])[0]
                        st.markdown(f"**{rank}.** {d} — {proba_arr[idx]*100:.1f}%")
            except Exception as e:
                st.error(f"Prediction error: {e}")
""",

    "pages/4_ℹ️_How_It_Works.py": """\
\"\"\"How It Works page.\"\"\"
import streamlit as st
st.set_page_config(page_title="How It Works | DiagnoX", page_icon="ℹ️", layout="wide")
st.title("ℹ️ How DiagnoX Works")
st.markdown(\"\"\"
DiagnoX uses pre-trained Scikit-Learn models to assess disease risk from your inputs.

### 🔬 ML Models at a Glance
| Module | Algorithm | Accuracy | Features |
|--------|-----------|----------|----------|
| Diabetes | SVM (RBF kernel) | 78–85% | 8 numerical |
| Heart Disease | SVM (Linear kernel) | 78–85% | 13 numerical |
| Symptom Checker | Random Forest (100 trees) | ~92% | 130+ binary |

### ⚙️ Prediction Pipeline
1. User enters inputs via form fields.
2. Inputs validated for range and completeness.
3. Numerical inputs scaled with the training-fitted `StandardScaler`.
4. `model.predict()` returns class; `model.predict_proba()` returns confidence.
5. Result, probability bar, and tailored precautions are displayed.

### 🧬 Training Datasets
| Module | Dataset | Records |
|--------|---------|---------|
| Diabetes | Pima Indians Diabetes | 768 |
| Heart Disease | Cleveland Heart Disease | 303 |
| Symptom Checker | Disease-Symptom Mapping (Kaggle) | 4,920 |

### ⚠️ Disclaimer
> This tool is for **informational purposes only** and is **NOT** a substitute for professional 
> medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

- Model accuracy is 78–92%; errors are possible.
- No personal data is stored or transmitted.
- Results do not replace physical examination or laboratory tests.
\"\"\")
""",
}

for path, content in PAGES.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✔  Written: {path}")

print("\n✅  apply_pages.py complete.")
print("   Launch the app with:  streamlit run main.py")