
TECH STACK & ARCHITECTURE SPECIFICATION
Multiple Disease Prediction System
Streamlit Web Application
Field	Details
Document Version	1.0
Status	Approved
Date	March 2026
Language	Python 3.10+
UI Framework	Streamlit
ML Library	Scikit-Learn

1. Overview

This document is the authoritative technical reference for the Multiple Disease Prediction System. It specifies every library, framework, tool, and architectural pattern that the development team must use to build, train, and deploy the application.

The system is built exclusively in Python. The web interface is powered by Streamlit, which allows a fully functional, interactive health-assessment dashboard to be written without any HTML, CSS, or JavaScript. All machine learning inference is handled by pre-trained Scikit-Learn models serialised with Pickle and loaded at runtime.

2. Tech Stack at a Glance

Layer	Technology	Purpose
Language	Python 3.10+	Primary and only programming language for the entire project
Web UI	Streamlit	Browser-based interactive dashboard; sidebar navigation between disease modes
ML Core	Scikit-Learn	SVM for Diabetes & Heart Disease; Random Forest for symptom-based prediction
Data Layer	Pandas & NumPy	Dataset loading, cleaning, feature engineering, and numerical operations
Model Persistence	Pickle	Serialise trained models to .pkl files for instant loading on app start
Development	Jupyter Notebook	EDA, model training experiments, and accuracy benchmarking
Version Control	Git & GitHub	Source control, collaboration, and CI trigger for deployment
Deployment	Streamlit Cloud	One-click hosting directly from a GitHub repository; zero server config
Environment	pip + venv	Dependency isolation via requirements.txt and virtual environments

3. Python — Core Language

Python 3.10 or higher is the sole programming language for this project. Every component — data preprocessing, model training, prediction logic, and the web UI — is written in Python.

3.1 Why Python
•	Unmatched ecosystem for data science and machine learning (Pandas, NumPy, Scikit-Learn).
•	Streamlit is Python-native, eliminating the need for a separate frontend language.
•	Rapid prototyping with Jupyter Notebooks during the EDA and training phase.
•	Single language across the full stack reduces context-switching and onboarding time.

3.2 Version & Setup
•	Minimum version: Python 3.10.
•	Use a virtual environment to isolate dependencies:
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
pip install -r requirements.txt

3.3 requirements.txt
All project dependencies must be pinned in requirements.txt and committed to the repository:
streamlit==1.35.0
scikit-learn==1.4.2
pandas==2.2.2
numpy==1.26.4
pickle5==0.0.12   # only if Python < 3.8; built-in otherwise

4. Scikit-Learn — Machine Learning Engine

Scikit-Learn is the primary ML library. It provides all algorithms, preprocessing utilities, and evaluation tools used in the project. Two distinct algorithm families are employed because the two data types (numerical vs. categorical symptoms) have different statistical characteristics.

4.1 Algorithm Selection
Disease Module	Algorithm	Scikit-Learn Class	Accuracy
Diabetes Prediction	Support Vector Machine	SVC (kernel='rbf')	78 – 85 %
Heart Disease Prediction	Support Vector Machine	SVC (kernel='linear')	78 – 85 %
General Symptom Checker	Random Forest Classifier	RandomForestClassifier	~92 %

4.2 Support Vector Machine (SVM) — Numerical Modules
SVM is used for Diabetes and Heart Disease because it excels at drawing a precise decision boundary in high-dimensional numerical feature spaces. It minimises classification error while maximising the margin between healthy and unhealthy data points.

Key Scikit-Learn calls:
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Preprocessing pipeline for numerical data:
1.	Load CSV with Pandas.
2.	Handle missing values — impute with column mean.
3.	Split into features (X) and label (y).
4.	Scale with StandardScaler (zero mean, unit variance).
5.	Split 80/20 train/test with stratify=y.
6.	Fit SVC, evaluate accuracy_score on test set.

4.3 Random Forest Classifier — Symptom Module
Random Forest is used for the General Symptom Checker because it handles high-cardinality categorical input well. Symptoms are converted to a binary vector (1 = present, 0 = absent) across 130+ features; Random Forest then votes across multiple decision trees to map the vector to one of 40+ diseases.

Key Scikit-Learn calls:
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

Preprocessing pipeline for symptom data:
7.	Load symptom-disease CSV with Pandas.
8.	Encode disease labels to integers using LabelEncoder.
9.	Convert symptom columns to binary integer matrix (0/1).
10.	Fit RandomForestClassifier(n_estimators=100, random_state=42).
11.	Evaluate on held-out test split; target >= 92% accuracy.

4.4 Model Evaluation Metrics
•	Primary metric: Accuracy Score (accuracy_score from sklearn.metrics).
•	Secondary: Confusion Matrix and Classification Report for per-class precision/recall.
•	Use cross_val_score with cv=5 for robust generalisation estimates during training.

5. Pandas & NumPy — Data Handling

Pandas and NumPy underpin all data operations. They are imported in both the Jupyter training notebooks and the Streamlit app itself (for runtime input shaping).

5.1 Pandas
Pandas provides the DataFrame abstraction used throughout the project.

Usage by phase:
Phase	Pandas Operations
Data Loading	pd.read_csv() to load all three datasets
EDA	df.describe(), df.info(), df.isnull().sum() — inspect distributions and missing values
Cleaning	df.fillna(), df.dropna(), df.replace() — handle nulls and outliers
Feature Engineering	df[cols].values to extract NumPy arrays; pd.get_dummies() for one-hot if needed
Runtime (Streamlit)	pd.DataFrame([input_dict]) to wrap user inputs before passing to model

5.2 NumPy
NumPy handles all low-level numerical operations and array manipulation.

Key uses:
•	Reshape user input into model-expected 2D arrays: np.array([[v1, v2, ...]]).
•	Compute feature statistics (np.mean, np.std) during preprocessing.
•	Provide the numerical backbone for StandardScaler transformations.
•	Random seed management: np.random.seed(42) for reproducibility.

6. Pickle — Model Persistence

Pickle serialises Python objects to binary files. In this project it is used to save trained Scikit-Learn models and their associated preprocessing scalers so they can be loaded instantly when the Streamlit app starts — without retraining on every request.

6.1 Saving a Trained Model
import pickle

# After fitting the model:
with open('models/diabetes_model.pkl', 'wb') as f:
    pickle.dump(diabetes_svm, f)

# Also save the scaler (must transform runtime inputs identically):
with open('models/diabetes_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

6.2 Loading at Runtime
import pickle

diabetes_model  = pickle.load(open('models/diabetes_model.pkl',  'rb'))
diabetes_scaler = pickle.load(open('models/diabetes_scaler.pkl', 'rb'))
heart_model     = pickle.load(open('models/heart_model.pkl',     'rb'))
heart_scaler    = pickle.load(open('models/heart_scaler.pkl',    'rb'))
symptom_model   = pickle.load(open('models/symptom_model.pkl',   'rb'))
symptom_encoder = pickle.load(open('models/symptom_encoder.pkl', 'rb'))

6.3 File Naming Convention
File	Type	Description
models/diabetes_model.pkl	SVC object	Trained diabetes SVM
models/diabetes_scaler.pkl	StandardScaler	Scaler fitted on diabetes train set
models/heart_model.pkl	SVC object	Trained heart disease SVM
models/heart_scaler.pkl	StandardScaler	Scaler fitted on heart train set
models/symptom_model.pkl	RandomForestClassifier	Trained symptom classifier
models/symptom_encoder.pkl	LabelEncoder	Disease label decoder

6.4 Security Note
Pickle files should only be loaded from trusted sources. Do not expose .pkl endpoints publicly. Store model files inside the repository's models/ directory and restrict write access in production.

7. Streamlit — Web Interface

Streamlit transforms the Python prediction logic into a fully interactive web application. It eliminates the need for HTML, CSS, JavaScript, or a separate API layer. The entire UI is declared in Python using Streamlit's component library.

7.1 Installation & Running Locally
pip install streamlit
streamlit run app.py
The app opens automatically in the default browser at http://localhost:8501.

7.2 Sidebar Navigation
Streamlit's sidebar (st.sidebar) is the primary navigation mechanism. It renders a persistent panel on the left side of the screen, providing a clean menu for switching between disease prediction modes without a page reload.

Implementation pattern:
import streamlit as st

st.set_page_config(page_title='Disease Prediction System', layout='wide')

with st.sidebar:
    st.title('Navigation')
    selected = st.selectbox(
        'Choose a Module',
        ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Symptom Checker']
    )

if selected == 'Diabetes Prediction':
    show_diabetes_page()
elif selected == 'Heart Disease Prediction':
    show_heart_page()
elif selected == 'Symptom Checker':
    show_symptom_page()

7.3 Key Streamlit Components
Component	Streamlit API	Use in Project
Sidebar menu	st.sidebar.selectbox()	Switch between Diabetes / Heart / Symptom modes
Number input	st.number_input()	Clinical values (Glucose, BP, Cholesterol, etc.)
Dropdown (select)	st.selectbox()	Categorical inputs (Chest Pain Type, Sex, etc.)
Multi-select	st.multiselect()	Symptom selection from 130+ options
Button	st.button()	Trigger prediction; clear form
Result banner	st.success() / st.error()	Green = Low Risk; Red = High Risk / Disease Detected
Spinner	st.spinner()	Loading indicator while model runs
Metric card	st.metric()	Display prediction probability / confidence
Progress bar	st.progress()	Visual confidence score display
Info / Warning box	st.info() / st.warning()	Medical disclaimer and precaution display
Columns layout	st.columns()	Side-by-side input grouping on Diagnostic forms
Expander	st.expander()	Collapsible 'How does this work?' sections

7.4 Page Structure per Mode
Diabetes & Heart Disease (Diagnostic Mode)
•	st.title() — page heading.
•	st.columns(2) — split inputs into left/right columns for readability.
•	st.number_input() for each clinical feature with min/max range guards.
•	st.button('Predict') — on click, scale inputs and call model.predict().
•	st.success() / st.error() for the binary outcome.
•	st.metric() for probability from model.predict_proba().
•	st.info() for the precaution checklist.

General Symptom Checker
•	st.multiselect() with all 130+ symptom strings as options.
•	Convert selected symptoms into a binary NumPy vector matching training feature order.
•	st.button('Check Symptoms') to run RandomForest inference.
•	Display predicted disease name and precautions.

7.5 Custom Styling
While Streamlit handles layout automatically, limited custom CSS can be injected for branded colours and typography:
st.markdown('''
  <style>
    .stButton>button { background-color:#2E75B6; color:white; border-radius:8px; }
    .stSelectbox label { font-weight:600; }
  </style>
''', unsafe_allow_html=True)

8. Project Directory Structure

The repository should be organised as follows. Every path is relative to the project root.

Path	Description
app.py	Main Streamlit entry point — navigation router, page layout
requirements.txt	Pinned Python dependencies
notebooks/	Jupyter notebooks for EDA, model training, and evaluation
notebooks/diabetes_training.ipynb	Diabetes SVM training notebook
notebooks/heart_training.ipynb	Heart Disease SVM training notebook
notebooks/symptom_training.ipynb	Random Forest training notebook
data/	Raw CSV datasets (not committed if large; use Git LFS)
data/diabetes.csv	768-record Pima Indians Diabetes dataset
data/heart.csv	Cleveland Heart Disease dataset
data/symptoms.csv	130+ symptom / 40+ disease mapping dataset
models/	Serialised .pkl model and scaler files
pages/	Optional: Streamlit multi-page files (diabetes.py, heart.py, symptoms.py)
utils/	Shared helper functions (input validation, precaution lookup, etc.)
utils/precautions.py	Dictionary mapping disease names to precaution lists
.streamlit/config.toml	Streamlit theme configuration (primary colour, font)

9. End-to-End Data Flow

9.1 Training Phase (Offline — Jupyter Notebooks)
12.	Load raw CSV dataset with pd.read_csv().
13.	Inspect and clean data (handle nulls, check dtypes) using Pandas.
14.	Separate features X and target y.
15.	Scale X with StandardScaler (numerical) or encode labels with LabelEncoder (symptoms).
16.	Train-test split (80/20, stratified).
17.	Fit the chosen algorithm (SVC or RandomForestClassifier).
18.	Evaluate with accuracy_score, confusion_matrix, classification_report.
19.	Serialise model and scaler to models/ with Pickle.

9.2 Inference Phase (Runtime — Streamlit App)
20.	App starts: all .pkl files loaded once into memory.
21.	User selects a module from the sidebar.
22.	Streamlit form collects inputs via number_input / selectbox / multiselect.
23.	On button click: inputs assembled into a NumPy array.
24.	Array scaled with the pre-fitted scaler (numerical modules).
25.	model.predict() returns the class label.
26.	model.predict_proba() returns confidence percentages.
27.	Result and precautions rendered in Streamlit UI.

10. Deployment

10.1 Streamlit Community Cloud (Recommended)
Streamlit Community Cloud provides free, one-click hosting directly from a GitHub repository. No server configuration, Docker setup, or CI pipeline is required for basic deployment.

Steps:
28.	Push the project to a public (or private with Pro plan) GitHub repository.
29.	Log in to share.streamlit.io with a GitHub account.
30.	Click 'New app', select the repository, branch, and entry file (app.py).
31.	Streamlit Cloud reads requirements.txt and installs dependencies automatically.
32.	The app is live at a public URL (e.g., https://yourapp.streamlit.app).
33.	Every push to the main branch triggers an automatic redeploy.

10.2 Streamlit Theme Configuration
Add a .streamlit/config.toml file to control the visual theme:
[theme]
primaryColor         = "#2E75B6"
backgroundColor      = "#FFFFFF"
secondaryBackgroundColor = "#F4F8FC"
textColor            = "#1A3A5C"
font                 = "sans serif"

10.3 Alternative Deployment Targets
Platform	Method	Notes
Render	Docker container or Python buildpack	Supports always-on free tier
Heroku	Procfile: web: streamlit run app.py	Requires paid dyno for production
AWS EC2	Nginx reverse proxy to streamlit port 8501	Full control; suitable for high traffic
Docker	FROM python:3.10; CMD streamlit run app.py	Portable; run anywhere with Docker

11. Non-Functional Technical Standards

Standard	Requirement	Implementation Note
Code Style	PEP 8 compliance	Enforce with flake8 or black formatter
Reproducibility	Fixed random seeds	np.random.seed(42); random_state=42 in all sklearn calls
Model Loading	Load once at startup	Use @st.cache_resource decorator on model-loading functions
Input Validation	Client-side range guards	min= and max= args on st.number_input()
Error Handling	Graceful failure messages	Wrap model calls in try/except; display st.error() on failure
Performance	Prediction latency < 1 second	Models are in-memory; SVM/RF inference is sub-millisecond
Security	No user data persisted	All inputs are session-scoped Streamlit state; no DB or logging
Disclaimer	Shown on every prediction result	st.warning() with fixed disclaimer text after each result

11.1 Caching Models for Performance
Model loading is expensive on cold start. Use Streamlit's built-in caching to load each model only once per session:
@st.cache_resource
def load_models():
    return {
        'diabetes_model':  pickle.load(open('models/diabetes_model.pkl',  'rb')),
        'diabetes_scaler': pickle.load(open('models/diabetes_scaler.pkl', 'rb')),
        'heart_model':     pickle.load(open('models/heart_model.pkl',     'rb')),
        'heart_scaler':    pickle.load(open('models/heart_scaler.pkl',    'rb')),
        'symptom_model':   pickle.load(open('models/symptom_model.pkl',   'rb')),
        'symptom_encoder': pickle.load(open('models/symptom_encoder.pkl', 'rb')),
    }

models = load_models()

12. Library Version Summary

Library / Tool	Version	Install Command
Python	3.10+	python.org/downloads
Streamlit	1.35.0	pip install streamlit==1.35.0
Scikit-Learn	1.4.2	pip install scikit-learn==1.4.2
Pandas	2.2.2	pip install pandas==2.2.2
NumPy	1.26.4	pip install numpy==1.26.4
Pickle	Built-in (3.x)	No install needed — import pickle
Jupyter Notebook	7.x	pip install notebook==7.2.0
Git	Latest	git-scm.com


End of Document  |  Tech Stack & Architecture Specification  |  v1.0  |  March 2026
