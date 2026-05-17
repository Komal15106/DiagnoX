
PRODUCT REQUIREMENTS DOCUMENT
Multiple Disease Prediction System
Web Application
Field	Details
Document Version	1.0
Status	Draft
Date	March 2026
Prepared By	Product Team
Project Type	Healthcare Web Application

1. Executive Summary

This document defines the product requirements for the Multiple Disease Prediction System website — a machine-learning-powered health assistant that enables users to assess their risk for a range of chronic and common diseases from their browser, without any installation.

The system covers two modes of diagnosis:
•	General Health Mode: Users select everyday symptoms (e.g., Fever, Cough, Rash) to identify common illnesses such as flu, dengue, or fungal infections.
•	Diagnostic Mode: Users enter clinical values (e.g., Blood Glucose, Blood Pressure, Cholesterol) to assess risk for serious chronic diseases such as Diabetes and Heart Disease.

The goal is to deliver a one-stop, accessible health-monitoring platform that bridges the gap between complex medical diagnostics and everyday wellness checks — acting as a virtual first-opinion health assistant.

2. Problem Statement

2.1 The Gap in Existing Health Apps
Most consumer health apps are narrowly focused — they either help users track symptoms for common colds or provide risk calculators for chronic diseases. There is no widely accessible, integrated platform that handles both types of health queries in a single interface.

2.2 Key Challenges
•	Silent Killers: Diseases like Heart Disease and Diabetes have few early symptoms, making it difficult for users to self-identify risk without lab reports.
•	Symptom Confusion: Daily symptoms such as fever, headache, and joint pain overlap across many illnesses (e.g., flu vs. dengue vs. malaria), making self-diagnosis error-prone.
•	Access Barriers: Many users lack immediate access to diagnostic expertise, especially in early-stage situations where a quick risk assessment could prompt timely medical consultation.
•	Data Type Mismatch: Symptom-based and numerical medical data require different analytical approaches — most single tools handle only one type.

3. Goals & Objectives

3.1 Product Goals
Goal	Success Metric
Provide accurate disease risk prediction	>=80% model accuracy across all disease modules
Offer a seamless dual-mode experience	Users can switch modes without page reload
Deliver actionable health guidance	Every prediction includes precautions & lifestyle tips
Ensure accessibility for non-technical users	No medical jargon in inputs; tooltips on all fields
Support mobile and desktop browsers	Responsive layout passing Lighthouse mobile score >=90

3.2 Key Objectives
•	Build a fully responsive web interface using a modern frontend framework.
•	Integrate three ML models (SVM for Diabetes, SVM for Heart Disease, Random Forest for General Symptoms) via a backend API.
•	Allow users to toggle between Symptom Check and Lab Report Check modes.
•	Display prediction results alongside specific precautions and lifestyle recommendations.
•	Ensure the system is production-ready with error handling, input validation, and fast response times.

4. Target Users

User Persona	Description	Primary Need
General Public	Adults managing their everyday health, no medical training	Quick symptom triage and guidance
At-Risk Individuals	People with family history of diabetes or heart disease	Risk assessment using clinical values
Rural Users	Limited access to healthcare facilities	First-opinion health check before clinic visit
Students / Researchers	Learning ML applications in healthcare	Explore disease prediction models
Healthcare Support Staff	Nurses or community health workers	Rapid pre-screening tool

5. Scope

5.1 In Scope
•	Landing page with project overview and mode selection.
•	General Health Mode: multi-select symptom form covering 130+ symptoms, mapped to 40+ diseases.
•	Diagnostic Mode: structured input forms for Diabetes (Glucose, BMI, Age, Insulin, etc.) and Heart Disease (Chest Pain Type, Resting BP, Cholesterol, etc.).
•	ML prediction engine exposed via REST API (Python/Flask or FastAPI backend).
•	Results page showing: Disease name, confidence/probability, precautions, and lifestyle advice.
•	Responsive design for mobile, tablet, and desktop.
•	Basic input validation and error messaging.

5.2 Out of Scope (V1)
•	User accounts, login, or saved history.
•	Image-based diagnosis (e.g., skin disease photo upload) — planned for V2.
•	Integration with wearable devices or EHR systems.
•	Real-time doctor consultation or telemedicine features.
•	Prescription or medication recommendations.

6. Functional Requirements

6.1 Homepage / Landing Page
•	Display a hero section with the app name, tagline, and a clear call-to-action to begin a health check.
•	Present two mode cards: "Check Symptoms" and "Check Lab Report" with brief descriptions.
•	Include a brief About section explaining how the prediction engine works (non-technical language).
•	Provide a disclaimer: "This tool is for informational purposes only and is not a substitute for professional medical advice."

6.2 General Health Mode (Symptom Checker)
•	Display a searchable, scrollable multi-select list of 130+ symptoms grouped by category (e.g., Respiratory, Skin, Neurological).
•	Allow users to select 1–10 symptoms and submit.
•	Show a loading indicator while the model runs inference.
•	Display predicted disease name with a confidence score.
•	List 4–6 specific precautions tailored to the predicted disease.
•	Provide a "Try Again" button to clear and restart.

6.3 Diagnostic Mode — Diabetes
•	Present a clean form with labeled inputs for: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age.
•	Include range hints and tooltips (e.g., "Normal fasting glucose: 70–100 mg/dL").
•	Validate that all fields are numeric and within medically plausible ranges.
•	Return a binary result: "High Risk of Diabetes" or "Low Risk of Diabetes" with a probability percentage.
•	Display tailored lifestyle tips (e.g., "Reduce sugar intake", "30 minutes of daily exercise").

6.4 Diagnostic Mode — Heart Disease
•	Form inputs: Age, Sex, Chest Pain Type (dropdown), Resting Blood Pressure, Serum Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced Angina, Oldpeak, Slope, Number of Major Vessels, Thalassemia.
•	Use dropdowns or radio buttons for categorical inputs (e.g., Chest Pain Type).
•	Return result: "High Risk of Heart Disease" or "Low Risk" with probability.
•	Display specific precautions: e.g., "Schedule a cardiology consultation", "Monitor blood pressure weekly".

6.5 Results Page
•	Clearly display the predicted condition in a prominent card with colour coding (Red = High Risk, Green = Low Risk, Amber = Detected Illness).
•	Show confidence/probability as a visual progress bar.
•	List precautions as a numbered checklist.
•	Provide a "Disclaimer" note encouraging users to consult a certified healthcare professional.
•	Option to share results via link or download as PDF.

7. Non-Functional Requirements

Category	Requirement	Target
Performance	API response time for prediction	< 2 seconds (p95)
Availability	Uptime SLA	>= 99.5%
Scalability	Concurrent users supported	>= 500 simultaneous users
Security	All traffic encrypted	HTTPS / TLS 1.2+
Accessibility	WCAG compliance level	WCAG 2.1 AA
Browser Support	Minimum browser versions	Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
Mobile	Responsive breakpoints	320px, 768px, 1024px, 1280px
Load Time	Initial page load (LCP)	< 2.5 seconds on 4G connection

8. Technical Architecture & Tech Stack

8.1 Frontend
•	Framework: React.js (with Vite) or Next.js for SSR/SEO benefits.
•	Styling: Tailwind CSS for responsive, utility-first design.
•	State Management: React Context or Zustand for form and result state.
•	Charting: Chart.js or Recharts for confidence visualizations.

8.2 Backend / API
•	Runtime: Python 3.10+ with FastAPI (preferred for performance) or Flask.
•	ML Inference: Scikit-Learn models loaded via Pickle.
•	Endpoints:
◦	POST /predict/symptoms — accepts symptom list, returns disease + precautions.
◦	POST /predict/diabetes — accepts 8 numerical inputs, returns risk + tips.
◦	POST /predict/heart — accepts 13 clinical inputs, returns risk + tips.
◦	GET /health — health check endpoint.

8.3 ML Models
Module	Algorithm	Accuracy
Diabetes Prediction	Support Vector Machine (SVM)	~78–85%
Heart Disease Prediction	Support Vector Machine (SVM)	~78–85%
General Symptom Checker	Random Forest Classifier	~92%

8.4 Deployment
•	Frontend Hosting: Vercel or Netlify (CDN-backed, auto-deploy from Git).
•	Backend Hosting: Render, Railway, or AWS EC2 with Docker containerization.
•	CI/CD: GitHub Actions for automated testing and deployment on merge to main.
•	Environment Config: .env files for API base URLs and secrets; never committed to version control.

9. User Flows

9.1 Symptom Checker Flow
1.	User lands on Homepage → clicks "Check Symptoms".
2.	Symptom selection page loads with a searchable multi-select list.
3.	User selects 1–10 symptoms and clicks "Predict".
4.	Loading spinner shown; API call made to POST /predict/symptoms.
5.	Results page shows: Predicted Disease, Confidence Bar, Precautions.
6.	User clicks "Try Again" or navigates to another mode.

9.2 Diagnostic Mode Flow
7.	User clicks "Check Lab Report" on the Homepage.
8.	Mode selector appears: "Diabetes" or "Heart Disease".
9.	Appropriate form renders with labeled inputs and range hints.
10.	User fills inputs; client-side validation highlights errors before submission.
11.	On submit, API call to /predict/diabetes or /predict/heart.
12.	Results page renders with risk label, probability, and lifestyle advice.

10. UI / UX Requirements

10.1 Design Principles
•	Clarity First: Medical information presented in plain, jargon-free language.
•	Trust Signals: Consistent use of medical iconography, clean typography, and a calming blue/white colour palette.
•	Minimal Friction: Users should reach a prediction result in 3 clicks or fewer.
•	Error Prevention: Inline validation and helpful hints before form submission.

10.2 Key Pages
•	Home / Landing Page
•	Symptom Checker Page
•	Diabetes Input Form Page
•	Heart Disease Input Form Page
•	Results Page (shared template, content dynamic per prediction)
•	About / How It Works Page

10.3 Accessibility
•	All images include descriptive alt text.
•	Form inputs are properly labelled for screen readers.
•	Colour contrast ratios meet WCAG 2.1 AA (4.5:1 minimum for normal text).
•	Keyboard navigation supported throughout.

11. Release Milestones

Phase	Deliverables	Target Timeline
Phase 1	Backend API development, ML model integration & testing	Weeks 1–2
Phase 2	Frontend: Landing page, Symptom Checker UI, Diagnostic forms	Weeks 3–4
Phase 3	Results page, input validation, accessibility audit	Week 5
Phase 4	End-to-end testing, performance optimization, deployment	Week 6
V2 Scope	Image-based skin disease detection, user accounts, history	TBD

12. Risks & Mitigations

Risk	Severity	Mitigation
Users may misinterpret predictions as medical diagnoses	High	Prominent disclaimer on every page; "Consult a doctor" CTA on results
Model accuracy drops for edge-case inputs	Medium	Input range validation; model retraining pipeline planned for V2
Backend API downtime causing failed predictions	High	Graceful error states in UI; health check endpoint; 99.5% SLA target
CORS issues between frontend and backend on deployment	Low	Configure CORS headers in FastAPI; test with staging environment

13. Future Scope (V2 & Beyond)

•	Skin Disease Detection: Image upload module using a CNN model to classify skin conditions from photos.
•	User Accounts & History: Allow users to create profiles, save past predictions, and track health trends over time.
•	Expanded Disease Coverage: Add modules for kidney disease, liver disease, and thyroid disorders.
•	Multilingual Support: Translate the interface into Hindi and other regional languages to increase accessibility.
•	Doctor Referral Integration: Partner with telemedicine platforms to offer a "Book a Doctor" CTA after high-risk predictions.
•	Wearable Integration: Pull real-time data from Apple Health, Google Fit, or Fitbit to pre-populate diagnostic forms.
•	Model Improvement Pipeline: Continuous retraining with new data to improve accuracy over time.


End of Document  |  Multiple Disease Prediction System PRD v1.0
