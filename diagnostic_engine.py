import json

class DiagnosticEngine:
    """
    Core intelligence engine for Diagno-X implementing normal ailments and model routing.
    """
    
    def __init__(self):
        self.emergency_flags = ['crushing chest pain', 'slurred speech', 'chest pain', 'sudden paralysis', 'weakness', 'loss of balance']
        
    def analyze(self, symptoms, duration_days=None, age=None, sex=None):
        symptoms_lower = [s.lower() for s in symptoms]
        
        # 1. SCOPE FILTER (Emergency Check)
        for flag in self.emergency_flags:
            if any(flag in s for s in symptoms_lower) or any(s in flag for s in symptoms_lower):
                return json.dumps({
                    "detected_normal_disease": "Emergency Flag",
                    "confidence_score": "100%",
                    "symptom_category": "Emergency / Critical",
                    "target_model_route": "EMERGENCY_ESCALATION",
                    "action_plan": "Immediately call emergency services or visit the nearest ER."
                })
                
        # 3. DYNAMIC MODEL ROUTING (Determine Category and Route first)
        category = "Unknown"
        route = "LIGHTWEIGHT_RULES_ENGINE"
        
        has_resp = any(s in ['cough', 'runny nose', 'sneezing', 'sore throat', 'continuous sneezing'] for s in symptoms_lower)
        has_neuro = any(s in ['severe headache', 'headache', 'scalp tenderness', 'stiff neck', 'neck pain'] for s in symptoms_lower)
        has_gastro = any(s in ['nausea', 'vomiting', 'stomach cramps', 'stomach pain', 'abdominal pain', 'diarrhoea', 'diarrhea'] for s in symptoms_lower)
        
        if len(symptoms_lower) <= 2 and (has_resp or has_neuro or has_gastro):
            route = "LIGHTWEIGHT_RULES_ENGINE"
        elif has_neuro:
            route = "NEURO_HEADACHE_MODEL"
            category = "Neurological"
        elif has_gastro:
            route = "GASTRO_MODEL"
            category = "Gastrointestinal"
        elif has_resp:
            route = "RESPIRATORY_MODEL"
            category = "Respiratory"
            
        if route == "LIGHTWEIGHT_RULES_ENGINE":
            if has_neuro: category = "Neurological"
            elif has_resp: category = "Respiratory"
            elif has_gastro: category = "Gastrointestinal"
            else: category = "General"

        # 2. SYMPTOM CLUSTER & PREDICTION LOGIC
        scores = {
            "Seasonal Flu": 0,
            "Common Cold": 0,
            "Severe Headache / Migraine": 0,
            "Seasonal Allergies": 0,
            "Acute Gastroenteritis (Stomach Bug)": 0
        }
        
        # Helpers
        has_high_fever = any(s in ['high fever'] for s in symptoms_lower)
        has_mild_fever = any(s in ['mild fever', 'fever'] for s in symptoms_lower)
        has_fever = has_high_fever or has_mild_fever
        has_severe_aches = any(s in ['severe body aches', 'muscle pain', 'joint pain', 'body aches'] for s in symptoms_lower)
        
        for s in symptoms_lower:
            # Flu
            if s in ['sudden onset', 'fatigue', 'high fever', 'severe body aches', 'muscle pain']:
                scores["Seasonal Flu"] += 25
                
            # Cold
            if s in ['runny nose', 'sore throat', 'gradual sore throat', 'mild fever', 'mild cough']:
                scores["Common Cold"] += 25
                
            # Headache / Migraine
            if s in ['throbbing pain', 'sensitivity to light', 'sensitivity to sound', 'nausea', 'headache', 'severe headache']:
                scores["Severe Headache / Migraine"] += 25
                
            # Allergies
            if s in ['sneezing', 'continuous sneezing', 'itchy eyes', 'watering from eyes']:
                scores["Seasonal Allergies"] += 30
                
            # Gastroenteritis
            if s in ['nausea', 'vomiting', 'stomach cramps', 'stomach pain', 'abdominal pain', 'diarrhoea', 'diarrhea']:
                scores["Acute Gastroenteritis (Stomach Bug)"] += 25

        # Logic Modifiers
        if has_fever:
            scores["Seasonal Allergies"] = 0
            
        if not has_severe_aches and not has_high_fever and scores["Common Cold"] > 0:
            scores["Common Cold"] += 20
            
        if has_high_fever and has_severe_aches:
            scores["Seasonal Flu"] += 30
            
        # Determine best prediction
        best_disease = max(scores, key=scores.get)
        max_score = scores[best_disease]
        
        confidence = min(100, max_score)
        
        if confidence == 0:
            best_disease = "Inconclusive / Insufficient Data"
            
        action_plans = {
            "Seasonal Flu": "Rest, hydrate, and take antipyretics. See a doctor if symptoms worsen.",
            "Common Cold": "Rest and drink clear fluids. Use over-the-counter cold remedies if needed.",
            "Severe Headache / Migraine": "Rest in a dark, quiet room. Take prescribed migraine medication or NSAIDs.",
            "Seasonal Allergies": "Avoid known allergens. Use over-the-counter antihistamines.",
            "Acute Gastroenteritis (Stomach Bug)": "Stay hydrated with oral rehydration salts. Eat a bland diet (BRAT) when tolerated.",
            "Inconclusive / Insufficient Data": "Monitor symptoms carefully and consult a healthcare professional."
        }

        return json.dumps({
            "detected_normal_disease": best_disease,
            "confidence_score": f"{int(confidence)}%",
            "symptom_category": category,
            "target_model_route": route,
            "action_plan": action_plans.get(best_disease, "Consult a doctor for guidance.")
        })
