import pickle
import os
import sys

# Ensure output is flushed
sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE, "models/symptom_model.pkl")

with open("output.txt", "w") as out:
    try:
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            out.write(f"n_features_in: {model.n_features_in_}\n")
            if hasattr(model, "feature_names_in_"):
                out.write("feature_names_in: " + ",".join(model.feature_names_in_) + "\n")
            else:
                out.write("No feature names found\n")
        else:
            out.write("Model not found\n")
    except Exception as e:
        out.write(f"Error: {e}\n")

print("Done")
