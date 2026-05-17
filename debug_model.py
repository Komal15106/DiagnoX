import pickle
import os

BASE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE, "models", "symptom_model.pkl")

with open(os.path.join(BASE, "model_debug.txt"), "w") as f:
    try:
        if os.path.exists(model_path):
            with open(model_path, "rb") as model_file:
                m = pickle.load(model_file)
            f.write(f"Features: {m.n_features_in_}\n")
            # If it's a RandomForest, we can't get names easily if they aren't there.
            # But let's check if the symptom_encoder has any clues.
            enc_path = os.path.join(BASE, "models", "symptom_encoder.pkl")
            if os.path.exists(enc_path):
                with open(enc_path, "rb") as enc_file:
                    e = pickle.load(enc_file)
                f.write(f"Diseases: {len(e.classes_)}\n")
                f.write(f"Classes: {','.join(e.classes_)}\n")
        else:
            f.write("Model not found\n")
    except Exception as ex:
        f.write(f"Error: {ex}\n")
