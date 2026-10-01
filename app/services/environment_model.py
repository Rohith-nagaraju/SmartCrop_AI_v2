from pathlib import Path
import joblib
from ..config import MODEL_DIR

def predict_environment(features):
    p=MODEL_DIR/'environment_model.joblib'
    if not p.exists(): return None
    return float(joblib.load(p).predict([features])[0])
