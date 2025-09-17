# SAVE THE MODEL (SCIKIT-LEARN)
from joblib import dump
import json, os, time

os.makedirs("models", exist_ok=True)
best_threshold = 0.42  # chosen on validation PR/F1

bundle = {
    "scaler": scaler,                  # e.g., StandardScaler fit on train
    "model": clf,                      # e.g., LogisticRegression/SVC/etc.
    "threshold": best_threshold,
    "meta": {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": __import__("sklearn").__version__,
        "notes": "SMOTE + LogReg baseline"
    }
}
dump(bundle, "models/baseline.joblib")

# LOAD AND USE THE MODEL 
from joblib import load
import numpy as np

bundle = load("models/baseline.joblib")
scaler = bundle["scaler"]
model = bundle["model"]
thr = bundle["threshold"]

X_proc = scaler.transform(X_new)
proba = model.predict_proba(X_proc)[:, 1]  # or decision_function then sigmoid
pred = (proba >= thr).astype(int)
