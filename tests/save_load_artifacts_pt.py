# SAVE THE MODEL (PYTORCH)

import torch, os, time
from joblib import dump  # to save the scaler used with the torch model

os.makedirs("models", exist_ok=True)
best_threshold = 0.37

# Save the PyTorch weights, optimizer (optional), and threshold
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),   # optional
    "threshold": best_threshold,
    "meta": {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "notes": "PyTorch MLP with pos_weight"
    }
}, "models/nn.pth")

# Save the scaler separately (so preprocessing matches at inference)
dump(scaler, "models/nn_scaler.joblib")

# LOAD AND USE THE MODEL

import torch
from joblib import load

# Recreate the same model class/architecture
model = MLP(input_dim=29, hidden=[128, 64], dropout=0.2)
checkpoint = torch.load("models/nn.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

scaler = load("models/nn_scaler.joblib")
thr = checkpoint["threshold"]

# Inference
import numpy as np
with torch.no_grad():
    X_proc = scaler.transform(X_new)                         # same preprocessing!
    logits = model(torch.tensor(X_proc, dtype=torch.float32))
    probs = torch.sigmoid(logits).numpy().ravel()
    pred = (probs >= thr).astype(int)

