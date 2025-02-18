# app/config.py

import torch

MODEL_PATH = "models/global_lstm_state.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Otros parámetros de configuración que necesites...
