# app/config.py

import torch

MODEL_PATH = "models/global_lstm_state (3).pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

