# app/model.py
import torch
import torch.nn as nn
from app.config import MODEL_PATH, DEVICE

class GlobalLSTM(nn.Module):
    def __init__(self,
                 num_stores,
                 num_depts,
                 num_types,
                 emb_dim_store=4,
                 emb_dim_dept=8,
                 emb_dim_type=2,
                 num_numeric_features=15,
                 hidden_size=64,
                 num_layers=1,
                 dropout=0.2):
        super().__init__()
        self.store_emb = nn.Embedding(num_stores, emb_dim_store)
        self.dept_emb = nn.Embedding(num_depts, emb_dim_dept)
        self.type_emb = nn.Embedding(num_types, emb_dim_type)
        self.input_dim = emb_dim_store + emb_dim_dept + emb_dim_type + num_numeric_features
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, X_cat, X_num):
        store_id = X_cat[:, :, 0].long()
        dept_id = X_cat[:, :, 1].long()
        type_id = X_cat[:, :, 2].long()
        store_emb_out = self.store_emb(store_id)
        dept_emb_out = self.dept_emb(dept_id)
        type_emb_out = self.type_emb(type_id)
        concat_input = torch.cat([store_emb_out, dept_emb_out, type_emb_out, X_num], dim=-1)
        lstm_out, _ = self.lstm(concat_input)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return torch.relu(out).squeeze(-1)  # Para evitar valores negativos

def load_model(num_stores, num_depts, num_types, num_numeric_features):
    model = GlobalLSTM(
        num_stores=num_stores,
        num_depts=num_depts,
        num_types=num_types,
        num_numeric_features=num_numeric_features
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model
