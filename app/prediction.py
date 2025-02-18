# app/prediction.py
import torch
import numpy as np
import pandas as pd

def predict_store_custom(model, df, store_id, dept_ids=None,
                           seq_length=8, forecast_horizon=1,
                           cat_cols=None, num_cols=None, device='cpu'):
    results = []
    if dept_ids is None:
        dept_ids = df[df['Store_id'] == store_id]['Dept_id'].unique()
    else:
        if not isinstance(dept_ids, (list, np.ndarray)):
            dept_ids = [dept_ids]

    for dept in dept_ids:
        sub_df = df[(df['Store_id'] == store_id) & (df['Dept_id'] == dept)].sort_values('Date')
        if len(sub_df) < seq_length:
            continue
        if forecast_horizon == 1:
            last_seq = sub_df.iloc[-seq_length:]
            X_cat = torch.tensor(last_seq[cat_cols].values[np.newaxis, :, :], dtype=torch.float32).to(device)
            X_num = torch.tensor(last_seq[num_cols].values[np.newaxis, :, :], dtype=torch.float32).to(device)
            model.eval()
            with torch.no_grad():
                pred = model(X_cat, X_num)
            results.append({
                'Store_id': store_id,
                'Dept_id': dept,
                'Forecast_Week': 1,
                'Predicted_Weekly_Sales': pred.item()
            })
        else:
            current_seq = sub_df.iloc[-seq_length:].copy()
            for h in range(1, forecast_horizon + 1):
                X_cat = torch.tensor(current_seq[cat_cols].values[np.newaxis, :, :], dtype=torch.float32).to(device)
                X_num = torch.tensor(current_seq[num_cols].values[np.newaxis, :, :], dtype=torch.float32).to(device)
                model.eval()
                with torch.no_grad():
                    pred = model(X_cat, X_num)
                results.append({
                    'Store_id': store_id,
                    'Dept_id': dept,
                    'Forecast_Week': h,
                    'Predicted_Weekly_Sales': pred.item()
                })
                last_date = current_seq['Date'].iloc[-1]
                next_date = last_date + pd.Timedelta(weeks=1)
                next_row_df = sub_df[sub_df['Date'] == next_date]
                if not next_row_df.empty:
                    new_row = next_row_df.iloc[0].copy()
                else:
                    new_row = current_seq.iloc[-1].copy()
                    new_row['Date'] = next_date
                    new_row['Year'] = next_date.year
                    new_row['Month'] = next_date.month
                    new_row['Week'] = next_date.isocalendar().week
                    new_row['DayOfYear'] = next_date.timetuple().tm_yday
                new_row_df = pd.DataFrame([new_row])
                current_seq = pd.concat([current_seq, new_row_df], ignore_index=True)
                current_seq = current_seq.iloc[1:].reset_index(drop=True)
    return pd.DataFrame(results)
