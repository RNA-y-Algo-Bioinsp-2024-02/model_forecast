# app/utils.py

import matplotlib.pyplot as plt
import io
import numpy as np
from fastapi.responses import StreamingResponse

def plot_predictions(pred_df):
    import matplotlib.pyplot as plt
    import io

    unique_depts = pred_df['Dept_id'].unique()
    n_plots = len(unique_depts)
    ncols = 2
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*4))
    
    # Si solo hay un subplot, conviértelo a una lista
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, dept in enumerate(unique_depts):
        sub_df = pred_df[pred_df['Dept_id'] == dept].sort_values("Forecast_Week")
        ax = axes[i]  # Aquí ax es un objeto Axes
        ax.plot(sub_df['Forecast_Week'], sub_df['Predicted_Weekly_Sales'], marker="o", label="Predicho")
        ax.set_title(f"Dept {dept}")
        ax.set_xlabel("Semana")
        ax.set_ylabel("Ventas Semanales")
        ax.legend()

    # Ocultar los ejes sobrantes, si existen
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf


