# app/utils.py

import matplotlib.pyplot as plt
import io
import numpy as np
from fastapi.responses import StreamingResponse

def plot_predictions(pred_df):
    unique_depts = pred_df['Dept_id'].unique()
    n_plots = len(unique_depts)
    ncols = 2
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*4))
    
    # Manejar correctamente los axes para cualquier número de subplots
    if n_plots == 1:
        axes = np.array([[axes]])  # Convertir a array 2D
    if nrows == 1:
        axes = axes.reshape(1, -1)  # Asegurar que sea 2D para una fila
    axes = axes.flatten()  # Aplanar para iterar

    for i, dept in enumerate(unique_depts):
        sub_df = pred_df[pred_df['Dept_id'] == dept].sort_values("Forecast_Week")
        ax = axes[i]
        ax.plot(sub_df['Forecast_Week'], sub_df['Predicted_Weekly_Sales'], marker="o", label="Predicho")
        ax.set_title(f"Dept {dept}")
        ax.set_xlabel("Semana")
        ax.set_ylabel("Ventas Semanales")
        ax.legend()

    # Ocultar los ejes sobrantes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf