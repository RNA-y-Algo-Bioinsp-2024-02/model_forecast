from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import torch
import joblib
import io
import numpy as np
from app.model import load_model  # Asegúrate de tener este módulo
from app.prediction import predict_store_custom
from app.utils import plot_predictions
from app.config import DEVICE

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a dominios específicos, por ejemplo: ["https://tudominio.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga el scaler guardado
SCALER_PATH = "models/scaler.pkl"
scaler = joblib.load(SCALER_PATH)

# Define las columnas numéricas a escalar (ejemplo)
# Asegúrate de que estas columnas sean las mismas que se usaron para entrenar el scaler.
numeric_cols_to_scale = ['Size', 'Temperature', 'Fuel_Price', 'Year', 'Month', 'Week', 'DayOfYear']

# Define tus columnas categóricas y numéricas completas (incluyendo IsHoliday sin escalar)
cat_cols = ['Store_id', 'Dept_id', 'Type_id']
all_num_cols = numeric_cols_to_scale + ['IsHoliday']

# Carga global del modelo (asegúrate de definir NUM_STORES, NUM_DEPTS, etc.)
NUM_STORES = 45       
NUM_DEPTS = 81        # Por ejemplo, si el modelo se entrenó con 81 departamentos
NUM_TYPES = 3        
NUM_NUMERIC_FEATURES = len(all_num_cols)  # Debe coincidir con el modelo
model = load_model(NUM_STORES, NUM_DEPTS, NUM_TYPES, NUM_NUMERIC_FEATURES)

# Modelo de request para el endpoint
class PredictionRequest(BaseModel):
    store_id: int
    dept_ids: list[int] = None
    forecast_horizon: int = 1
    seq_length: int = 8
    output_type: str = "data"  # "data" o "graph"
    historical_data: list[dict]

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        df_input = pd.DataFrame(request.historical_data)
        if 'Date' not in df_input.columns:
            raise HTTPException(status_code=400, detail="La data histórica debe incluir la columna 'Date'")
        df_input['Date'] = pd.to_datetime(df_input['Date'])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando la data histórica: {str(e)}")
    
    # Aplica la transformación a las variables numéricas (solamente a las que fueron escaladas)
    try:
        # Aseguramos que las columnas estén en el orden esperado.
        df_input[numeric_cols_to_scale] = scaler.transform(df_input[numeric_cols_to_scale])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al escalar las variables numéricas: {str(e)}")
    
    try:
        df_input['Store'] =  df_input['Store_id']+1
        df_input['Dept'] =  df_input['Dept_id']+1
        mapeo = {0: 'A', 1: 'B', 2: 'C'}
        df_input['Type'] = df_input['Type_id'].map(mapeo)       
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al estructurar el df_input: {str(e)}")
    
    # Genera las predicciones usando la función predict_store_custom
    pred_df = predict_store_custom(
        model=model,
        df=df_input,
        store_id=request.store_id,
        dept_ids=request.dept_ids,
        seq_length=request.seq_length,
        forecast_horizon=request.forecast_horizon,
        cat_cols=cat_cols,
        num_cols=all_num_cols,
        device=DEVICE
    )
    
    if request.output_type.lower() == "data":
        return JSONResponse(content=pred_df.to_dict(orient="records"))
    elif request.output_type.lower() == "graph":
        buf = plot_predictions(pred_df)
        return StreamingResponse(buf, media_type="image/png")
    else:
        raise HTTPException(status_code=400, detail="output_type debe ser 'data' o 'graph'")
