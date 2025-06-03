from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar modelo y scaler
model = joblib.load("modelo.joblib")
scaler = joblib.load("scaler.joblib")

app = FastAPI()

# Clase para recibir los datos
class DatosPaciente(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predecir(data: DatosPaciente):
    datos = [[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]]
    datos_escalados = scaler.transform(datos)
    resultado = model.predict(datos_escalados)[0]
    return {"prediccion": int(resultado)}
