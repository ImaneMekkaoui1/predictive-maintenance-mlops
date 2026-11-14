
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_URI = "models:/m-e117bda89bf74cd3b09486671b1f0e06"

THRESHOLD = 0.15

FEATURE_COLUMNS = [
    "Type_L",
    "Type_M",
    "Air temperature _K_",
    "Process temperature _K_",
    "Rotational speed _rpm_",
    "Torque _Nm_",
    "Tool wear _min_",
]


# --------------------------------------------------
# Chargement du modèle au démarrage
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    print("Chargement du modèle XGBoost depuis MLflow...")

    app.state.model = mlflow.xgboost.load_model(MODEL_URI)

    print("Modèle chargé avec succès !")

    yield

    print("Arrêt de l'API.")


app = FastAPI(
    title="Predictive Maintenance API",
    description="API de prédiction des défaillances industrielles avec XGBoost.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Schéma des données reçues
# --------------------------------------------------

class MachineData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    machine_type: Literal["L", "M", "H"] = Field(alias="type")

    air_temperature_k: float = Field(gt=0)
    process_temperature_k: float = Field(gt=0)
    rotational_speed_rpm: float = Field(gt=0)
    torque_nm: float = Field(gt=0)
    tool_wear_min: float = Field(ge=0)


# --------------------------------------------------
# Préparation des données
# --------------------------------------------------

def prepare_input(data: MachineData) -> pd.DataFrame:

    machine_features = {
        "Type_L": 1 if data.machine_type == "L" else 0,
        "Type_M": 1 if data.machine_type == "M" else 0,
        "Air temperature _K_": data.air_temperature_k,
        "Process temperature _K_": data.process_temperature_k,
        "Rotational speed _rpm_": data.rotational_speed_rpm,
        "Torque _Nm_": data.torque_nm,
        "Tool wear _min_": data.tool_wear_min,
    }

    return pd.DataFrame(
        [machine_features],
        columns=FEATURE_COLUMNS
    )


# --------------------------------------------------
# Routes API
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API",
        "status": "running",
        "documentation": "/docs"
    }


@app.get("/health")
def health(request: Request):
    model_loaded = hasattr(request.app.state, "model")

    return {
        "status": "healthy" if model_loaded else "unavailable",
        "model_loaded": model_loaded
    }


@app.post("/predict")
def predict(data: MachineData, request: Request):

    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas disponible."
        )

    try:
        X = prepare_input(data)

        probability = float(model.predict_proba(X)[0][1])

        prediction = int(probability > THRESHOLD)

        return {
            "prediction": prediction,
            "failure_detected": bool(prediction),
            "failure_probability": round(probability, 4),
            "threshold": THRESHOLD
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {str(error)}"
        )
