
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "final_model"

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

    print(f"Chargement du modèle XGBoost depuis : {MODEL_PATH}")

    app.state.model = mlflow.xgboost.load_model(str(MODEL_PATH))

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
# Détermination du niveau de risque
# --------------------------------------------------

def determine_risk_level(probability: float) -> str:

    if probability < 0.15:
        return "Faible"

    elif probability < 0.50:
        return "Moyen"

    else:
        return "Élevé"


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
        # Préparation des données
        X = prepare_input(data)

        # Calcul de la probabilité
        probability = float(model.predict_proba(X)[0][1])

        # Décision selon le seuil de classification
        prediction = int(probability > THRESHOLD)

        # Calcul du niveau de risque
        risk_level = determine_risk_level(probability)

        return {
            "prediction": prediction,
            "failure_detected": bool(prediction),
            "failure_probability": round(probability, 4),
            "risk_level": risk_level,
            "threshold": THRESHOLD
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {str(error)}"
        )
