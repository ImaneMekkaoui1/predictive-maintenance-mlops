
from fastapi.testclient import TestClient

from predictive_maintenance.api.main import app


def test_health_endpoint():
    """Vérifie que l'API démarre et que le modèle est chargé."""

    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


def test_prediction_low_risk():
    """Vérifie une prédiction avec un risque faible."""

    payload = {
        "type": "L",
        "air_temperature_k": 300.5,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1345,
        "torque_nm": 62.7,
        "tool_wear_min": 153
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

        assert response.status_code == 200

        data = response.json()

        assert data["risk_level"] == "Faible"
        assert data["prediction"] == 0
        assert data["failure_detected"] is False


def test_prediction_medium_risk():
    """Vérifie une prédiction avec un risque moyen."""

    payload = {
        "type": "M",
        "air_temperature_k": 302.9,
        "process_temperature_k": 311.2,
        "rotational_speed_rpm": 1330,
        "torque_nm": 53.7,
        "tool_wear_min": 200
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

        assert response.status_code == 200

        data = response.json()

        assert data["risk_level"] == "Moyen"
        assert data["prediction"] == 1
        assert data["failure_detected"] is True


def test_prediction_high_risk():
    """Vérifie une prédiction avec un risque élevé."""

    payload = {
        "type": "L",
        "air_temperature_k": 300.8,
        "process_temperature_k": 309.9,
        "rotational_speed_rpm": 1312,
        "torque_nm": 65.3,
        "tool_wear_min": 192
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

        assert response.status_code == 200

        data = response.json()

        assert data["risk_level"] == "Élevé"
        assert data["prediction"] == 1
        assert data["failure_detected"] is True






def test_prediction_negative_temperature():
    """Vérifie le rejet d'une température négative."""

    payload = {
        "type": "L",
        "air_temperature_k": -300.5,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1345,
        "torque_nm": 62.7,
        "tool_wear_min": 153
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

        assert response.status_code == 422


def test_prediction_invalid_machine_type():
    """Vérifie le rejet d'un type de machine invalide."""

    payload = {
        "type": "Z",
        "air_temperature_k": 300.5,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1345,
        "torque_nm": 62.7,
        "tool_wear_min": 153
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

        assert response.status_code == 422


def test_prediction_missing_field():
    """Vérifie le rejet d'un champ obligatoire manquant."""

    payload = {
        "type": "L",
        "air_temperature_k": 300.5,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1345,
        "torque_nm": 62.7
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

        assert response.status_code == 422


def test_metrics_endpoint():
    """Vérifie les métriques HTTP et les prédictions ML."""

    payload = {
        "type": "L",
        "air_temperature_k": 300.5,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1345,
        "torque_nm": 62.7,
        "tool_wear_min": 153
    }

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json=payload).status_code == 200

        response = client.get("/metrics")

        assert response.status_code == 200
        assert "http_requests_total" in response.text
        assert "http_request_duration_seconds_bucket" in response.text
        assert "maintenance_predictions_total" in response.text
        assert 'risk_level="Faible"' in response.text
