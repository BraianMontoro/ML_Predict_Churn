from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.utils.paths import MLP_MODEL_PATH

client = TestClient(app)

PAYLOAD = {
    "count": 1,
    "country": "United States",
    "state": "California",
    "city": "Los Angeles",
    "zip_code": "90001",
    "lat_long": "33.973616, -118.24902",
    "latitude": 33.973616,
    "longitude": -118.24902,
    "gender": "Male",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "Monthly Charges": 79.9,
    "total_charges": 958.8,
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "X-Process-Time-Ms" in response.headers


def test_predict():
    response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 200
    assert "X-Process-Time-Ms" in response.headers
    body = response.json()
    assert "prediction" in body
    assert "churn_probability" in body
    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_mlp():
    if not Path(MLP_MODEL_PATH).exists():
        pytest.skip("Bundle da MLP ainda não foi treinado no ambiente local.")

    response = client.post("/predict?model_name=mlp", json=PAYLOAD)

    assert response.status_code == 200
    assert "X-Process-Time-Ms" in response.headers
    body = response.json()
    assert "prediction" in body
    assert "churn_probability" in body
    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["churn_probability"] <= 1.0
