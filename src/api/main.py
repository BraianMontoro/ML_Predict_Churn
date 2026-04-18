from fastapi import FastAPI

from src.api.schemas import ChurnInput
from src.models.predict import predict_single

app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0.0",
    description="API para inferência de churn em clientes de telecom."
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: ChurnInput):
    result = predict_single(payload.model_dump(by_alias=True))
    return result