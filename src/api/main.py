import logging
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, Request

from src.api.schemas import ChurnInput, PredictionResponse
from src.models.predict import predict_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0.0",
    description="API para inferencia de churn em clientes de telecom.",
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = perf_counter()
    response = await call_next(request)
    elapsed_ms = (perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info(
        "request_completed path=%s method=%s status=%s latency_ms=%.2f",
        request.url.path,
        request.method,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(
    payload: ChurnInput,
    model_name: Literal["logistic", "mlp"] = "logistic",
) -> PredictionResponse:
    result = predict_single(payload.model_dump(), model_name=model_name)
    return PredictionResponse(**result)
