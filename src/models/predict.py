import logging

import joblib
import pandas as pd

from src.data.schemas import validate_inference_frame
from src.utils.paths import LOGISTIC_MODEL_PATH, MLP_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_logistic_model():
    logger.info("Carregando modelo salvo em: %s", LOGISTIC_MODEL_PATH)
    if not LOGISTIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo logistico nao encontrado em {LOGISTIC_MODEL_PATH}. Execute o treino primeiro."
        )
    return joblib.load(LOGISTIC_MODEL_PATH)


def predict_single_logistic(input_data: dict):
    logger.info("Iniciando inferencia com regressao logistica")
    logger.info("Payload recebido: %s", input_data)

    model = load_logistic_model()
    df = validate_inference_frame(pd.DataFrame([input_data]))

    logger.info("Colunas recebidas pela API: %s", df.columns.tolist())
    logger.info("Quantidade de colunas recebidas: %d", len(df.columns))

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    logger.info(
        "Inferencia concluida com sucesso | prediction=%s | churn_probability=%.4f",
        int(prediction),
        float(probability),
    )

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
    }


def predict_single(input_data: dict, model_name: str = "logistic"):
    try:
        if model_name == "logistic":
            return predict_single_logistic(input_data)
        if model_name == "mlp":
            # Lazy import keeps the default API startup lighter for cloud deploys.
            try:
                from src.models.mlp import predict_single_mlp
            except ImportError as exc:
                raise RuntimeError(
                    "Dependencias da MLP nao estao disponiveis neste ambiente. "
                    "Use model_name=logistic ou instale os pacotes de treino."
                ) from exc

            return predict_single_mlp(input_data, MLP_MODEL_PATH)
        raise ValueError(f"Modelo invalido: {model_name}")
    except Exception as exc:
        logger.exception("Erro durante a inferencia: %s", str(exc))
        raise
