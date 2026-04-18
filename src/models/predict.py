import logging

import joblib
import pandas as pd

from src.utils.paths import TRAINED_MODELS_DIR

MODEL_PATH = TRAINED_MODELS_DIR / "logistic_pipeline.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

def load_model():
    logger.info("Carregando modelo salvo em: %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)

def predict_single(input_data: dict):
    try:
        logger.info("Iniciando inferência")
        logger.info("Payload recebido: %s", input_data)

        model = load_model()
        df = pd.DataFrame([input_data])

        logger.info("Colunas recebidas pela API: %s", df.columns.tolist())
        logger.info("Quantidade de colunas recebidas: %d", len(df.columns))

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        logger.info(
            "Inferência concluída com sucesso | prediction=%s | churn_probability=%.4f",
            int(prediction),
            float(probability)
        )

        return {
            "prediction": int(prediction),
            "churn_probability": float(probability)
        }

    except Exception as e:
        logger.exception("Erro durante a inferência: %s", str(e))
        raise