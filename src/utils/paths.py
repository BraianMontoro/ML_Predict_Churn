# src/utils/paths.py
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
MLRUNS_DIR = ROOT_DIR / "mlruns"
LOGISTIC_MODEL_PATH = TRAINED_MODELS_DIR / "logistic_pipeline.joblib"
MLP_MODEL_PATH = TRAINED_MODELS_DIR / "mlp_bundle.joblib"
