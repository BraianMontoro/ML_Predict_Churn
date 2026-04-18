import logging
import random

import joblib
import mlflow
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.feature_pipeline import build_preprocessor
from src.data.load_data import get_telco_data_metadata, load_telco_data
from src.data.preprocess import basic_cleaning, split_features_target
from src.data.schemas import validate_training_data
from src.utils.paths import ARTIFACTS_DIR, MLRUNS_DIR

EXPERIMENT_NAME = "telco_churn_baselines"
RANDOM_STATE = 42
TEST_SIZE = 0.2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)

def evaluate_model(name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics["auc"] = float(roc_auc_score(y_test, y_proba))

    logger.info("Metricas do baseline %s: %s", name, metrics)
    return metrics


def log_baseline_run(
    run_name: str,
    params: dict,
    metrics: dict,
    pipeline: Pipeline,
    dataset_metadata: dict[str, str | int],
) -> None:
    artifact_path = ARTIFACTS_DIR / f"{run_name}.joblib"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**params, **dataset_metadata})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(artifact_path), artifact_path="model")


def main() -> None:
    set_seed(RANDOM_STATE)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Carregando dataset para registro dos baselines")
    df = basic_cleaning(load_telco_data())
    dataset_metadata = get_telco_data_metadata()
    X, y = split_features_target(df)
    X, y = validate_training_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    baselines = [
        (
            "dummy_classifier",
            DummyClassifier(strategy="most_frequent"),
            {
                "model": "DummyClassifier",
                "strategy": "most_frequent",
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            },
        ),
        (
            "logistic_regression",
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "model": "LogisticRegression",
                "max_iter": 1000,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            },
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
            ),
            {
                "model": "RandomForestClassifier",
                "n_estimators": 200,
                "max_depth": "None",
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            },
        ),
    ]

    for run_name, estimator, params in baselines:
        logger.info("Treinando baseline: %s", run_name)
        preprocessor, _, _ = build_preprocessor(X_train)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(run_name, pipeline, X_test, y_test)
        log_baseline_run(run_name, params, metrics, pipeline, dataset_metadata)


if __name__ == "__main__":
    main()
