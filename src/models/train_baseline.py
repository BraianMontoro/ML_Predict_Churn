import logging
import random

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import load_telco_data
from src.data.preprocess import basic_cleaning, split_features_target
from src.utils.paths import TRAINED_MODELS_DIR

EXPERIMENT_NAME = "telco_churn_baselines"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_pipeline(X):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def main():
    set_seed(RANDOM_STATE)
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Carregando dataset")
    df = load_telco_data()
    df = basic_cleaning(df)
    X, y = split_features_target(df)

    logger.info("Colunas usadas no treino: %s", X.columns.tolist())
    logger.info("Quantidade de colunas: %d", len(X.columns))

    pipeline = build_pipeline(X)

    logger.info("Executando validação cruzada estratificada")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    cv_metrics = {
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_precision_mean": float(np.mean(cv_results["test_precision"])),
        "cv_recall_mean": float(np.mean(cv_results["test_recall"])),
        "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
        "cv_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
    }

    logger.info("Métricas médias da CV: %s", cv_metrics)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info("Treinando pipeline final no conjunto de treino")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    holdout_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
    }

    logger.info("Métricas no holdout: %s", holdout_metrics)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="logistic_regression_stage3"):
        mlflow.log_params({
            "model": "LogisticRegression",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_n_splits": N_SPLITS,
        })
        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics(holdout_metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    model_path = TRAINED_MODELS_DIR / "logistic_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    logger.info("Modelo salvo em: %s", model_path)


if __name__ == "__main__":
    main()