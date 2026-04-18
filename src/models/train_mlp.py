import logging
import json

import mlflow
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.feature_pipeline import build_preprocessor, prepare_features_frame
from src.data.load_data import get_telco_data_metadata, load_telco_data
from src.data.preprocess import basic_cleaning, split_features_target
from src.data.schemas import validate_training_data
from src.models.mlp import (
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_THRESHOLD,
    ChurnMLP,
    predict_probabilities,
    save_mlp_bundle,
    set_seed,
    to_target_tensor,
    to_tensor,
    train_model,
)
from src.utils.paths import ARTIFACTS_DIR, MLP_MODEL_PATH, MLRUNS_DIR

EXPERIMENT_NAME = "telco_churn_baselines"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    set_seed(RANDOM_STATE)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    MLP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Carregando dataset para treino da MLP")
    df = basic_cleaning(load_telco_data())
    dataset_metadata = get_telco_data_metadata()
    X, y = split_features_target(df)
    X, y = validate_training_data(X, y)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    preprocessor, numerical_cols, categorical_cols = build_preprocessor(X_train)
    X_train_prepared = prepare_features_frame(X_train, categorical_cols)
    X_val_prepared = prepare_features_frame(X_val, categorical_cols)
    X_test_prepared = prepare_features_frame(X_test, categorical_cols)

    X_train_processed = preprocessor.fit_transform(X_train_prepared)
    X_val_processed = preprocessor.transform(X_val_prepared)
    X_test_processed = preprocessor.transform(X_test_prepared)

    X_train_tensor = to_tensor(X_train_processed)
    X_val_tensor = to_tensor(X_val_processed)
    X_test_tensor = to_tensor(X_test_processed)
    y_train_tensor = to_target_tensor(y_train)
    y_val_tensor = to_target_tensor(y_val)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_tensor.shape[1]
    model = ChurnMLP(input_dim=input_dim, hidden_dims=DEFAULT_HIDDEN_DIMS).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info("Treinando MLP em dispositivo: %s", device)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        patience=PATIENCE,
    )

    threshold = DEFAULT_THRESHOLD
    y_pred_prob = predict_probabilities(model, X_test_tensor, device=device)
    y_pred = (y_pred_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_pred_prob)),
    }
    logger.info("Métricas da MLP no holdout: %s", metrics)

    save_mlp_bundle(
        bundle_path=MLP_MODEL_PATH,
        preprocessor=preprocessor,
        model=model.to(torch.device("cpu")),
        input_dim=input_dim,
        threshold=threshold,
        hidden_dims=DEFAULT_HIDDEN_DIMS,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        metrics=metrics,
    )
    logger.info("Bundle da MLP salvo em: %s", MLP_MODEL_PATH)

    history_path = ARTIFACTS_DIR / "mlp_training_history.json"
    history_payload = {
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
        "metrics": metrics,
        "threshold": threshold,
        "feature_columns": list(X.columns),
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="mlp_stage3"):
        mlflow.log_params({
            "model": "MLP",
            "input_dim": input_dim,
            "hidden_layer_1": DEFAULT_HIDDEN_DIMS[0],
            "hidden_layer_2": DEFAULT_HIDDEN_DIMS[1],
            "activation": "ReLU",
            "loss_function": "BCELoss",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "epochs_max": EPOCHS,
            "epochs_trained": history["epochs_trained"],
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
            "threshold": threshold,
            "test_size": TEST_SIZE,
            "validation_size": VALIDATION_SIZE,
            "random_state": RANDOM_STATE,
            "feature_count": len(X.columns),
            **dataset_metadata,
        })
        mlflow.log_metrics({
            **metrics,
            "best_val_loss": history["best_val_loss"],
        })
        mlflow.log_artifact(str(MLP_MODEL_PATH), artifact_path="model_bundle")
        mlflow.log_artifact(str(history_path), artifact_path="training")


if __name__ == "__main__":
    main()
