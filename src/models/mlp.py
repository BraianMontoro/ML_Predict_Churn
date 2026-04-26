import copy
import logging
import random
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch
import pandas as pd
from sklearn.compose import ColumnTransformer
from torch import nn

from src.data.feature_pipeline import build_preprocessor, prepare_features_frame
from src.data.schemas import validate_inference_frame

DEFAULT_HIDDEN_DIMS = (64, 32)
DEFAULT_THRESHOLD = 0.3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int] = DEFAULT_HIDDEN_DIMS):
        super().__init__()
        hidden_1, hidden_2 = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_dense_array(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def to_tensor(matrix) -> torch.Tensor:
    return torch.tensor(to_dense_array(matrix), dtype=torch.float32)


def to_target_tensor(target: pd.Series) -> torch.Tensor:
    return torch.tensor(target.to_numpy(), dtype=torch.float32).view(-1, 1)


def train_model(
    model: ChurnMLP,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
) -> dict:
    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for features_batch, target_batch in train_loader:
            features_batch = features_batch.to(device)
            target_batch = target_batch.to(device)

            outputs = model(features_batch)
            loss = criterion(outputs, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for features_batch, target_batch in val_loader:
                features_batch = features_batch.to(device)
                target_batch = target_batch.to(device)

                outputs = model(features_batch)
                loss = criterion(outputs, target_batch)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        logger.info(
            "Epoch %03d | Train Loss: %.4f | Val Loss: %.4f",
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("Early stopping ativado na época %d", epoch + 1)
            break

    model.load_state_dict(best_model_state)

    return {
        "best_val_loss": float(best_val_loss),
        "epochs_trained": len(train_losses),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def predict_probabilities(
    model: ChurnMLP,
    features_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        probabilities = model(features_tensor.to(device)).cpu().numpy().ravel()
    return probabilities


def save_mlp_bundle(
    bundle_path: Path,
    preprocessor: ColumnTransformer,
    model: ChurnMLP,
    input_dim: int,
    threshold: float,
    hidden_dims: tuple[int, int],
    numerical_cols: list[str],
    categorical_cols: list[str],
    metrics: dict,
) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model_type": "mlp_torch",
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "threshold": float(threshold),
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "preprocessor": preprocessor,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metrics": metrics,
    }
    joblib.dump(bundle, bundle_path)


@lru_cache(maxsize=1)
def load_mlp_bundle(bundle_path: Path) -> dict:
    logger.info("Carregando bundle da MLP em: %s", bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Bundle da MLP não encontrado em {bundle_path}. Execute o treino da MLP primeiro."
        )
    return joblib.load(bundle_path)


@lru_cache(maxsize=1)
def load_mlp_model(bundle_path: Path) -> tuple[dict, ChurnMLP]:
    bundle = load_mlp_bundle(bundle_path)
    model = ChurnMLP(
        input_dim=int(bundle["input_dim"]),
        hidden_dims=tuple(bundle["hidden_dims"]),
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return bundle, model


def predict_single_mlp(input_data: dict, bundle_path: Path) -> dict:
    logger.info("Iniciando inferência com MLP")
    logger.info("Payload recebido: %s", input_data)

    bundle, model = load_mlp_model(bundle_path)
    df = validate_inference_frame(pd.DataFrame([input_data]))
    prepared = prepare_features_frame(df, bundle["categorical_cols"])
    processed = bundle["preprocessor"].transform(prepared)
    features_tensor = to_tensor(processed)

    probability = float(
        predict_probabilities(model, features_tensor, device=torch.device("cpu"))[0]
    )
    prediction = int(probability >= float(bundle["threshold"]))

    logger.info(
        "Inferência MLP concluída com sucesso | prediction=%s | churn_probability=%.4f",
        prediction,
        probability,
    )

    return {
        "prediction": prediction,
        "churn_probability": probability,
    }
