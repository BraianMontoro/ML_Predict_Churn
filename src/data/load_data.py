from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.paths import RAW_DIR

DEFAULT_TELCO_FILENAME = "Telco_customer_churn.xlsx"


def get_telco_data_path(filename: str = DEFAULT_TELCO_FILENAME) -> Path:
    return RAW_DIR / filename


def get_telco_data_metadata(filename: str = DEFAULT_TELCO_FILENAME) -> dict[str, str | int]:
    file_path = get_telco_data_path(filename)
    file_stats = file_path.stat()
    return {
        "dataset_filename": file_path.name,
        "dataset_path": str(file_path),
        "dataset_size_bytes": int(file_stats.st_size),
        "dataset_modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(
            timespec="seconds"
        ),
    }


def load_telco_data(filename: str = DEFAULT_TELCO_FILENAME) -> pd.DataFrame:
    file_path = get_telco_data_path(filename)
    df = pd.read_excel(file_path)
    return df
