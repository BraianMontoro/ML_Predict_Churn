import pandas as pd

from src.data.column_mapping import COLUMN_MAPPING

TARGET_COL = "churn_value"

DROP_COLS = [
    "customer_id",
    "churn_label",
    "churn_score",
    "cltv",
    "churn_reason",
]

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns=COLUMN_MAPPING)
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = normalize_column_names(df)

    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    return df

def split_features_target(df: pd.DataFrame):
    df = df.copy()
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL] + [col for col in DROP_COLS if col in df.columns])
    return X, y