import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

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

    if "senior_citizen" in df.columns and (
        is_object_dtype(df["senior_citizen"]) or is_string_dtype(df["senior_citizen"])
    ):
        normalized = df["senior_citizen"].astype("string").str.strip().str.lower()
        df["senior_citizen"] = normalized.map(
            {
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
            }
        ).fillna(df["senior_citizen"])
        df["senior_citizen"] = pd.to_numeric(df["senior_citizen"], errors="coerce")

    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    if "monthly_charges" in df.columns:
        df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce")

    return df

def split_features_target(df: pd.DataFrame):
    df = df.copy()
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL] + [col for col in DROP_COLS if col in df.columns])
    return X, y
