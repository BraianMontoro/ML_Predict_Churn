import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numerical_cols = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    if "zip_code" in numerical_cols:
        numerical_cols.remove("zip_code")
        categorical_cols.append("zip_code")

    return numerical_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numerical_cols, categorical_cols = split_feature_types(X)

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

    return preprocessor, numerical_cols, categorical_cols


def prepare_features_frame(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    prepared = df.copy()
    existing_categorical = [col for col in categorical_cols if col in prepared.columns]
    if existing_categorical:
        prepared[existing_categorical] = prepared[existing_categorical].astype("string")
    return prepared
