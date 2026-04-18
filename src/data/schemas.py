import pandas as pd
import pandera.pandas as pa

from src.data.preprocess import TARGET_COL, normalize_column_names

STRING_COLUMNS = [
    "country",
    "state",
    "city",
    "zip_code",
    "lat_long",
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

NUMERIC_COLUMNS = [
    "count",
    "latitude",
    "longitude",
    "senior_citizen",
    "tenure_months",
    "monthly_charges",
    "total_charges",
]


INFERENCE_SCHEMA = pa.DataFrameSchema(
    columns={
        "count": pa.Column(int, checks=pa.Check.ge(0), nullable=False, coerce=True),
        "country": pa.Column(str, nullable=False, coerce=True),
        "state": pa.Column(str, nullable=False, coerce=True),
        "city": pa.Column(str, nullable=False, coerce=True),
        "zip_code": pa.Column(str, nullable=False, coerce=True),
        "lat_long": pa.Column(str, nullable=False, coerce=True),
        "latitude": pa.Column(
            float,
            checks=[pa.Check.ge(-90), pa.Check.le(90)],
            nullable=False,
            coerce=True,
        ),
        "longitude": pa.Column(
            float,
            checks=[pa.Check.ge(-180), pa.Check.le(180)],
            nullable=False,
            coerce=True,
        ),
        "gender": pa.Column(str, nullable=False, coerce=True),
        "senior_citizen": pa.Column(
            int,
            checks=pa.Check.isin([0, 1]),
            nullable=False,
            coerce=True,
        ),
        "partner": pa.Column(str, nullable=False, coerce=True),
        "dependents": pa.Column(str, nullable=False, coerce=True),
        "tenure_months": pa.Column(int, checks=pa.Check.ge(0), nullable=False, coerce=True),
        "phone_service": pa.Column(str, nullable=False, coerce=True),
        "multiple_lines": pa.Column(str, nullable=False, coerce=True),
        "internet_service": pa.Column(str, nullable=False, coerce=True),
        "online_security": pa.Column(str, nullable=False, coerce=True),
        "online_backup": pa.Column(str, nullable=False, coerce=True),
        "device_protection": pa.Column(str, nullable=False, coerce=True),
        "tech_support": pa.Column(str, nullable=False, coerce=True),
        "streaming_tv": pa.Column(str, nullable=False, coerce=True),
        "streaming_movies": pa.Column(str, nullable=False, coerce=True),
        "contract": pa.Column(str, nullable=False, coerce=True),
        "paperless_billing": pa.Column(str, nullable=False, coerce=True),
        "payment_method": pa.Column(str, nullable=False, coerce=True),
        "monthly_charges": pa.Column(float, checks=pa.Check.ge(0), nullable=False, coerce=True),
        "total_charges": pa.Column(float, checks=pa.Check.ge(0), nullable=True, coerce=True),
    },
    strict=True,
    coerce=True,
)

TARGET_SCHEMA = pa.SeriesSchema(
    int,
    checks=pa.Check.isin([0, 1]),
    nullable=False,
    coerce=True,
    name=TARGET_COL,
)


def _prepare_frame_for_validation(df: pd.DataFrame) -> pd.DataFrame:
    prepared = normalize_column_names(df.copy())

    duplicated_columns = prepared.columns[prepared.columns.duplicated()].tolist()
    if duplicated_columns:
        duplicated = ", ".join(sorted(set(duplicated_columns)))
        raise ValueError(
            "Foram encontradas colunas duplicadas apos a normalizacao: "
            f"{duplicated}"
        )

    existing_string_columns = [column for column in STRING_COLUMNS if column in prepared.columns]
    for column in existing_string_columns:
        prepared[column] = prepared[column].astype("string")

    if "senior_citizen" in prepared.columns:
        senior_series = prepared["senior_citizen"]
        if senior_series.dtype == "string" or senior_series.dtype == object:
            normalized = senior_series.astype("string").str.strip().str.lower()
            prepared["senior_citizen"] = normalized.map({
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
            }).fillna(senior_series)

    existing_numeric_columns = [column for column in NUMERIC_COLUMNS if column in prepared.columns]
    for column in existing_numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    return prepared


def validate_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_frame_for_validation(df)
    return INFERENCE_SCHEMA.validate(prepared, lazy=True)


def validate_training_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    validated_X = validate_inference_frame(X)
    validated_y = TARGET_SCHEMA.validate(pd.Series(y, name=TARGET_COL), lazy=True)
    return validated_X, validated_y
