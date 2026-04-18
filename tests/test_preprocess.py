import pandas as pd

from src.data.feature_pipeline import prepare_features_frame, split_feature_types
from src.data.preprocess import basic_cleaning


def test_basic_cleaning_normalizes_columns_and_types():
    raw = pd.DataFrame([
        {
            "Senior Citizen": "Yes",
            "Monthly Charges": "79.9",
            "Total Charges": "958.8",
        }
    ])

    cleaned = basic_cleaning(raw)

    assert "monthly_charges" in cleaned.columns
    assert "total_charges" in cleaned.columns
    assert cleaned.loc[0, "senior_citizen"] == 1
    assert cleaned.loc[0, "monthly_charges"] == 79.9
    assert cleaned.loc[0, "total_charges"] == 958.8


def test_basic_cleaning_handles_pandas_string_dtype():
    raw = pd.DataFrame(
        {
            "Senior Citizen": pd.Series(["Yes"], dtype="string"),
            "Monthly Charges": pd.Series(["79.9"], dtype="string"),
            "Total Charges": pd.Series(["958.8"], dtype="string"),
        }
    )

    cleaned = basic_cleaning(raw)

    assert cleaned.loc[0, "senior_citizen"] == 1
    assert cleaned.loc[0, "monthly_charges"] == 79.9
    assert cleaned.loc[0, "total_charges"] == 958.8


def test_feature_pipeline_treats_zip_code_as_categorical():
    features = pd.DataFrame([
        {
            "zip_code": 90001,
            "monthly_charges": 79.9,
            "contract": "Month-to-month",
        }
    ])

    numerical_cols, categorical_cols = split_feature_types(features)

    assert "zip_code" not in numerical_cols
    assert "zip_code" in categorical_cols

    prepared = prepare_features_frame(features, categorical_cols)

    assert str(prepared["zip_code"].dtype) == "string"
