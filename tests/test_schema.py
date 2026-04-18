import pandas as pd
import pandera.pandas as pa
import pytest

from src.api.schemas import ChurnInput
from src.data.schemas import validate_inference_frame

PAYLOAD = {
    "count": 1,
    "country": "United States",
    "state": "California",
    "city": "Los Angeles",
    "zip_code": "90001",
    "lat_long": "33.973616, -118.24902",
    "latitude": 33.973616,
    "longitude": -118.24902,
    "gender": "Male",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "Monthly Charges": 79.9,
    "total_charges": 958.8,
}


def test_schema_valid():
    obj = ChurnInput(**PAYLOAD)
    validated = validate_inference_frame(pd.DataFrame([PAYLOAD]))

    assert obj.monthly_charges == 79.9
    assert obj.country == "United States"
    assert validated.iloc[0]["country"] == "United States"
    assert validated.iloc[0]["monthly_charges"] == 79.9


def test_schema_accepts_canonical_feature_names():
    canonical_payload = PAYLOAD.copy()
    canonical_payload["monthly_charges"] = canonical_payload.pop("Monthly Charges")

    validated = validate_inference_frame(pd.DataFrame([canonical_payload]))

    assert validated.iloc[0]["monthly_charges"] == 79.9


def test_schema_invalid_with_pandera():
    invalid_payload = PAYLOAD | {"senior_citizen": 3}

    with pytest.raises(pa.errors.SchemaErrors):
        validate_inference_frame(pd.DataFrame([invalid_payload]))
