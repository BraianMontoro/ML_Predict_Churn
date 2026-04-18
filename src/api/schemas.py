from pydantic import BaseModel, Field


class ChurnInput(BaseModel):
    count: int
    country: str
    state: str
    city: str
    zip_code: str
    lat_long: str
    latitude: float
    longitude: float

    gender: str
    senior_citizen: int
    partner: str
    dependents: str
    tenure_months: int
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str

    monthly_charges: float = Field(alias="Monthly Charges")
    total_charges: float

    model_config = {
        "populate_by_name": True
    }


class PredictionResponse(BaseModel):
    prediction: int
    churn_probability: float
