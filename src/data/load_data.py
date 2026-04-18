import pandas as pd

from src.utils.paths import RAW_DIR


def load_telco_data(filename: str = "Telco_customer_churn.xlsx") -> pd.DataFrame:
    file_path = RAW_DIR / filename
    df = pd.read_excel(file_path)
    return df