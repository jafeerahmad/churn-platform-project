"""Generate a synthetic dataset using the selected Telco-style feature subset.

The output is for demonstrations and tests only; it is not the original Kaggle
Telco Customer Churn dataset and contains no real customer records.
"""
import numpy as np
import pandas as pd


def generate(n: int = 7000, seed: int = 42) -> pd.DataFrame:
    if n < 2:
        raise ValueError("At least two rows are required to generate a dataset.")
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .21, .24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.34, .44, .22])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    base = np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 50, 22))
    monthly = np.clip(base + rng.normal(0, 8, n), 18, 120).round(2)
    total = np.clip(monthly * tenure + rng.normal(0, 20, n), 0, None).round(2)
    senior = (rng.random(n) < .16).astype(int)
    partner = rng.choice(["Yes", "No"], n)
    dependents = rng.choice(["Yes", "No"], n, p=[.3, .7])
    paperless = rng.choice(["Yes", "No"], n, p=[.6, .4])

    log_odds = (
        -0.5 - 0.035 * tenure + 0.015 * (monthly - 60)
        + 1.0 * (contract == "Month-to-month")
        - 1.0 * (contract == "Two year")
        + 0.5 * (internet == "Fiber optic")
        + 0.4 * (payment == "Electronic check")
        + 0.3 * senior + 0.2 * (paperless == "Yes")
        - 0.3 * (partner == "Yes")
    )
    churn_probability = 1 / (1 + np.exp(-log_odds))
    churn = rng.random(n) < churn_probability

    return pd.DataFrame({
        "customerID": [f"C{100000 + index}" for index in range(n)],
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": np.where(churn, "Yes", "No"),
    })
