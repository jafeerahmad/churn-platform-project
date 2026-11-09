NUM = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CAT = ["Partner", "Dependents", "InternetService", "Contract",
       "PaperlessBilling", "PaymentMethod"]
FEATURES = NUM + CAT


def risk_level(p: float) -> str:
    return "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"
