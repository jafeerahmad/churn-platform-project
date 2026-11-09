"""FastAPI service. Run with `uvicorn src.api:app --reload`; docs at `/docs`."""
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from .db import engine
from .features import FEATURES, risk_level
from .predict import get_model, load_metrics, score

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict churn risk using the project's trained demonstration model.",
    version="1.1.0",
)


class Customer(BaseModel):
    SeniorCitizen: int = Field(0, ge=0, le=1)
    tenure: int = Field(12, ge=0, le=72)
    MonthlyCharges: float = Field(70.0, ge=0, le=200)
    TotalCharges: float = Field(840.0, ge=0)
    Partner: str = "No"
    Dependents: str = "No"
    InternetService: str = "Fiber optic"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"


@app.get("/health")
def health():
    try:
        get_model()
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model or database is unavailable.") from exc
    return {"status": "ok", "best_model": load_metrics().get("best_model")}


@app.post("/predict")
def predict(customer: Customer):
    probability = float(score(pd.DataFrame([customer.model_dump()]))[0])
    return {
        "churn_probability": round(probability, 4),
        "risk_level": risk_level(probability),
        "decision_threshold": 0.5,
    }


@app.get("/customers/at-risk")
def at_risk(limit: int = Query(default=20, ge=1, le=100)):
    customers = pd.read_sql("SELECT * FROM customers", engine)
    customers["churn_probability"] = score(customers[FEATURES]).round(4)
    top = customers.sort_values("churn_probability", ascending=False).head(limit)
    return top[["customerID", "Contract", "tenure", "MonthlyCharges",
                "churn_probability"]].to_dict("records")
