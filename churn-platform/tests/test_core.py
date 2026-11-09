import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.features import FEATURES, risk_level
from src.generate_data import generate
from src.simulator import make_events
from src.train import build, load


def test_risk_levels_and_boundaries():
    assert risk_level(0.0) == "Low"
    assert risk_level(0.2999) == "Low"
    assert risk_level(0.3) == "Medium"
    assert risk_level(0.5999) == "Medium"
    assert risk_level(0.6) == "High"
    assert risk_level(1.0) == "High"


def test_generated_data_schema_and_unique_ids():
    df = generate(n=500)
    assert len(df) == 500
    assert set(FEATURES + ["customerID", "Churn"]).issubset(df.columns)
    assert df["Churn"].isin(["Yes", "No"]).all()
    assert df["customerID"].is_unique


def test_pipeline_trains_and_predicts_unknown_category():
    df = generate(n=800)
    y = (df["Churn"] == "Yes").astype(int)
    pipe = build(RandomForestClassifier(n_estimators=20, random_state=0)).fit(df[FEATURES], y)
    probability = pipe.predict_proba(df[FEATURES].head(5))[:, 1]
    assert len(probability) == 5
    assert ((probability >= 0) & (probability <= 1)).all()

    row = df[FEATURES].head(1).copy()
    row["Contract"] = "New contract category"
    assert 0 <= pipe.predict_proba(row)[0, 1] <= 1


def test_load_fills_blank_total_charges_and_normalizes_churn(tmp_path, monkeypatch):
    from src import train

    frame = generate(n=120)
    frame.loc[0, "TotalCharges"] = np.nan
    frame["Churn"] = frame["Churn"].str.lower()
    csv_path = tmp_path / "telco_churn.csv"
    frame.to_csv(csv_path, index=False)
    monkeypatch.setattr(train, "RAW_CSV", csv_path)

    loaded = load()
    assert loaded.loc[0, "TotalCharges"] == 0
    assert set(loaded["Churn"].unique()) == {0, 1}
    assert len(loaded) == 120


def test_load_rejects_duplicate_customer_ids(tmp_path, monkeypatch):
    from src import train

    frame = generate(n=100)
    frame.loc[1, "customerID"] = frame.loc[0, "customerID"]
    csv_path = tmp_path / "telco_churn.csv"
    frame.to_csv(csv_path, index=False)
    monkeypatch.setattr(train, "RAW_CSV", csv_path)

    with pytest.raises(ValueError, match="duplicate"):
        load()


def test_event_generation_is_reproducible_and_valid():
    ids = np.array(["C1", "C2", "C3"])
    start = pd.Timestamp("2025-01-01 00:00:00").to_pydatetime()
    end = pd.Timestamp("2025-01-02 00:00:00").to_pydatetime()
    first = make_events(ids, 25, start, end, np.random.default_rng(10))
    second = make_events(ids, 25, start, end, np.random.default_rng(10))
    pd.testing.assert_frame_equal(first, second)
    assert set(first["event_type"]).issubset({"login", "order", "support_ticket", "cancellation"})
    assert (first.loc[first.event_type != "order", "amount"] == 0).all()
    assert (first.loc[first.event_type == "order", "amount"] >= 0).all()


def test_event_generation_rejects_invalid_customer_list():
    start = pd.Timestamp("2025-01-01").to_pydatetime()
    with pytest.raises(ValueError, match="customer ID"):
        make_events([], 10, start, start)


def test_load_rejects_invalid_numeric_text(tmp_path, monkeypatch):
    from src import train

    frame = generate(n=100)
    frame["MonthlyCharges"] = frame["MonthlyCharges"].astype(object)
    frame.loc[0, "MonthlyCharges"] = "not-a-number"
    csv_path = tmp_path / "telco_churn.csv"
    frame.to_csv(csv_path, index=False)
    monkeypatch.setattr(train, "RAW_CSV", csv_path)

    with pytest.raises(ValueError, match="invalid values"):
        load()
