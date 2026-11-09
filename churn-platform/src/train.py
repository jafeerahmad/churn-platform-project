"""Train and evaluate churn models, persist artifacts, and load customers into SQL.

Run with: python -m src.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import METRICS_PATH, MODEL_PATH, RAW_CSV, ROOT
from .features import CAT, FEATURES, NUM
from .generate_data import generate

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load() -> pd.DataFrame:
    """Load the bundled CSV or create it once; validate and normalize its schema."""
    if not RAW_CSV.exists():
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        print("No data/telco_churn.csv found; creating a synthetic 7,000-row dataset.")
        generate().to_csv(RAW_CSV, index=False)

    df = pd.read_csv(RAW_CSV)
    required = {"customerID", *FEATURES, "Churn"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")

    df = df.copy()
    df["customerID"] = df["customerID"].astype("string").str.strip()
    if df["customerID"].isna().any() or (df["customerID"] == "").any():
        raise ValueError("customerID contains blank values.")
    if df["customerID"].duplicated().any():
        duplicate_count = int(df["customerID"].duplicated().sum())
        raise ValueError(f"customerID contains {duplicate_count} duplicate rows.")

    # Blank TotalCharges cells occur in some public Telco CSV copies for zero-tenure customers.
    invalid_numeric = []
    for column in NUM:
        original = df[column]
        parsed = pd.to_numeric(original, errors="coerce")
        nonblank_invalid = parsed.isna() & original.notna() & original.astype("string").str.strip().ne("")
        if nonblank_invalid.any():
            invalid_numeric.append(column)
        df[column] = parsed
    if invalid_numeric:
        raise ValueError(
            f"Numeric fields contain invalid values: {', '.join(invalid_numeric)}"
        )
    required_numeric = [column for column in NUM if column != "TotalCharges"]
    missing_numeric = [column for column in required_numeric if df[column].isna().any()]
    if missing_numeric:
        raise ValueError(
            f"Numeric fields contain missing values: {', '.join(missing_numeric)}"
        )
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    for column in NUM:
        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite numeric values.")

    if not df["SeniorCitizen"].isin([0, 1]).all():
        raise ValueError("SeniorCitizen must contain only 0 or 1.")
    if not df["tenure"].between(0, 72).all():
        raise ValueError("tenure must be between 0 and 72 months.")
    if (df[["MonthlyCharges", "TotalCharges"]] < 0).any().any():
        raise ValueError("MonthlyCharges and TotalCharges cannot be negative.")

    for column in CAT:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].fillna("Unknown").replace("", "Unknown")

    churn = df["Churn"].astype("string").str.strip().str.lower()
    churn = churn.replace({"1": "yes", "0": "no"})
    if not churn.isin(["yes", "no"]).all():
        invalid = sorted(churn.loc[~churn.isin(["yes", "no"])].dropna().unique().tolist())
        raise ValueError(f"Churn must contain Yes/No (or 1/0); invalid values: {invalid[:5]}")
    df["Churn"] = churn.map({"yes": 1, "no": 0}).astype(int)

    if df["Churn"].nunique() != 2:
        raise ValueError("Training requires both churn classes (Yes and No).")
    return df


def build(model) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", preprocessor), ("clf", model)])


def _evaluate(y_true: pd.Series, probability: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(y_true, probability),
        "specificity": (tn / (tn + fp)) if (tn + fp) else 0.0,
    }
    matrix = np.array([[tn, fp], [fn, tp]], dtype=int)
    return {key: round(float(value), 4) for key, value in metrics.items()}, matrix, prediction


def main() -> dict:
    from .db import engine

    df = load()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["Churn"], test_size=TEST_SIZE,
        stratify=df["Churn"], random_state=RANDOM_STATE,
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
    }

    results: dict[str, dict] = {}
    pipelines: dict[str, Pipeline] = {}
    prediction_rows: list[pd.DataFrame] = []
    roc_rows: list[pd.DataFrame] = []
    best_name, best_pipe, best_auc = None, None, -1.0

    for name, estimator in candidates.items():
        pipeline = build(estimator).fit(X_train, y_train)
        probability = pipeline.predict_proba(X_test)[:, 1]
        metrics, matrix, prediction = _evaluate(y_test, probability)
        results[name] = {
            **metrics,
            "confusion_matrix": {
                "true_negative": int(matrix[0, 0]),
                "false_positive": int(matrix[0, 1]),
                "false_negative": int(matrix[1, 0]),
                "true_positive": int(matrix[1, 1]),
            },
        }
        pipelines[name] = pipeline
        prediction_rows.append(pd.DataFrame({
            "customerID": df.loc[X_test.index, "customerID"].to_numpy(),
            "actual_churn": y_test.to_numpy(),
            "model": name,
            "churn_probability": probability,
            "predicted_churn_at_0_5": prediction,
        }))
        fpr, tpr, thresholds = roc_curve(y_test, probability)
        roc_rows.append(pd.DataFrame({
            "model": name, "false_positive_rate": fpr,
            "true_positive_rate": tpr, "threshold": thresholds,
        }))
        print(f"{name}: {metrics}")
        if metrics["roc_auc"] > best_auc:
            best_name, best_pipe, best_auc = name, pipeline, metrics["roc_auc"]

    assert best_name is not None and best_pipe is not None
    fitted_classifier = best_pipe.named_steps["clf"]
    feature_names = best_pipe.named_steps["pre"].get_feature_names_out()
    if hasattr(fitted_classifier, "feature_importances_"):
        signed_effect = np.asarray(fitted_classifier.feature_importances_, dtype=float)
        effect_type = "random_forest_importance"
    else:
        signed_effect = np.asarray(fitted_classifier.coef_[0], dtype=float)
        effect_type = "absolute_logistic_coefficient"

    importance = pd.DataFrame({
        "feature": [name.split("__", 1)[-1] for name in feature_names],
        "importance": np.abs(signed_effect),
        "signed_effect": signed_effect,
        "effect_type": effect_type,
    }).sort_values("importance", ascending=False)
    top = importance.head(10)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipe, MODEL_PATH)
    metrics_document = {
        "best_model": best_name,
        "selection_metric": "roc_auc",
        "decision_threshold": 0.5,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "dataset": {
            "rows": int(len(df)),
            "customers": int(df["customerID"].nunique()),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "churned_customers": int(df["Churn"].sum()),
            "non_churned_customers": int((df["Churn"] == 0).sum()),
            "churn_rate": round(float(df["Churn"].mean()), 4),
            "source": "data/telco_churn.csv; check data/DATA_DICTIONARY.md for provenance",
        },
        "results": results,
        "top_features": [[str(row.feature), round(float(row.importance), 4)] for row in top.itertuples()],
        "interpretation_note": (
            "Offline holdout metrics on this dataset only. They are not evidence of real-world "
            "retention uplift, causal impact, or production performance."
        ),
    }
    json_text = json.dumps(metrics_document, indent=2)
    METRICS_PATH.write_text(json_text, encoding="utf-8")
    (results_dir / "metrics.json").write_text(json_text, encoding="utf-8")

    pd.DataFrame([
        {"model": name, **{k: v for k, v in values.items() if k != "confusion_matrix"}}
        for name, values in results.items()
    ]).to_csv(results_dir / "model_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        results_dir / "test_predictions.csv", index=False)
    pd.concat(roc_rows, ignore_index=True).to_csv(
        results_dir / "roc_curve_points.csv", index=False)
    importance.to_csv(results_dir / "feature_importance.csv", index=False)
    df["Churn"].value_counts().rename_axis("churn").reset_index(name="customers").to_csv(
        results_dir / "churn_distribution.csv", index=False)

    customer_table = df[["customerID"] + FEATURES + ["Churn"]].copy()
    customer_table.to_sql("customers", engine, if_exists="replace", index=False)

    print(
        f"Best model: {best_name} (ROC-AUC {best_auc:.4f}); "
        f"{len(df):,} customers loaded into the database."
    )
    print(f"Metrics: {METRICS_PATH}")
    print(f"Result exports: {results_dir}")
    return metrics_document


if __name__ == "__main__":
    main()
