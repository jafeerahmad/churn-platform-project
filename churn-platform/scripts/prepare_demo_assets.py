"""Build the bundled sample event data, customer scores, and report charts.

Run from the repository root:
    python scripts/prepare_demo_assets.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from src.features import FEATURES, risk_level
from src.simulator import make_events

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SCREENSHOTS_DIR = ROOT / "docs" / "screenshots"


def save_figure(fig, filename: str):
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(DATA_DIR / "telco_churn.csv")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    model = joblib.load(ROOT / "models" / "churn_model.joblib")

    # A fixed timestamp and seed keep the committed sample export reproducible.
    end = datetime(2025, 1, 31, 23, 59, 59)
    ids = customers["customerID"].to_numpy()
    rng = np.random.default_rng(20261109)
    daily_frames = []
    for day_number in range(30, 0, -1):
        day_start = end - timedelta(days=day_number)
        daily_frames.append(make_events(ids, 1500, day_start, day_start + timedelta(days=1), rng))
    events = pd.concat(daily_frames, ignore_index=True)
    events = events.sort_values("event_time").reset_index(drop=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    events.to_csv(DATA_DIR / "events_sample.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    scored = customers.copy()
    scored["churn_probability"] = model.predict_proba(scored[FEATURES])[:, 1]
    scored["risk_level"] = scored["churn_probability"].map(risk_level)
    scored[["customerID", "Contract", "tenure", "MonthlyCharges", "Churn",
            "churn_probability", "risk_level"]].to_csv(
        DATA_DIR / "customer_risk_scores.csv", index=False
    )

    # 1. Class distribution.
    counts = customers["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(["No churn", "Churn"], counts.values)
    ax.set_title("Customer churn class distribution")
    ax.set_ylabel("Customers")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:,}",
                ha="center", va="bottom")
    save_figure(fig, "churn_distribution.png")

    # 2. Comparison metrics.
    metric_df = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
    plotted_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(plotted_metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, row in metric_df.iterrows():
        ax.bar(x + (index - 0.5) * width, [row[key] for key in plotted_metrics],
               width, label=row["model"].replace("_", " ").title())
    ax.set_xticks(x, [label.replace("_", " ").title() for label in plotted_metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Holdout model comparison (threshold 0.50)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "model_comparison.png")

    # 3. ROC curves.
    roc_data = pd.read_csv(RESULTS_DIR / "roc_curve_points.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, group in roc_data.groupby("model"):
        auc = metrics["results"][name]["roc_auc"]
        ax.plot(group["false_positive_rate"], group["true_positive_rate"],
                label=f"{name.replace('_', ' ').title()} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    save_figure(fig, "roc_curves.png")

    # 4. Confusion matrices.
    names = list(metrics["results"].keys())
    fig, axes = plt.subplots(1, len(names), figsize=(10, 4.3))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        cm = metrics["results"][name]["confusion_matrix"]
        matrix = np.array([[cm["true_negative"], cm["false_positive"]],
                           [cm["false_negative"], cm["true_positive"]]])
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xticks([0, 1], ["Predicted no", "Predicted yes"])
        ax.set_yticks([0, 1], ["Actual no", "Actual yes"])
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:,}", ha="center", va="center")
    fig.suptitle("Confusion matrices (decision threshold 0.50)")
    save_figure(fig, "confusion_matrices.png")

    # 5. Top features; for the selected model this is absolute coefficient size.
    features = pd.read_csv(RESULTS_DIR / "feature_importance.csv").head(10).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(features["feature"], features["importance"])
    ax.set_xlabel("Absolute coefficient magnitude")
    ax.set_title("Top churn model features (selected Logistic Regression model)")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, "top_churn_drivers.png")

    # 6. Simulated event activity and order value.
    events["event_time"] = pd.to_datetime(events["event_time"])
    event_counts = events.groupby("event_type").size().sort_values(ascending=False)
    daily_order_value = (
        events[events["event_type"] == "order"].set_index("event_time")["amount"]
        .resample("1D").sum()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(event_counts.index, event_counts.values)
    axes[0].set_title("Sample events by type")
    axes[0].set_ylabel("Events")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].plot(daily_order_value.index, daily_order_value.values, marker="o", markersize=3)
    axes[1].set_title("Daily simulated order value")
    axes[1].set_ylabel("Amount (synthetic currency units)")
    axes[1].tick_params(axis="x", rotation=25)
    for ax in axes:
        ax.grid(alpha=0.25)
    save_figure(fig, "event_activity.png")

    # 7. Risk distribution and concise dashboard-style result summary.
    risk_counts = scored["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    best = metrics["best_model"]
    best_metrics = metrics["results"][best]
    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.65, 1.35])
    top_ax = fig.add_subplot(grid[0, :])
    top_ax.axis("off")
    top_ax.text(0.02, 0.68, f"{len(customers):,} customers", fontsize=20, weight="bold")
    top_ax.text(0.36, 0.68, f"{len(events):,} sample events", fontsize=20, weight="bold")
    top_ax.text(0.72, 0.68, f"{metrics['dataset']['churn_rate']:.1%} churn rate", fontsize=20, weight="bold")
    top_ax.text(0.02, 0.22,
                f"Selected model: {best.replace('_', ' ').title()}  |  "
                f"ROC-AUC: {best_metrics['roc_auc']:.4f}  |  "
                f"F1: {best_metrics['f1']:.4f}", fontsize=12)

    risk_ax = fig.add_subplot(grid[1, 0])
    risk_ax.bar(risk_counts.index, risk_counts.values)
    risk_ax.set_title("Customers by predicted risk band")
    risk_ax.set_ylabel("Customers")
    risk_ax.grid(axis="y", alpha=0.25)

    revenue_ax = fig.add_subplot(grid[1, 1])
    revenue_ax.plot(daily_order_value.index, daily_order_value.values, marker="o", markersize=3)
    revenue_ax.set_title("Daily simulated order value")
    revenue_ax.set_ylabel("Amount (synthetic currency units)")
    revenue_ax.tick_params(axis="x", rotation=25)
    revenue_ax.grid(alpha=0.25)
    fig.suptitle("Customer analytics platform - generated results overview", fontsize=16)
    save_figure(fig, "dashboard_results_overview.png")

    print(f"Saved {len(events):,} sample events and report charts to {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    main()
