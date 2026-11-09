# Generated result charts

The PNG files in this folder are generated analytical charts from the bundled
synthetic customer data, held-out model predictions, and fixed-seed sample event
export. They are not captured screenshots of a live Streamlit browser session.

Rebuild all report assets from the repository root with:

```bash
python -m src.train
python scripts/prepare_demo_assets.py
```

Files:

- `churn_distribution.png` - distribution of the two target classes.
- `model_comparison.png` - holdout classification metrics for both models.
- `roc_curves.png` - ROC curves for the 1,400-row holdout test set.
- `confusion_matrices.png` - confusion matrices at a 0.50 probability threshold.
- `top_churn_drivers.png` - largest absolute coefficients for the selected model.
- `event_activity.png` - event type counts and daily simulated order value.
- `dashboard_results_overview.png` - dashboard-style summary of model and sample data.
