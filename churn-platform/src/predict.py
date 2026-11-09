import json
from functools import lru_cache

import joblib
import pandas as pd

from .config import METRICS_PATH, MODEL_PATH
from .features import FEATURES


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run:  python -m src.train")
    return joblib.load(MODEL_PATH)


def score(df: pd.DataFrame):
    """Return churn probability for each row."""
    return get_model().predict_proba(df[FEATURES])[:, 1]


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
