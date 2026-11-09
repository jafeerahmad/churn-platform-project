"""First-run setup so `streamlit run app/dashboard.py` works with no other commands
(also what makes one-click Streamlit Cloud deployment possible)."""
import pandas as pd

from .config import MODEL_PATH
from .db import engine


def has_rows(table: str) -> bool:
    try:
        n = pd.read_sql(f"SELECT COUNT(*) AS n FROM {table}", engine)["n"].iloc[0]
        return int(n) > 0
    except Exception:
        return False


def ensure_ready():
    from . import simulator, train
    if not MODEL_PATH.exists() or not has_rows("customers"):
        train.main()
    if not has_rows("events"):
        simulator.backfill(simulator.customer_ids(), days=30)
