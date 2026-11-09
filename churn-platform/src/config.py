import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "telco_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

# SQLite by default (zero setup). For MySQL set:
# DATABASE_URL=mysql+pymysql://user:password@host:3306/churn
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'platform.db'}")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
