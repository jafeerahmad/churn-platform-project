"""Simulate customer activity events for the demo platform.

Examples:
  python -m src.simulator --backfill 30 --once
  python -m src.simulator --interval 3
"""
import argparse
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import utcnow
from .db import engine

TYPES = ["login", "order", "support_ticket", "cancellation"]
PROBS = [0.55, 0.27, 0.15, 0.03]
rng = np.random.default_rng()


def customer_ids():
    ids = pd.read_sql("SELECT customerID FROM customers", engine)["customerID"].to_numpy()
    if len(ids) == 0:
        raise RuntimeError("No customers found. Run `python -m src.train` first.")
    return ids


def make_events(ids, n, start, end, random_generator=None) -> pd.DataFrame:
    """Create an event batch; pass a seeded generator for reproducible demo exports."""
    if len(ids) == 0:
        raise ValueError("At least one customer ID is required.")
    if n < 0:
        raise ValueError("Event count cannot be negative.")
    if end < start:
        raise ValueError("Event end must not be earlier than event start.")
    generator = random_generator if random_generator is not None else rng
    seconds = max((end - start).total_seconds(), 0)
    offsets = generator.uniform(0, seconds, n) if seconds else np.zeros(n)
    event_types = generator.choice(TYPES, n, p=PROBS)
    amount = np.where(event_types == "order", generator.gamma(2.5, 25, n).round(2), 0.0)
    return pd.DataFrame({
        "customer_id": generator.choice(ids, n),
        "event_type": event_types,
        "amount": amount,
        "event_time": [start + timedelta(seconds=float(value)) for value in offsets],
    })


def save(df: pd.DataFrame):
    if not df.empty:
        df.to_sql("events", engine, if_exists="append", index=False)


def backfill(ids, days: int = 30, per_day: int = 1500):
    if days < 1 or per_day < 1:
        raise ValueError("days and per_day must both be positive integers.")
    now = utcnow()
    for day_number in range(days, 0, -1):
        start = now - timedelta(days=day_number)
        save(make_events(ids, per_day, start, start + timedelta(days=1)))


def tick(ids) -> int:
    """Write a small batch over the preceding few seconds, not in the future."""
    n = int(rng.integers(3, 11))
    now = utcnow()
    save(make_events(ids, n, now - timedelta(seconds=3), now))
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=3)
    parser.add_argument("--backfill", type=int, default=0, help="days of history to seed")
    parser.add_argument("--per-day", type=int, default=1500)
    parser.add_argument("--once", action="store_true", help="exit after backfill")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    ids = customer_ids()
    if args.backfill:
        backfill(ids, args.backfill, args.per_day)
        print(f"Backfilled {args.backfill} days of simulated events")
    if args.once:
        return
    print("Streaming simulated events. Press Ctrl+C to stop.")
    while True:
        count = tick(ids)
        print(f"{utcnow()} | +{count} events")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
