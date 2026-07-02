"""
src/ingest.py

Simulates incremental customer data arriving over time. On the very first
run: downloads the Telco churn dataset (cached locally after that), carves
out a FIXED holdout set (stratified by churn, 15% of data) that is never
touched by ingestion again — this is what makes precision/recall/F1
comparable across every retraining run, since they're always evaluated
against the exact same unseen data. The remaining rows are split into 5
batches with a fixed random seed (reproducible, not arbitrary).

Each run releases the next unreleased batch into the `customers` table.
Idempotent — safe to re-run. If all batches are already released, it exits
cleanly (exit code 0, not an error) with a message saying so.

Run from the project root:
    python src/ingest.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_CACHE_PATH = Path("data/raw/telco_churn_full.csv")
DB_PATH = Path("data/pipeline.db")

N_BATCHES = 5
HOLDOUT_FRACTION = 0.15
RANDOM_SEED = 42


def download_dataset() -> pd.DataFrame:
    RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_CACHE_PATH.exists():
        print(f"Using cached dataset at {RAW_CACHE_PATH}")
        return pd.read_csv(RAW_CACHE_PATH)

    print(f"Downloading dataset from {DATA_URL} ...")
    resp = requests.get(DATA_URL, timeout=30)
    resp.raise_for_status()
    RAW_CACHE_PATH.write_bytes(resp.content)
    print(f"Saved {len(resp.content)} bytes to {RAW_CACHE_PATH}")
    return pd.read_csv(RAW_CACHE_PATH)


def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customerID TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            batch_id INTEGER NOT NULL,
            ingested_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS holdout (
            customerID TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()


def get_state(conn, key, default=None):
    row = conn.execute("SELECT value FROM pipeline_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_state(conn, key, value):
    conn.execute(
        "INSERT INTO pipeline_state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )
    conn.commit()


def first_time_setup(conn, df: pd.DataFrame):
    """Carves the fixed holdout and assigns batch IDs. Runs exactly once —
    every subsequent run skips this entirely, so the holdout and batch
    assignments never change once set."""
    print("First run detected — carving out fixed holdout and assigning batches...")

    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    train_pool, holdout = train_test_split(
        df, test_size=HOLDOUT_FRACTION, stratify=df["Churn"], random_state=RANDOM_SEED
    )

    train_pool = train_pool.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    train_pool["batch_id"] = pd.qcut(train_pool.index, N_BATCHES, labels=False)

    for _, row in train_pool.iterrows():
        conn.execute(
            "INSERT OR IGNORE INTO customers (customerID, data, batch_id, ingested_at) VALUES (?, ?, ?, NULL)",
            (row["customerID"], row.to_json(), int(row["batch_id"])),
        )
    for _, row in holdout.iterrows():
        conn.execute(
            "INSERT OR IGNORE INTO holdout (customerID, data) VALUES (?, ?)",
            (row["customerID"], row.to_json()),
        )

    set_state(conn, "next_batch_to_release", 0)
    set_state(conn, "total_batches", N_BATCHES)
    conn.commit()

    print(f"Holdout: {len(holdout)} rows (fixed, never used for training)")
    print(f"Training pool: {len(train_pool)} rows split into {N_BATCHES} batches")


def release_next_batch(conn) -> bool:
    next_batch = int(get_state(conn, "next_batch_to_release", -1))
    total_batches = int(get_state(conn, "total_batches", N_BATCHES))

    if next_batch >= total_batches:
        print(f"All {total_batches} batches already released. Nothing new to ingest.")
        return False

    now = pd.Timestamp.now().isoformat()
    cur = conn.execute(
        "UPDATE customers SET ingested_at = ? WHERE batch_id = ? AND ingested_at IS NULL",
        (now, next_batch),
    )
    conn.commit()
    set_state(conn, "next_batch_to_release", next_batch + 1)

    print(f"Released batch {next_batch} ({cur.rowcount} customers) at {now}")
    return True


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    if get_state(conn, "next_batch_to_release") is None:
        df = download_dataset()
        first_time_setup(conn, df)

    released = release_next_batch(conn)

    total_ingested = conn.execute(
        "SELECT COUNT(*) FROM customers WHERE ingested_at IS NOT NULL"
    ).fetchone()[0]
    print(f"Total customers ingested so far (available for training): {total_ingested}")

    conn.close()
    sys.exit(0)  # not-released is a normal, expected end state — not an error


if __name__ == "__main__":
    main()
