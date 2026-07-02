"""
src/train.py

Retrains a churn classifier on all customer data ingested so far, evaluates
against the FIXED holdout (carved once in ingest.py's first run, never
touched by ingestion — same holdout every time, so results are directly
comparable across retraining runs), and appends the result to a
training_runs history table. This history is what "performance over time"
in the dashboard/resume claim is actually built from.

Run from the project root, after ingest.py:
    python src/train.py
"""

import json
import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DB_PATH = Path("data/pipeline.db")
MODEL_PATH = Path("models/model.pkl")

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
TARGET = "Churn"


def load_json_rows(conn, table, where="") -> pd.DataFrame:
    rows = conn.execute(f"SELECT data FROM {table} {where}").fetchall()
    return pd.DataFrame([json.loads(r[0]) for r in rows])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges arrives as a string with blanks for brand-new customers
    # (tenure=0). Coerce to numeric; a blank for a tenure=0 customer correctly
    # means zero charges so far, not a missing-data problem to hide.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    if TARGET in df.columns and df[TARGET].dtype == object:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def ensure_runs_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT NOT NULL,
            n_customers_trained_on INTEGER NOT NULL,
            batches_included INTEGER NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            f1 REAL NOT NULL,
            accuracy REAL NOT NULL,
            tn INTEGER, fp INTEGER, fn INTEGER, tp INTEGER
        )
    """)
    conn.commit()


def main():
    if not DB_PATH.exists():
        sys.exit(f"ERROR: {DB_PATH} not found. Run src/ingest.py first.")

    conn = sqlite3.connect(DB_PATH)
    ensure_runs_table(conn)

    train_df = load_json_rows(conn, "customers", "WHERE ingested_at IS NOT NULL")
    holdout_df = load_json_rows(conn, "holdout")

    if train_df.empty:
        sys.exit("ERROR: no ingested customers found. Run src/ingest.py first (release at least one batch).")
    if holdout_df.empty:
        sys.exit("ERROR: no holdout data found — ingest.py's first-time setup may not have run correctly.")

    train_df = clean(train_df)
    holdout_df = clean(holdout_df)

    X_train, y_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df[TARGET].astype(int)
    X_holdout, y_holdout = holdout_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], holdout_df[TARGET].astype(int)

    print(f"Training on {len(X_train)} customers (all batches ingested so far)")
    print(f"Evaluating on fixed holdout: {len(X_holdout)} customers (never used for training)")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_holdout)

    precision = precision_score(y_holdout, y_pred, zero_division=0)
    recall = recall_score(y_holdout, y_pred, zero_division=0)
    f1 = f1_score(y_holdout, y_pred, zero_division=0)
    accuracy = accuracy_score(y_holdout, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_holdout, y_pred, labels=[0, 1]).ravel()

    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  Accuracy: {accuracy:.3f}")
    print(f"Confusion matrix: tn={tn} fp={fp} fn={fn} tp={tp}")

    batches_included = conn.execute(
        "SELECT COUNT(DISTINCT batch_id) FROM customers WHERE ingested_at IS NOT NULL"
    ).fetchone()[0]

    conn.execute(
        """INSERT INTO training_runs
           (run_at, n_customers_trained_on, batches_included, precision, recall, f1, accuracy, tn, fp, fn, tp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (pd.Timestamp.now().isoformat(), len(X_train), batches_included,
         precision, recall, f1, accuracy, int(tn), int(fp), int(fn), int(tp)),
    )
    conn.commit()
    conn.close()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES},
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")

    print()
    print("NOTE: the numbers above are real, from this actual run against the fixed holdout.")
    print("Query the training_runs table for the full run history before writing any")
    print("resume/README claim about performance over time.")


if __name__ == "__main__":
    main()
