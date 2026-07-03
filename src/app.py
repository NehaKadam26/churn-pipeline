"""
Streamlit dashboard for the Automated Churn Prediction Pipeline.

Shows:
- Pipeline status (last run time, batches released so far)
- Trend chart: F1/precision/recall across batches 1-5 (the real "more data -> better model" story)
- Automation verification: any run(s) that retrained on the SAME data (e.g. a GitHub Actions
  run that found no new batches) are shown separately as proof the pipeline works end-to-end
  on GitHub, not folded into the trend line as if they were new data points.

Run locally with:
    streamlit run src/app.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "pipeline.db"

st.set_page_config(
    page_title="Churn Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=60)
def load_runs() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT run_id, run_at, n_customers_trained_on, batches_included,
               precision, recall, f1, accuracy, tn, fp, fn, tp
        FROM training_runs
        ORDER BY run_id
        """,
        conn,
    )
    conn.close()
    df["run_at"] = pd.to_datetime(df["run_at"])
    return df


def split_trend_and_repeats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The trend line should only reflect runs that trained on genuinely NEW data
    (i.e. the first run to reach a given batches_included count). Any later run
    that retrained on an already-seen batch count (e.g. a manual GitHub Actions
    trigger after all batches were released) is a reproducibility/automation
    check, not a new data point, and is shown separately.
    """
    first_at_each_batch_count = df.drop_duplicates(subset="batches_included", keep="first")
    repeats = df[~df["run_id"].isin(first_at_each_batch_count["run_id"])]
    return first_at_each_batch_count, repeats


df = load_runs()

if df.empty:
    st.warning("No training runs found yet in `data/pipeline.db`. Run `train.py` at least once.")
    st.stop()

trend_df, repeat_df = split_trend_and_repeats(df)

# ---------- Header / status strip ----------
st.title("📊 Automated Churn Prediction Pipeline")
st.caption("Live status pulled from `training_runs` — refreshes automatically after each pipeline run.")

latest = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total runs logged", len(df))
col2.metric("Customers trained on (latest)", f"{latest['n_customers_trained_on']:,}")
col3.metric("Latest F1", f"{latest['f1']:.3f}")
col4.metric("Last run", latest["run_at"].strftime("%b %d, %Y %H:%M"))

st.divider()

# ---------- Trend chart: real data-growth story ----------
st.subheader("Model performance as training data grows")
st.caption(
    "Each point is the first run to train on that many batches — this is the genuine "
    "'more data → better model' trend, not repeated runs on unchanged data."
)

fig = go.Figure()
for metric, color in [
    ("f1", "#2563eb"),
    ("precision", "#16a34a"),
    ("recall", "#dc2626"),
    ("accuracy", "#7c3aed"),
]:
    fig.add_trace(
        go.Scatter(
            x=trend_df["batches_included"],
            y=trend_df[metric],
            mode="lines+markers",
            name=metric.capitalize(),
            line=dict(color=color, width=2),
            marker=dict(size=8),
        )
    )

fig.update_layout(
    xaxis_title="Batches included (cumulative)",
    yaxis_title="Score",
    yaxis=dict(range=[0, 1]),
    hovermode="x unified",
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)

if len(trend_df) >= 2:
    f1_start = trend_df.iloc[0]["f1"]
    f1_end = trend_df.iloc[-1]["f1"]
    pct_change = (f1_end - f1_start) / f1_start * 100
    st.success(
        f"F1 improved from **{f1_start:.3f}** to **{f1_end:.3f}** "
        f"(~{pct_change:.1f}% relative improvement) as training data grew from "
        f"{trend_df.iloc[0]['n_customers_trained_on']:,} to "
        f"{trend_df.iloc[-1]['n_customers_trained_on']:,} customers."
    )

st.divider()

# ---------- Automation verification (separate from the trend) ----------
st.subheader("Automation verification")

if repeat_df.empty:
    st.info(
        "No repeat runs yet. Once the scheduled or manually-triggered GitHub Actions run "
        "retrains on an already-seen batch count, it will show up here as a reproducibility check."
    )
else:
    for _, row in repeat_df.iterrows():
        matching_trend_row = trend_df[trend_df["batches_included"] == row["batches_included"]].iloc[0]
        f1_match = abs(row["f1"] - matching_trend_row["f1"]) < 1e-9
        status = "✅ identical results — deterministic, reproducible" if f1_match else "⚠️ results differ from original run"
        st.write(
            f"**Run #{row['run_id']}** — {row['run_at'].strftime('%b %d, %Y %H:%M')} — "
            f"retrained on {row['n_customers_trained_on']:,} customers ({row['batches_included']} batches): {status}"
        )
    st.caption(
        "These runs confirm the pipeline executes correctly end-to-end on GitHub Actions "
        "(ingest → train → commit) — not new data, but proof the automation itself works."
    )

st.divider()

# ---------- Raw run history ----------
with st.expander("Full run history (raw table)"):
    st.dataframe(
        df[["run_id", "run_at", "n_customers_trained_on", "batches_included", "precision", "recall", "f1", "accuracy"]],
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Note: this pipeline uses a fixed, pre-assigned batch simulation (5 batches, seeded split) "
    "to stand in for incrementally arriving customer data — it is not a live customer feed."
)
