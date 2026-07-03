"""
Streamlit dashboard for the Automated Churn Prediction Pipeline.

Shows:
- Pipeline status (last run time, batches released so far)
- Trend chart: F1/precision/recall across batches 1-5 (the real "more data -> better model" story)
- Confusion matrix for the latest run
- Full run history table
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

# ──────────────────────────────────────────────────────────────────────────
# THEME (forest green — change these to re-skin the whole dashboard)
# ──────────────────────────────────────────────────────────────────────────
PIPELINE_NAME = "ChurnPipe"
SUBTITLE = "Automated retraining · Telco churn model"
PRIMARY = "#0E7C4A"
PRIMARY_DARK = "#0B5C39"
SIDEBAR_BG = "#0B4A30"
SIDEBAR_BG_ACTIVE = "#12613D"
ACCENT = "#2563EB"
BG_CARD = "#FFFFFF"
BG_PAGE = "#F5F5FA"
TEXT_MUTED = "#6B7280"
RISK_RED = "#DC2626"
GOOD_GREEN = "#059669"

st.set_page_config(
    page_title="Churn Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {BG_PAGE}; }}
        #MainMenu, footer, header {{ visibility: hidden; }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {{
            background-color: {SIDEBAR_BG};
        }}
        section[data-testid="stSidebar"] * {{
            color: #E6F4EC;
        }}
        .sb-logo {{
            font-size: 24px;
            font-weight: 800;
            color: white;
            margin-top: 4px;
        }}
        .sb-logo-sub {{
            font-size: 11px;
            letter-spacing: 0.12em;
            color: #9FD8B8;
            margin-bottom: 22px;
        }}
        .sb-section-label {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.1em;
            color: #7FBF9C;
            text-transform: uppercase;
            margin: 18px 0 8px 2px;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            background: transparent;
            border-radius: 8px;
            padding: 8px 10px;
            margin-bottom: 2px;
            width: 100%;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background: rgba(255,255,255,0.06);
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] input:checked + div {{
            font-weight: 700;
        }}
        .sb-stat-box {{
            background: rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 10px;
        }}
        .sb-stat-value {{ font-size: 20px; font-weight: 700; color: white; }}
        .sb-stat-label {{ font-size: 11px; color: #9FD8B8; }}
        .sb-live-dot {{
            display: inline-block;
            width: 7px; height: 7px;
            border-radius: 50%;
            background: #4ADE80;
            margin-right: 6px;
        }}

        /* ---- Top bar ---- */
        .topbar {{
            background: linear-gradient(90deg, {PRIMARY_DARK} 0%, {PRIMARY} 100%);
            padding: 16px 26px;
            border-radius: 10px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .topbar-title {{ font-size: 20px; font-weight: 700; }}
        .topbar-sub {{ font-size: 13px; opacity: 0.85; }}
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 8px;
        }}
        .badge-live {{ background: rgba(255,255,255,0.18); color: white; }}
        .badge-good {{ background: #ECFDF5; color: {GOOD_GREEN}; }}
        .badge-warn {{ background: #FEF2F2; color: {RISK_RED}; }}
        .page-subtitle {{
            font-size: 20px;
            font-weight: 700;
            color: #111827;
            margin: 4px 0 18px 2px;
        }}

        /* ---- KPI cards ---- */
        .metric-card {{
            background: {BG_CARD};
            border-left: 4px solid {PRIMARY};
            border-radius: 8px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            height: 118px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }}
        .metric-label {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: {TEXT_MUTED};
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 26px;
            font-weight: 700;
            color: #111827;
            margin-top: 10px;
            white-space: nowrap;
        }}
        /* Smaller, single-line variant for longer values like timestamps */
        .metric-value-date {{
            font-size: 17px;
            font-weight: 700;
            color: #111827;
            margin-top: 14px;
            white-space: nowrap;
        }}

        .section-label {{
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            margin: 22px 0 10px 0;
        }}

        .panel-title {{ font-size: 15px; font-weight: 700; color: #111827; }}
        .panel-desc {{ font-size: 12px; color: {TEXT_MUTED}; margin-bottom: 10px; }}

        /* Style Streamlit's native bordered container to look like our "panel" card.
           Scoped to the main content area only — NOT the sidebar, which uses the
           same data-testid internally for its own layout wrapper. */
        section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: {BG_CARD};
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            border: none !important;
        }}
        section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            border: none !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: transparent !important;
            box-shadow: none !important;
            border-radius: 0 !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────
# DATA LOADING (unchanged from your original logic)
# ──────────────────────────────────────────────────────────────────────────
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


if not DB_PATH.exists():
    st.error(f"No database found at `{DB_PATH}`. Run `train.py` at least once to create it.")
    st.stop()

df = load_runs()

if df.empty:
    st.warning("No training runs found yet in `data/pipeline.db`. Run `train.py` at least once.")
    st.stop()

trend_df, repeat_df = split_trend_and_repeats(df)
latest = df.iloc[-1]

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="sb-logo">📊 {PIPELINE_NAME}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-logo-sub">AUTOMATED ML PIPELINE</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">Analytics</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Overview", "Run History", "Automation Check"],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sb-section-label"><span class="sb-live-dot"></span>Live Data</div>',
        unsafe_allow_html=True,
    )
    sb1, sb2 = st.columns(2)
    with sb1:
        st.markdown(
            f"""<div class="sb-stat-box">
                    <div class="sb-stat-value">{len(df)}</div>
                    <div class="sb-stat-label">Total runs</div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div class="sb-stat-box">
                    <div class="sb-stat-value">{latest['f1']:.3f}</div>
                    <div class="sb-stat-label">Latest F1</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with sb2:
        st.markdown(
            f"""<div class="sb-stat-box">
                    <div class="sb-stat-value">{latest['n_customers_trained_on']:,}</div>
                    <div class="sb-stat-label">Customers</div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div class="sb-stat-box">
                    <div class="sb-stat-value">{int(latest['batches_included'])}</div>
                    <div class="sb-stat-label">Batches</div>
                </div>""",
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────────────────
# TOP BAR (shown on every page)
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="topbar">
        <div>
            <span class="topbar-title">{PIPELINE_NAME}</span>
            <span class="topbar-sub"> &nbsp;|&nbsp; {page} · {SUBTITLE}</span>
        </div>
        <div>
            <span class="badge badge-live">Run #{int(latest['run_id'])}</span>
            <span class="badge badge-good">Live</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(f'<div class="page-subtitle">{page} Dashboard</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# KEY METRICS (shown on every page, like the reference)
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
metric_cells = [
    (c1, "Total Runs Logged", f"{len(df)}", False),
    (c2, "Customers Trained On", f"{latest['n_customers_trained_on']:,}", False),
    (c3, "Latest F1", f"{latest['f1']:.3f}", False),
    (c4, "Batches Included", f"{int(latest['batches_included'])}", False),
    (c5, "Last Run", latest["run_at"].strftime("%b %d, %Y · %H:%M"), True),
]
for col, label, value, is_long_value in metric_cells:
    with col:
        value_class = "metric-value-date" if is_long_value else "metric-value"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="{value_class}">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW — trend chart + confusion matrix side by side
# ──────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="panel-title">Model performance as training data grows</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-desc">Each point is the first run to train on that many '
            'batches — the genuine "more data → better model" trend, not repeated runs on '
            'unchanged data.</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        for metric, color in [
            ("f1", PRIMARY),
            ("precision", GOOD_GREEN),
            ("recall", RISK_RED),
            ("accuracy", ACCENT),
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
            plot_bgcolor=BG_CARD,
            paper_bgcolor=BG_CARD,
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

# ──────────────────────────────────────────────────────────────────────────
# PAGE: RUN HISTORY — full table
# ──────────────────────────────────────────────────────────────────────────
elif page == "Run History":
    st.markdown('<div class="section-label">Pipeline History</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="panel-title">Full Run History</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-desc">Raw `training_runs` table, most recent first</div>', unsafe_allow_html=True)
        display_df = df.sort_values("run_id", ascending=False)[
            ["run_id", "run_at", "n_customers_trained_on", "batches_included", "precision", "recall", "f1", "accuracy"]
        ].rename(
            columns={
                "run_id": "Run",
                "run_at": "Run At",
                "n_customers_trained_on": "Customers",
                "batches_included": "Batches",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "accuracy": "Accuracy",
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE: AUTOMATION CHECK — repeat-run verification
# ──────────────────────────────────────────────────────────────────────────
else:
    st.markdown('<div class="section-label">Pipeline History</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="panel-title">Automation Verification</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-desc">Runs that retrained on an already-seen batch count — '
            'proof the pipeline runs end-to-end on GitHub Actions, not new data.</div>',
            unsafe_allow_html=True,
        )

        if repeat_df.empty:
            st.markdown(
                f"""
                <div style="font-size:13px;color:{TEXT_MUTED};">
                    No repeat runs yet. Once a scheduled or manually-triggered GitHub Actions run
                    retrains on an already-seen batch count, it will show up here as a
                    reproducibility check.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            cols = st.columns(3)
            for i, (_, row) in enumerate(repeat_df.iterrows()):
                matching_trend_row = trend_df[trend_df["batches_included"] == row["batches_included"]].iloc[0]
                f1_match = abs(row["f1"] - matching_trend_row["f1"]) < 1e-9
                status_text = "Identical results — deterministic ✅" if f1_match else "Results differ ⚠️"
                status_color = GOOD_GREEN if f1_match else RISK_RED
                with cols[i % 3]:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="margin-bottom:14px;border-left-color:{status_color};">
                            <div class="metric-label">Run #{int(row['run_id'])} · {row['run_at'].strftime('%b %d, %H:%M')}</div>
                            <div class="metric-value" style="font-size:15px;white-space:normal;">{status_text}</div>
                            <div style="font-size:12px;color:{TEXT_MUTED};margin-top:2px;">
                                {int(row['n_customers_trained_on']):,} customers · {int(row['batches_included'])} batches
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

st.markdown(
    f"""
    <div style="text-align:center;color:{TEXT_MUTED};font-size:12px;margin-top:26px;">
        Note: this pipeline uses a fixed, pre-assigned batch simulation (5 batches, seeded split)
        to stand in for incrementally arriving customer data — it is not a live customer feed.
    </div>
    """,
    unsafe_allow_html=True,
)