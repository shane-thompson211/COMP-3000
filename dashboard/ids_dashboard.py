import os
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# -----------------------------
# Page setup (title + layout)
# -----------------------------
st.set_page_config(
    page_title="AI IDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# Simple styling (optional)
# -----------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .stMetric { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# CONFIG (robust absolute path)
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # folder where app.py lives
FEED_CSV = os.path.join(BASE_DIR, "data", "ids_live_feed.csv")

# These are the 13 model features you said you use
FEATURE_COLS = [
    "dur", "proto", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl",
    "rate", "sload", "dload", "smean", "dmean"
]

# =============================
# Helper functions
# =============================
def safe_read_feed(path: str) -> pd.DataFrame:
    """
    Read the IDS feed safely.
    - If the file doesn't exist, return empty DF.
    - Converts timestamp to datetime where possible.
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def risk_label(prob: float) -> str:
    """
    Convert probability into a simple risk label.
    """
    if prob < 0.30:
        return "LOW"
    if prob < 0.70:
        return "MEDIUM"
    return "HIGH"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute standard classification metrics (only if you have ground-truth labels).
    """
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)

    try:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["ROC-AUC"] = None

    return metrics


def coerce_binary_labels(series: pd.Series) -> np.ndarray:
    """
    Try to convert common label formats to 0/1.
    Accepts:
    - 0/1
    - 'benign'/'attack'
    - 'normal'/'malicious'
    """
    vals = series.to_numpy()

    if vals.dtype != object:
        # numeric already
        return vals.astype(int)

    mapping = {
        "benign": 0,
        "normal": 0,
        "0": 0,
        "attack": 1,
        "malicious": 1,
        "1": 1
    }
    mapped = []
    for v in vals:
        key = str(v).strip().lower()
        mapped.append(mapping.get(key, 0))
    return np.array(mapped, dtype=int)


# =============================
# Sidebar navigation + controls
# =============================
st.sidebar.title("🛡️ AI IDS Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Live Detection", "Model Performance", "Explainability", "Cross-Dataset View", "About"]
)

st.sidebar.divider()

# Threshold slider (applied on top of malicious_prob from the feed)
threshold = st.sidebar.slider(
    "Detection Threshold (Malicious if prob ≥ threshold)",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05
)
st.sidebar.caption(f"Current threshold: **{threshold:.2f}**")

# Manual refresh (useful during demo)
if st.sidebar.button("🔄 Refresh feed"):
    st.rerun()

# Performance: only load the most recent N rows
max_rows = st.sidebar.slider("Rows to load (latest N)", 200, 50000, 5000, 200)

st.sidebar.divider()
st.sidebar.caption("Data source:")
st.sidebar.code(FEED_CSV)
st.sidebar.caption(f"File exists? {os.path.exists(FEED_CSV)}")

# =============================
# Load feed data
# =============================
df_raw = safe_read_feed(FEED_CSV)

if not df_raw.empty and len(df_raw) > max_rows:
    df_raw = df_raw.tail(max_rows).copy()

# If the feed exists, we build df_results for the dashboard
data_ready = not df_raw.empty
df_results = None

if data_ready:
    df_results = df_raw.copy()

    # Ensure malicious_prob exists (Option A should have it)
    if "malicious_prob" not in df_results.columns:
        st.error("Feed CSV is missing 'malicious_prob'. Re-check your pcap_to_ids_feed.py output row.")
        st.stop()

    # Apply threshold in dashboard (so you can demonstrate threshold tradeoffs)
    df_results["prediction"] = (df_results["malicious_prob"] >= threshold).astype(int)

    # Add risk label
    df_results["risk"] = df_results["malicious_prob"].apply(risk_label)

    # If severity exists in feed, keep it. Otherwise generate a simple one.
    if "severity" not in df_results.columns:
        def severity_from_prob(p):
            if p >= 0.95:
                return "High"
            if p >= 0.85:
                return "Medium"
            if p >= threshold:
                return "Low"
            return "None"
        df_results["severity"] = df_results["malicious_prob"].apply(severity_from_prob)

# =============================
# PAGE 1: OVERVIEW
# =============================
if page == "Overview":
    st.title("Overview")
    st.caption("High-level summary suitable for a non-technical audience.")

    if not data_ready:
        st.warning(f"No feed found at: {FEED_CSV}")
        st.info("Run `python pcap_to_ids_feed.py` first to generate the feed CSV.")
        st.stop()

    total = len(df_results)
    benign = int((df_results["prediction"] == 0).sum())
    malicious = int((df_results["prediction"] == 1).sum())
    malicious_rate = malicious / total if total > 0 else 0.0

    if malicious_rate < 0.10:
        threat_level = "LOW"
    elif malicious_rate < 0.30:
        threat_level = "MEDIUM"
    else:
        threat_level = "HIGH"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events Analysed", f"{total:,}")
    c2.metric("Benign", f"{benign:,}")
    c3.metric("Malicious", f"{malicious:,}")
    c4.metric("Threat Level", threat_level)

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Malicious Probability Trend")
        if "timestamp" in df_results.columns and df_results["timestamp"].notna().any():
            plot_df = df_results[["timestamp", "malicious_prob"]].dropna().set_index("timestamp")
            st.line_chart(plot_df)
        else:
            st.line_chart(df_results[["malicious_prob"]])

    with right:
        st.subheader("Severity Breakdown")
        sev_counts = df_results["severity"].value_counts().rename_axis("severity").to_frame("count")
        st.bar_chart(sev_counts)

    st.divider()
    st.subheader("Top Suspicious Events")
    
    top_suspicious = df_results.sort_values("malicious_prob", ascending=False).head(25)
    st.dataframe(top_suspicious, use_container_width=True, height=420)

# =============================
# PAGE 2: LIVE DETECTION
# =============================
elif page == "Live Detection":
    st.title("Live Detection")
    st.caption("Analyst view: filter alerts and inspect events.")

    if not data_ready:
        st.warning("No feed loaded yet. Run `pcap_to_ids_feed.py` first.")
        st.stop()

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        show_only = st.selectbox("Filter by prediction", ["All", "Benign (0)", "Malicious (1)"])
    with col_b:
        risk_filter = st.selectbox("Filter by risk", ["All", "LOW", "MEDIUM", "HIGH"])
    with col_c:
        min_prob = st.slider("Minimum malicious probability", 0.0, 1.0, 0.0, 0.05)

    df_view = df_results.copy()

    if show_only == "Benign (0)":
        df_view = df_view[df_view["prediction"] == 0]
    elif show_only == "Malicious (1)":
        df_view = df_view[df_view["prediction"] == 1]

    if risk_filter != "All":
        df_view = df_view[df_view["risk"] == risk_filter]

    df_view = df_view[df_view["malicious_prob"] >= min_prob]

    st.subheader("Detection Table")
    st.dataframe(df_view, use_container_width=True, height=520)

    st.divider()

    st.subheader("Inspect a Single Event")
    if len(df_view) == 0:
        st.warning("No events match your filters.")
        st.stop()

    # Use a stable selection index
    df_view = df_view.reset_index(drop=True)
    df_view["event_id"] = df_view.index
    chosen = st.selectbox("Choose event_id", df_view["event_id"].tolist())

    row = df_view[df_view["event_id"] == chosen].iloc[0]
    prob = float(row["malicious_prob"])
    pred = int(row["prediction"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction", "Malicious" if pred == 1 else "Benign")
    c2.metric("Malicious Probability", f"{prob:.3f}")
    c3.metric("Risk", row.get("risk", "N/A"))
    c4.metric("Severity", row.get("severity", "N/A"))

    st.write("Full event record:")
    st.json(row.to_dict())

# =============================
# PAGE 3: MODEL PERFORMANCE
# =============================
elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("Metrics are only available if your feed CSV contains ground-truth labels.")

    if not data_ready:
        st.warning("No feed loaded yet. Run `pcap_to_ids_feed.py` first.")
        st.stop()

    LABEL_COL = st.text_input("Label column name (ground truth)", value="label")

    if LABEL_COL not in df_results.columns:
        st.warning(
            f"No ground-truth column found named '{LABEL_COL}'. "
            "If you want metrics here, add a label column to your feed or upload labelled data."
        )
        st.stop()

    y_true = coerce_binary_labels(df_results[LABEL_COL])
    y_pred = df_results["prediction"].to_numpy()
    y_prob = df_results["malicious_prob"].to_numpy()

    metrics = compute_metrics(y_true, y_pred, y_prob)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['Precision']:.3f}")
    m3.metric("Recall", f"{metrics['Recall']:.3f}")
    m4.metric("F1", f"{metrics['F1']:.3f}")
    m5.metric("ROC-AUC", "N/A" if metrics["ROC-AUC"] is None else f"{metrics['ROC-AUC']:.3f}")

    st.divider()

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["True Benign", "True Malicious"], columns=["Pred Benign", "Pred Malicious"])
    st.dataframe(cm_df, use_container_width=True)

    st.divider()

    st.subheader("Probability Distribution")
    st.bar_chart(pd.Series(y_prob).value_counts(bins=20).sort_index())

# =============================
# PAGE 4: EXPLAINABILITY (Option A friendly)
# =============================
elif page == "Explainability":
    st.title("Explainability")
    

    if not data_ready:
        st.warning("No feed loaded yet. Run `pcap_to_ids_feed.py` first.")
        st.stop()

    # Check which feature columns exist in the feed
    present_feats = [c for c in FEATURE_COLS if c in df_results.columns]

    if len(present_feats) == 0:
        st.warning("No model feature columns found in the feed CSV.")
        st.info("Make sure pcap_to_ids_feed.py writes the 13 features into ids_live_feed.csv.")
        st.stop()

    st.subheader("Top features linked to malicious probability (correlation)")
    

    corr_df = df_results[present_feats + ["malicious_prob"]].corr(numeric_only=True)
    corr_series = corr_df["malicious_prob"].drop("malicious_prob").sort_values(key=lambda s: s.abs(), ascending=False)

    corr_table = corr_series.rename("corr_with_malicious_prob").to_frame()
    st.dataframe(corr_table, use_container_width=True, height=420)

    st.divider()

    st.subheader("Feature explorer")
    chosen_feat = st.selectbox("Choose a feature to visualise", present_feats)

    if "timestamp" in df_results.columns and df_results["timestamp"].notna().any():
        plot_df = df_results[["timestamp", chosen_feat]].dropna().set_index("timestamp")
        st.line_chart(plot_df)
    else:
        st.line_chart(df_results[[chosen_feat]])

    st.divider()
    st.subheader("Distribution by Prediction")
    st.caption("Shows how this feature differs between predicted benign vs malicious.")

    summary = df_results.groupby("prediction")[chosen_feat].describe()
    st.dataframe(summary, use_container_width=True)

# =============================
# PAGE 5: CROSS-DATASET VIEW (Option A style)
# =============================
elif page == "Cross-Dataset View":
    st.title("Cross-Dataset View")
    st.caption("Compare two feed CSVs (already predicted) to show generalisation and behaviour differences.")

    st.write(
        "Upload two CSV feeds (e.g., generated from different PCAPs or environments) "
        "and compare how the IDS behaves."
    )

    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Feed A (CSV)", type=["csv"], key="cross_a")
    with col2:
        file_b = st.file_uploader("Feed B (CSV)", type=["csv"], key="cross_b")

    if file_a and file_b:
        dfA = pd.read_csv(file_a)
        dfB = pd.read_csv(file_b)

        # Both must have malicious_prob to be comparable
        if "malicious_prob" not in dfA.columns or "malicious_prob" not in dfB.columns:
            st.error("Both feeds must include 'malicious_prob'.")
            st.stop()

        dfA["prediction"] = (dfA["malicious_prob"] >= threshold).astype(int)
        dfB["prediction"] = (dfB["malicious_prob"] >= threshold).astype(int)

        # Summary comparison
        def summarise(dfX, name):
            total = len(dfX)
            mal = int((dfX["prediction"] == 1).sum())
            return {
                "Dataset": name,
                "Events": total,
                "Malicious": mal,
                "Malicious Rate": (mal / total) if total > 0 else 0.0,
                "Mean Prob": float(dfX["malicious_prob"].mean()),
                "P95 Prob": float(dfX["malicious_prob"].quantile(0.95)),
            }

        comp = pd.DataFrame([summarise(dfA, "A"), summarise(dfB, "B")])
        st.subheader("Comparison Summary")
        st.dataframe(comp, use_container_width=True)

        st.divider()
        st.subheader("Probability distributions")
        st.write("Feed A")
        st.bar_chart(dfA["malicious_prob"].value_counts(bins=20).sort_index())
        st.write("Feed B")
        st.bar_chart(dfB["malicious_prob"].value_counts(bins=20).sort_index())

# =============================
# PAGE 6: ABOUT
# =============================
elif page == "About":
    st.title("About")
    st.write("This dashboard visualises an AI-based Intrusion Detection System (IDS) event feed.")

    st.markdown(
        f"""
        **Build timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        **Mode:** Option A (PCAP → model in script → CSV feed → Streamlit dashboard)

        **Key features:**
        - Reads `ids_live_feed.csv` generated by your Scapy pipeline
        - Threshold-based detection (adjustable live in sidebar)
        - Executive overview + analyst triage
        - Optional performance metrics if you have labels
        - Explainability via feature behaviour/correlation (no model loading here)
        """
    )

    st.info(
        "For your showcase/viva: demo the full pipeline: PCAP → flow features → ML prediction → SOC-style dashboard triage."
    )