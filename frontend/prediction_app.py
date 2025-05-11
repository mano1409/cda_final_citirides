import os
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime as dt

# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT CONFIG
st.set_page_config(page_title="Citi Bike Dashboard", layout="wide")

# === Absolute paths to your resources ===
PARQUET_PATH = Path("/Users/manu/Desktop/cda_final/data/processed/cleaned_citibike/citibike_2023_top3.parquet")
MODEL_FILE   = Path("/Users/manu/Desktop/cda_final/notebook/lgbm_28lag.pkl")
MAX_LAG      = 28

# MLflow (DagsHub) Settings
USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "manogna145")
PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "2207a6e5841298a92645f15b687a350ec0044c23")
os.environ["MLFLOW_TRACKING_USERNAME"] = USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = PASSWORD
mlflow.set_tracking_uri("https://dagshub.com/manogna145/cda_finalproject.mlflow")
EXPERIMENT_NAME = "CitiBike_TripDuration_Models"

# ──────────────────────────────────────────────────────────────────────────────

def load_model():
    if not MODEL_FILE.exists():
        st.error(f"❌ Could not find model file:\n  {MODEL_FILE}")
        st.stop()
    return joblib.load(MODEL_FILE)

@st.cache_data(ttl=300)
def fetch_metrics():
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        st.error(f"Experiment '{EXPERIMENT_NAME}' not found.")
        st.stop()
    runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time ASC"])
    records = []
    for r in runs:
        history = client.get_metric_history(r.info.run_id, "mae")
        if history:
            records.append({
                "time":   r.info.start_time,
                "run_id": r.info.run_id,
                "mae":    history[-1].value
            })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.set_index("time").sort_index()

@st.cache_data
def build_lag_features(as_of: pd.Timestamp) -> pd.DataFrame:
    if not PARQUET_PATH.exists():
        st.error(f"❌ Could not find data file:\n  {PARQUET_PATH}")
        st.stop()
    df = pd.read_parquet(PARQUET_PATH)
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["datetime"] = df["started_at"].dt.floor("H")
    df = df[df["datetime"] <= as_of]
    agg = df.groupby("datetime").size().reset_index(name="count").sort_values("datetime")
    counts = agg["count"].values
    if len(counts) < MAX_LAG:
        st.error(f"Need {MAX_LAG} historical hours but only found {len(counts)} up to {as_of}.")
        st.stop()
    last_counts = counts[-MAX_LAG:]
    row = {f"lag_{i}": last_counts[-i] for i in range(1, MAX_LAG+1)}
    return pd.DataFrame([row])

# ──────────────────────────────────────────────────────────────────────────────
# UI Tabs
tabs = st.tabs(["🚀 Predict Next Hour", "📈 Monitoring"])

# ---- Prediction Tab ----
with tabs[0]:
    st.header("🚲 Predict Next‐Hour Citi Bike Trips")
    model = load_model()

    as_of_date = st.sidebar.date_input("As-of Date", pd.Timestamp.now().date())
    as_of_time = st.sidebar.time_input("As-of Time", pd.Timestamp.now().time())
    as_of = pd.to_datetime(dt.combine(as_of_date, as_of_time))

    if st.sidebar.button("Run Prediction"):
        lags_df = build_lag_features(as_of)
        pred = model.predict(lags_df)[0]
        st.metric("📊 Predicted Trips Next Hour", f"{pred:.0f}")
        st.write("**Using data up to:**", as_of)

# ---- Monitoring Tab ----
with tabs[1]:
    st.header("📊 Model MAE Over Time")
    metrics_df = fetch_metrics()
    if metrics_df.empty:
        st.info("No MAE metrics logged yet.")
    else:
        st.line_chart(metrics_df["mae"])
        st.dataframe(metrics_df)

st.markdown("---")
st.write("Built with MLflow, DagsHub & Streamlit.")
