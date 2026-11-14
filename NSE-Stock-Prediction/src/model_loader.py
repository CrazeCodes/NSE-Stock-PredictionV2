"""Cached loading of models, metrics and data for the Streamlit app."""
import json

import joblib
import pandas as pd
import streamlit as st

from .config import METRICS_FILE, MODEL_NAMES, MODELS_DIR, PRED_DIR, STOCKS
from .data_loader import has_snapshot, load_snapshot
from .model_training import train_all


def artifacts_ready() -> bool:
    if not METRICS_FILE.exists():
        return False
    try:
        stocks = json.loads(METRICS_FILE.read_text())["stocks"]
    except (json.JSONDecodeError, KeyError):
        return False
    return bool(stocks) and all(
        (MODELS_DIR / f"{s}_{n}.joblib").exists() and (PRED_DIR / f"{s}_test.csv").exists()
        for s in stocks for n in MODEL_NAMES)


def ensure_artifacts() -> None:
    """If models are missing (fresh deploy), train them once from committed snapshots."""
    if artifacts_ready():
        return
    use_live = not all(has_snapshot(s) for s in STOCKS)
    with st.spinner("First launch: training models (about a minute)..."):
        train_all(use_live=use_live, log=lambda *_: None)
    st.cache_data.clear()
    st.cache_resource.clear()


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads(METRICS_FILE.read_text())


def available_stocks() -> list:
    return [s for s in STOCKS if s in load_metrics()["stocks"]]


@st.cache_resource(show_spinner=False)
def load_model(stock: str, name: str):
    return joblib.load(MODELS_DIR / f"{stock}_{name}.joblib")


@st.cache_resource(show_spinner=False)
def load_scaler(stock: str):
    return joblib.load(MODELS_DIR / f"{stock}_scaler.joblib")


@st.cache_data(show_spinner=False)
def load_test_predictions(stock: str) -> pd.DataFrame:
    return pd.read_csv(PRED_DIR / f"{stock}_test.csv", parse_dates=["Date"])


@st.cache_data(show_spinner=False)
def load_snapshot_df(stock: str) -> pd.DataFrame:
    return load_snapshot(stock)
