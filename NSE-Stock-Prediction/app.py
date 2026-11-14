"""Page 1: Overview Dashboard (entry point: streamlit run app.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import MODEL_LABELS, STOCK_LABELS
from src.model_loader import available_stocks, ensure_artifacts, load_metrics, load_snapshot_df
from src.ui import PALETTE, disclaimer, header, rupee, setup_page, show

setup_page("Overview")
header("NSE Predict: Stock Market Analytics & Forecasting",
       "Machine-learning price models for five large-cap NSE stocks")

try:
    ensure_artifacts()
    metrics = load_metrics()
    stocks = available_stocks()
except Exception as exc:
    st.error("Could not load model artifacts or market data. If this is a fresh deployment, "
             "make sure the `data/` snapshots and `models/` folder are in the repository.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

st.markdown(
    "This project began as a Jupyter notebook on Kaggle and is now an interactive app. "
    "It downloads two years of daily prices for five NSE stocks, engineers technical "
    "features (moving averages, returns, volatility, volume ratios, lagged closes), trains a "
    "**Random Forest** and a **Linear Regression** model per stock, and keeps the one with "
    "the lower test RMSE."
)
disclaimer()

stock = st.selectbox("Stock", stocks, format_func=lambda s: f"{s} · {STOCK_LABELS[s]}")
info = metrics["stocks"][stock]
df = load_snapshot_df(stock)
best = info["best_model"]
best_test = info["models"][best]["test"]

last, prev = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Latest close", rupee(last), f"{(last / prev - 1) * 100:+.2f}% (1 day)")
c2.metric("Data as of", df["Date"].iloc[-1].strftime("%d %b %Y"))
c3.metric("Selected model", MODEL_LABELS[best])
c4.metric("Test R²", f"{best_test['r2']:.3f}")
c5.metric("Test RMSE", rupee(best_test["rmse"]))
st.caption("Model metrics are measured on the held-out final 20% of the data using the "
           "original notebook methodology. See *Model Evaluation* for details and limitations.")

fig = go.Figure(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close",
                           line=dict(color=PALETTE["actual"], width=2),
                           hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>"))
fig.update_layout(title=f"{STOCK_LABELS[stock]} closing price", yaxis_title="Price (₹)")
fig.update_xaxes(rangeselector=dict(buttons=[
    dict(count=1, label="1M", step="month", stepmode="backward"),
    dict(count=3, label="3M", step="month", stepmode="backward"),
    dict(count=6, label="6M", step="month", stepmode="backward"),
    dict(step="all", label="All")]), rangeslider=dict(visible=False))
show(fig)

left, right = st.columns(2)
with left:
    st.subheader("Dataset")
    st.dataframe(pd.DataFrame({
        "Item": ["Source", "Period", "Daily rows (raw)", "Rows after feature engineering",
                 "Features used", "Train / test rows", "Train/test split date", "Snapshot taken"],
        "Value": ["Yahoo Finance via yfinance", f"{info['data_start']} to {info['data_end']}",
                  info["n_raw"], info["n_model_rows"], len(info["feature_cols"]),
                  f"{info['n_train']} / {info['n_test']}", info["split_date"],
                  info["data_fetched_at"]]}),
        hide_index=True, width="stretch")
with right:
    st.subheader("All stocks at a glance")
    rows = []
    for s in stocks:
        i = metrics["stocks"][s]
        t = i["models"][i["best_model"]]["test"]
        rows.append({"Stock": s, "Selected model": MODEL_LABELS[i["best_model"]],
                     "Test R²": round(t["r2"], 3), "Test RMSE (₹)": round(t["rmse"], 2),
                     "Test MAE (₹)": round(t["mae"], 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.subheader("Model overview")
st.markdown(
    "- **Target:** closing price (₹).\n"
    "- **Split:** first 80% of days for training, last 20% for testing (chronological, no shuffling).\n"
    "- **Scaling:** `StandardScaler` fitted on the training window only.\n"
    "- **Models:** Random Forest (100 trees, max depth 10) and Linear Regression, one set per stock.\n"
    "- **Forecasting:** recursive multi-day forecast on the *Prediction Lab* page."
)
