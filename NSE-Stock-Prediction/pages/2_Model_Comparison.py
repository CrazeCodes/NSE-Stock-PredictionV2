"""Page 3: Model Comparison."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import MODEL_LABELS, MODEL_NAMES, STOCK_LABELS
from src.evaluation import residuals
from src.model_loader import available_stocks, ensure_artifacts, load_metrics, load_test_predictions
from src.ui import MODEL_COLORS, PALETTE, header, setup_page, show

setup_page("Model Comparison", "⚖️")
header("Model Comparison", "Random Forest vs Linear Regression on the same chronological test window")

try:
    ensure_artifacts()
    metrics = load_metrics()
    stocks = available_stocks()
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()

stock = st.selectbox("Stock", stocks, format_func=lambda s: f"{s} · {STOCK_LABELS[s]}")
info = metrics["stocks"][stock]
st.caption(f"Both models are evaluated on the same {info['n_test']} trading days "
           f"(from {info['split_date']}); training data comes strictly before that date.")

table = pd.DataFrame([{"Model": MODEL_LABELS[m],
                       "R²": info["models"][m]["test"]["r2"],
                       "MAE (₹)": info["models"][m]["test"]["mae"],
                       "RMSE (₹)": info["models"][m]["test"]["rmse"]} for m in MODEL_NAMES])
st.dataframe(table.style.format({"R²": "{:.4f}", "MAE (₹)": "{:.2f}", "RMSE (₹)": "{:.2f}"}),
             hide_index=True, width="stretch")
st.success(f"Lower-RMSE model for {stock}: **{MODEL_LABELS[info['best_model']]}**")

preds = load_test_predictions(stock)
fig = go.Figure()
fig.add_trace(go.Scatter(x=preds["Date"], y=preds["Actual"], name="Actual", mode="lines",
                         line=dict(color=PALETTE["actual"], width=2.5)))
for m in MODEL_NAMES:
    fig.add_trace(go.Scatter(x=preds["Date"], y=preds[m], name=MODEL_LABELS[m], mode="lines",
                             line=dict(color=MODEL_COLORS[m], width=1.8, dash="dot")))
fig.update_layout(title="Actual vs predicted close (test window)", yaxis_title="Price (₹)",
                  hovermode="x unified")
show(fig, 460)

fig = go.Figure()
for m in MODEL_NAMES:
    fig.add_trace(go.Histogram(x=residuals(preds["Actual"], preds[m]), name=MODEL_LABELS[m],
                               opacity=0.65, marker_color=MODEL_COLORS[m], nbinsx=30))
fig.add_vline(x=0, line_dash="dash", line_color=PALETTE["neutral"])
fig.update_layout(title="Residual distribution (predicted − actual)", barmode="overlay",
                  xaxis_title="Error (₹)", yaxis_title="Days")
show(fig, 380)

st.subheader("All stocks")
fig = make_subplots(rows=1, cols=2, subplot_titles=("Test RMSE (lower is better)",
                                                    "Test R² (higher is better)"))
for m in MODEL_NAMES:
    fig.add_trace(go.Bar(x=stocks, y=[metrics["stocks"][s]["models"][m]["test"]["rmse"] for s in stocks],
                         name=MODEL_LABELS[m], marker_color=MODEL_COLORS[m], legendgroup=m), row=1, col=1)
    fig.add_trace(go.Bar(x=stocks, y=[metrics["stocks"][s]["models"][m]["test"]["r2"] for s in stocks],
                         name=MODEL_LABELS[m], marker_color=MODEL_COLORS[m], legendgroup=m,
                         showlegend=False), row=1, col=2)
fig.update_layout(barmode="group")
show(fig, 420)
st.caption("Negative R² means the model did worse than predicting the test-set mean. Tree models "
           "cannot predict prices outside the range seen in training, which hurts when a stock "
           "trends beyond that range.")
