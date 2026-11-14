"""Page 5: Model Evaluation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import MODEL_LABELS, MODEL_NAMES, STOCK_LABELS
from src.evaluation import feature_importance, residuals
from src.model_loader import (available_stocks, ensure_artifacts, load_metrics, load_model,
                              load_test_predictions)
from src.ui import MODEL_COLORS, PALETTE, header, setup_page, show

setup_page("Model Evaluation", "📊")
header("Model Evaluation", "Train vs test performance, residuals and feature importance")

try:
    ensure_artifacts()
    metrics = load_metrics()
    stocks = available_stocks()
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()

c1, c2 = st.columns(2)
stock = c1.selectbox("Stock", stocks, format_func=lambda s: f"{s} · {STOCK_LABELS[s]}")
info = metrics["stocks"][stock]
model_name = c2.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(info["best_model"]),
                          format_func=lambda m: MODEL_LABELS[m])

rows = []
for m in MODEL_NAMES:
    for split in ("train", "test"):
        v = info["models"][m][split]
        rows.append({"Model": MODEL_LABELS[m], "Split": split.capitalize(),
                     "R²": v["r2"], "MAE (₹)": v["mae"], "RMSE (₹)": v["rmse"]})
st.subheader("Training vs testing performance")
st.dataframe(pd.DataFrame(rows).style.format({"R²": "{:.4f}", "MAE (₹)": "{:.2f}", "RMSE (₹)": "{:.2f}"}),
             hide_index=True, width="stretch")
st.caption(f"Train: {info['n_train']} days before {info['split_date']}. "
           f"Test: {info['n_test']} days from {info['split_date']}. "
           "Only the test rows measure out-of-sample performance.")

preds = load_test_predictions(stock)
pred = preds[model_name]
err = residuals(preds["Actual"], pred)

fig = go.Figure()
fig.add_trace(go.Scatter(x=preds["Date"], y=preds["Actual"], name="Actual", mode="lines",
                         line=dict(color=PALETTE["actual"], width=2.5)))
fig.add_trace(go.Scatter(x=preds["Date"], y=pred, name=MODEL_LABELS[model_name], mode="lines",
                         line=dict(color=MODEL_COLORS[model_name], width=2, dash="dot")))
fig.update_layout(title="Actual vs predicted (test window)", yaxis_title="Price (₹)",
                  hovermode="x unified")
show(fig, 420)

a, b = st.columns(2)
with a:
    lo, hi = float(min(preds["Actual"].min(), pred.min())), float(max(preds["Actual"].max(), pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=preds["Actual"], y=pred, mode="markers", name="Days",
                             marker=dict(color=MODEL_COLORS[model_name], opacity=0.7)))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit",
                             line=dict(color=PALETTE["neutral"], dash="dash")))
    fig.update_layout(title="Predicted vs actual", xaxis_title="Actual (₹)", yaxis_title="Predicted (₹)")
    show(fig, 380)
with b:
    fig = go.Figure(go.Histogram(x=err, nbinsx=30, marker_color=PALETTE["rf"]))
    fig.add_vline(x=0, line_dash="dash", line_color=PALETTE["neutral"])
    fig.update_layout(title="Residual distribution", xaxis_title="Error (₹)", yaxis_title="Days")
    show(fig, 380)

fig = go.Figure(go.Scatter(x=preds["Date"], y=err, mode="lines+markers",
                           line=dict(color=PALETTE["forecast"], width=1.5), marker=dict(size=4)))
fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["neutral"])
fig.update_layout(title="Residuals over time (check for drift or clustering)", yaxis_title="Error (₹)")
show(fig, 320)

st.subheader("Feature importance")
fi = feature_importance(load_model(stock, model_name), info["feature_cols"]).head(15)
fig = go.Figure(go.Bar(x=fi["Importance"][::-1], y=fi["Feature"][::-1], orientation="h",
                       marker_color=MODEL_COLORS[model_name]))
fig.update_layout(title=f"{fi['Kind'].iloc[0]} (top 15)")
show(fig, 460)

st.subheader("Methodology notes and limitations")
st.markdown(
    "This page documents **version 1**, which reproduces the original notebook's methodology. "
    "Please read the numbers with the following in mind:\n\n"
    "- **Same-day features.** Several inputs (open, high, low, and moving averages) are computed "
    "from the same day whose close is being predicted. The scores therefore describe how well the "
    "model *reconstructs a day's close from that day's data*, which is easier than forecasting "
    "tomorrow. Very high R² values are expected here and should not be read as forecasting skill.\n"
    "- **Price-level metrics are flattering.** Prices trend, so R² on price levels is high even for "
    "naive models. No naive baseline (e.g. \"tomorrow = today\") is shown in version 1.\n"
    "- **Small test window and one split.** Roughly the last 95 trading days, a single chronological "
    "split, and the better model is picked by test RMSE, so the reported test score is slightly "
    "optimistic.\n"
    "- **Tree-model range limit.** Random Forests cannot output prices outside their training "
    "range, which explains negative R² on stocks that moved to new highs or lows.\n"
    "- **Multi-day forecasts compound error** and rely on projected (not real) intermediate days.\n\n"
    "A version 2 with next-day targets, walk-forward validation and a naive baseline is planned. "
    "This is an experimental portfolio project and **not financial advice**."
)
