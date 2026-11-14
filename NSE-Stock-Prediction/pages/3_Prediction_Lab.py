"""Page 4: Prediction Lab."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

from src.config import MAX_HORIZON, MODEL_LABELS, MODEL_NAMES, STOCK_LABELS
from src.model_loader import (available_stocks, ensure_artifacts, load_metrics, load_model,
                              load_scaler, load_snapshot_df)
from src.prediction import forecast_next_days
from src.ui import PALETTE, disclaimer, header, rupee, setup_page, show

setup_page("Prediction Lab", "🧪")
header("Prediction Lab", "Recursive multi-day forecasts from the trained models")
disclaimer()

try:
    ensure_artifacts()
    metrics = load_metrics()
    stocks = available_stocks()
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()

c1, c2, c3 = st.columns([2, 2, 2])
stock = c1.selectbox("Stock", stocks, format_func=lambda s: f"{s} · {STOCK_LABELS[s]}")
info = metrics["stocks"][stock]
model_name = c2.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(info["best_model"]),
                          format_func=lambda m: MODEL_LABELS[m])
horizon = c3.slider("Forecast horizon (trading days)", 1, MAX_HORIZON, 5)
st.caption("Only the five stocks above have trained models; other tickers are not supported. "
           f"The maximum horizon is {MAX_HORIZON} trading days.")

try:
    raw = load_snapshot_df(stock)
    with st.spinner("Generating forecast..."):
        fc = forecast_next_days(load_model(stock, model_name), load_scaler(stock),
                                info["feature_cols"], raw, horizon)
except Exception as exc:
    st.error("The forecast could not be generated for this selection.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

last_close, last_date = float(raw["Close"].iloc[-1]), raw["Date"].iloc[-1]
final = fc.iloc[-1]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Last close", rupee(last_close), last_date.strftime("%d %b %Y"))
k2.metric(f"Day {horizon} forecast", rupee(final["Predicted_Price"]),
          f"{final['Change_vs_Last_Close_%']:+.2f}% vs last close")
k3.metric("Forecast window", f"{fc['Date'].iloc[0]:%d %b} → {fc['Date'].iloc[-1]:%d %b}")
k4.metric("Model test RMSE", rupee(info["models"][model_name]["test"]["rmse"]))

hist = raw.tail(60)
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Close"], name="Historical close", mode="lines",
                         line=dict(color=PALETTE["actual"], width=2.5)))
fig.add_trace(go.Scatter(x=[last_date, *fc["Date"]], y=[last_close, *fc["Predicted_Price"]],
                         name="Forecast", mode="lines+markers",
                         line=dict(color=PALETTE["forecast"], width=2.5, dash="dash")))
fig.update_layout(title=f"{stock}: last 60 sessions and {horizon}-day forecast",
                  yaxis_title="Price (₹)", hovermode="x unified")
show(fig, 460)

out = fc.copy()
out["Date"] = out["Date"].dt.strftime("%a %d %b %Y")
out["Predicted_Price"] = out["Predicted_Price"].round(2)
out["Change_vs_Last_Close_%"] = out["Change_vs_Last_Close_%"].round(2)
st.dataframe(out.rename(columns={"Predicted_Price": "Predicted price (₹)",
                                 "Change_vs_Last_Close_%": "Change vs last close (%)"}),
             hide_index=True, width="stretch")

with st.expander("Methodology and limitations", expanded=True):
    st.markdown(
        f"- The model takes the {len(info['feature_cols'])} input features for the most recent trading day, scaled "
        f"with the training-window scaler, and predicts that day's close.\n"
        f"- For multi-day horizons the prediction is appended as a projected day (open = close = "
        f"prediction, high/low ±1%, volume = 10-day average) and the features are recomputed. "
        f"**Errors compound with each step**, so later days are less reliable.\n"
        f"- No random noise is added. Dates skip weekends but not NSE market holidays.\n"
        f"- Historical accuracy (see *Model Evaluation*) does not imply future performance. "
        f"This is a learning project, not a trading tool."
    )
