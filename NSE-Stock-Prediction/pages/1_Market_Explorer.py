"""Page 2: Market Explorer."""
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

from src.config import STOCKS, STOCK_LABELS
from src.data_loader import DataUnavailableError, get_history
from src.ui import PALETTE, header, rupee, setup_page, show

setup_page("Market Explorer", "🔎")
header("Market Explorer", "Historical prices, candlesticks, volume and moving averages")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_history(stock, start, end):
    return get_history(stock, start, end)


c1, c2, c3 = st.columns([2, 1, 1])
stock = c1.selectbox("Stock", STOCKS, format_func=lambda s: f"{s} · {STOCK_LABELS[s]}")
start = c2.date_input("From", date.today() - timedelta(days=365), max_value=date.today())
end = c3.date_input("To", date.today(), max_value=date.today())
if start >= end:
    st.error("The start date must be before the end date.")
    st.stop()

c4, c5 = st.columns(2)
sma_windows = c4.multiselect("SMA windows (days)", [5, 10, 20, 50, 100, 200], default=[20, 50])
ema_windows = c5.multiselect("EMA windows (days)", [5, 10, 20, 50, 100, 200], default=[20])

try:
    with st.spinner("Loading market data..."):
        df, source, stamp = cached_history(stock, start, end)
except DataUnavailableError:
    st.error("Market data is currently unavailable and no local snapshot exists for this stock. "
             "Please try again later.")
    st.stop()
if df.empty:
    st.warning("No trading data in the selected range. Try widening the date range.")
    st.stop()

st.caption(f"Source: {source} · Retrieved/snapshot time: {stamp}. Daily bars; prices are "
           "split/dividend-adjusted by Yahoo Finance and may differ slightly from NSE official "
           "closes. Free data can be delayed or incomplete.")

first, last = float(df["Close"].iloc[0]), float(df["Close"].iloc[-1])
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Latest close", rupee(last))
k2.metric("Change over period", f"{(last / first - 1) * 100:+.2f}%")
k3.metric("Period high", rupee(float(df["High"].max())))
k4.metric("Period low", rupee(float(df["Low"].min())))
k5.metric("Avg. volume", f"{df['Volume'].mean():,.0f}")

for w in sma_windows:
    df[f"SMA_{w}"] = df["Close"].rolling(w).mean()
for w in ema_windows:
    df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()


def add_overlays(fig):
    for w in sma_windows:
        fig.add_trace(go.Scatter(x=df["Date"], y=df[f"SMA_{w}"], name=f"SMA {w}", mode="lines",
                                 line=dict(width=1.5)))
    for w in ema_windows:
        fig.add_trace(go.Scatter(x=df["Date"], y=df[f"EMA_{w}"], name=f"EMA {w}", mode="lines",
                                 line=dict(width=1.5, dash="dot")))


tab_line, tab_candle, tab_vol, tab_data = st.tabs(["Price", "Candlestick", "Volume", "Data"])
with tab_line:
    fig = go.Figure(go.Scatter(x=df["Date"], y=df["Close"], name="Close", mode="lines",
                               line=dict(color=PALETTE["actual"], width=2)))
    add_overlays(fig)
    fig.update_layout(title="Closing price", yaxis_title="Price (₹)", hovermode="x unified")
    fig.update_xaxes(rangeslider=dict(visible=True))
    show(fig, 520)
with tab_candle:
    fig = go.Figure(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"],
                                   close=df["Close"], name="OHLC"))
    add_overlays(fig)
    fig.update_layout(title="Candlestick (OHLC)", yaxis_title="Price (₹)",
                      xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    show(fig, 520)
with tab_vol:
    up = df["Close"] >= df["Open"]
    colors = [PALETTE["lr"] if u else PALETTE["forecast"] for u in up]
    fig = go.Figure(go.Bar(x=df["Date"], y=df["Volume"], marker_color=colors, name="Volume"))
    fig.update_layout(title="Traded volume (green = close ≥ open)", yaxis_title="Shares")
    show(fig, 420)
with tab_data:
    st.dataframe(df.drop(columns=["Symbol"]), hide_index=True, width="stretch")
    st.download_button("Download CSV", df.to_csv(index=False).encode(),
                       file_name=f"{stock}_{start}_{end}.csv", mime="text/csv")
