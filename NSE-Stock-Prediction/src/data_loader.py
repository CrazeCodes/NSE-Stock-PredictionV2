"""Data access: live yfinance fetch with a local CSV snapshot fallback."""
import json
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from .config import (BASE_COLUMNS, DATA_DIR, DEFAULT_PERIOD, SNAPSHOT_META_FILE,
                     STOCK_SYMBOLS)


class DataUnavailableError(Exception):
    """Raised when neither live data nor a local snapshot is available."""


def _clean(df: pd.DataFrame, stock: str) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    first = df.columns[0]
    if first != "Date":
        df = df.rename(columns={first: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    df["Date"] = df["Date"].dt.normalize()
    for col in ("Dividends", "Stock Splits"):
        if col not in df.columns:
            df[col] = 0.0
    df["Symbol"] = stock
    df = df[BASE_COLUMNS].dropna(subset=["Open", "High", "Low", "Close"])
    return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def fetch_live(stock: str, period: Optional[str] = DEFAULT_PERIOD,
               start=None, end=None) -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance, trying each ticker alternative."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataUnavailableError("yfinance is not installed.") from exc

    last_error = None
    for symbol in STOCK_SYMBOLS[stock]:
        try:
            ticker = yf.Ticker(symbol)
            if start is not None:
                raw = ticker.history(start=start, end=end, auto_adjust=True)
            else:
                raw = ticker.history(period=period, auto_adjust=True)
            if raw is not None and not raw.empty:
                return _clean(raw, stock)
        except Exception as exc:  # network / rate limit / parsing
            last_error = exc
    raise DataUnavailableError(f"Live data unavailable for {stock}: {last_error}")


def snapshot_path(stock: str):
    return DATA_DIR / f"{stock}.csv"


def has_snapshot(stock: str) -> bool:
    return snapshot_path(stock).exists()


def _read_meta() -> dict:
    if SNAPSHOT_META_FILE.exists():
        try:
            return json.loads(SNAPSHOT_META_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_snapshot(stock: str, df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(snapshot_path(stock), index=False)
    meta = _read_meta()
    meta[stock] = datetime.now().isoformat(timespec="seconds")
    SNAPSHOT_META_FILE.write_text(json.dumps(meta, indent=2))


def load_snapshot(stock: str) -> pd.DataFrame:
    path = snapshot_path(stock)
    if not path.exists():
        raise DataUnavailableError(f"No local data snapshot for {stock}.")
    return _clean(pd.read_csv(path, parse_dates=["Date"]), stock)


def snapshot_timestamp(stock: str) -> str:
    return _read_meta().get(stock, "unknown")


def get_history(stock: str, start, end) -> Tuple[pd.DataFrame, str, str]:
    """Market Explorer data: live first, snapshot fallback.

    Returns (dataframe, source_label, timestamp).
    """
    try:
        end_inclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
        df = fetch_live(stock, start=pd.Timestamp(start), end=end_inclusive)
        return df, "Live (Yahoo Finance via yfinance)", datetime.now().strftime("%Y-%m-%d %H:%M")
    except DataUnavailableError:
        snap = load_snapshot(stock)  # raises DataUnavailableError if missing
        mask = (snap["Date"] >= pd.Timestamp(start)) & (snap["Date"] <= pd.Timestamp(end))
        return (snap.loc[mask].reset_index(drop=True),
                "Local snapshot (live feed unavailable)", snapshot_timestamp(stock))
