"""Feature engineering, kept identical to the original notebook."""
from typing import List

import pandas as pd


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_5_ratio"] = df["Close"] / df["MA_5"]
    df["MA_20_ratio"] = df["Close"] / df["MA_20"]
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
    df["High_Low_ratio"] = df["High"] / df["Low"]
    df["Close_Open_ratio"] = df["Close"] / df["Open"]
    df["Volume_MA_10"] = df["Volume"].rolling(10).mean()
    df["Volume_ratio"] = df["Volume"] / df["Volume_MA_10"]
    for lag in (1, 2, 3):
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
    return df.dropna().reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Same rule as the notebook: numeric columns except Date, Close, Symbol."""
    exclude = {"Date", "Close", "Symbol"}
    return [c for c in df.columns
            if c not in exclude and str(df[c].dtype) in ("float64", "int64")]
