"""Recursive multi-day forecast using the trained notebook models.

Differences from the notebook (deliberate):
  * no random noise is added to predictions;
  * forecast dates skip weekends (business days);
  * projected rows carry Dividends / Stock Splits = 0 so the feature pipeline
    really updates between steps.
Projected rows assume Open = Close = prediction, High = +1%, Low = -1%,
Volume = trailing 10-day mean (same assumption as the notebook).
"""
import pandas as pd

from .config import MAX_HORIZON
from .preprocessing import add_technical_features


def forecast_next_days(model, scaler, feature_cols, raw_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"Horizon must be between 1 and {MAX_HORIZON} trading days.")

    current = raw_df.copy()
    rows = []
    for step in range(1, horizon + 1):
        feats = add_technical_features(current)
        if feats.empty:
            raise ValueError("Not enough history to build features.")
        x = feats[feature_cols].iloc[[-1]]
        price = float(model.predict(scaler.transform(x))[0])

        next_date = current["Date"].iloc[-1] + pd.offsets.BDay(1)
        rows.append({"Day": step, "Date": next_date, "Predicted_Price": price})

        new_row = {c: 0.0 for c in current.columns if c not in ("Date", "Symbol")}
        new_row.update({
            "Date": next_date, "Symbol": current["Symbol"].iloc[0],
            "Open": price, "High": price * 1.01, "Low": price * 0.99, "Close": price,
            "Volume": float(current["Volume"].tail(10).mean()),
        })
        current = pd.concat([current, pd.DataFrame([new_row])], ignore_index=True)

    out = pd.DataFrame(rows)
    last_close = float(raw_df["Close"].iloc[-1])
    out["Change_vs_Last_Close_%"] = (out["Predicted_Price"] / last_close - 1) * 100
    return out
