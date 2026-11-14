"""Training pipeline: notebook methodology (RF + Linear Regression, 80/20 chronological split)."""
import json
from datetime import datetime
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .config import (METRICS_FILE, MIN_DATA_POINTS, MODELS_DIR, PRED_DIR, RF_PARAMS,
                     STOCKS, TRAIN_TEST_SPLIT)
from .data_loader import (DataUnavailableError, fetch_live, has_snapshot, load_snapshot,
                          save_snapshot, snapshot_timestamp)
from .evaluation import regression_metrics
from .preprocessing import add_technical_features, get_feature_columns


def train_stock(stock: str, raw_df: pd.DataFrame) -> dict:
    """Train both models for one stock, persist artifacts, return a metadata dict."""
    feats = add_technical_features(raw_df)
    feature_cols = get_feature_columns(feats)
    X, y = feats[feature_cols], feats["Close"]

    split_idx = int(len(X) * TRAIN_TEST_SPLIT)       # chronological split, no shuffling
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = feats["Date"].iloc[split_idx:]

    scaler = StandardScaler()                         # fit on training data only
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Random_Forest": RandomForestRegressor(**RF_PARAMS),
        "Linear_Regression": LinearRegression(),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    pred_table = pd.DataFrame({"Date": test_dates.values, "Actual": y_test.values})
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred_train, pred_test = model.predict(X_train_s), model.predict(X_test_s)
        results[name] = {
            "train": regression_metrics(y_train, pred_train),
            "test": regression_metrics(y_test, pred_test),
        }
        pred_table[name] = pred_test
        joblib.dump(model, MODELS_DIR / f"{stock}_{name}.joblib", compress=3)
    joblib.dump(scaler, MODELS_DIR / f"{stock}_scaler.joblib", compress=3)
    pred_table.to_csv(PRED_DIR / f"{stock}_test.csv", index=False)

    best = min(results, key=lambda n: results[n]["test"]["rmse"])  # notebook rule: lowest RMSE
    return {
        "best_model": best,
        "models": results,
        "feature_cols": feature_cols,
        "n_raw": int(len(raw_df)),
        "n_model_rows": int(len(feats)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split_date": str(feats["Date"].iloc[split_idx].date()),
        "data_start": str(raw_df["Date"].iloc[0].date()),
        "data_end": str(raw_df["Date"].iloc[-1].date()),
        "data_fetched_at": snapshot_timestamp(stock),
    }


def train_all(stocks: List[str] = None, use_live: bool = True, log=print) -> Dict:
    """Fetch (or reuse) data, train every stock, write data/metrics.json."""
    stocks = stocks or STOCKS
    out = {}
    for stock in stocks:
        raw = None
        if use_live:
            try:
                fetched = fetch_live(stock)
                if len(fetched) > MIN_DATA_POINTS:
                    save_snapshot(stock, fetched)
                    raw = fetched
                    log(f"[{stock}] fetched {len(raw)} rows from yfinance")
            except DataUnavailableError as exc:
                log(f"[{stock}] {exc}")
        if raw is None and has_snapshot(stock):
            raw = load_snapshot(stock)
            log(f"[{stock}] using local snapshot ({len(raw)} rows)")
        if raw is None or len(raw) <= MIN_DATA_POINTS:
            log(f"[{stock}] skipped: not enough data")
            continue
        out[stock] = train_stock(stock, raw)
        best = out[stock]["best_model"]
        t = out[stock]["models"][best]["test"]
        log(f"[{stock}] best={best} test RMSE={t['rmse']:.2f} R2={t['r2']:.3f}")

    if not out:
        raise DataUnavailableError("No data available for any stock; cannot train.")
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(
        {"generated_at": datetime.now().isoformat(timespec="seconds"), "stocks": out}, indent=2))
    return out
