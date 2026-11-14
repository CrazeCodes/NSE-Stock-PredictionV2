# NSE Predict: Stock Market Analytics & Forecasting

Interactive Streamlit dashboard built from a Kaggle notebook. It trains a Random Forest and a
Linear Regression model for five NSE stocks (RELIANCE, TCS, HDFCBANK, INFY, ITC) on two years
of daily Yahoo Finance data and lets you explore prices, compare models, inspect errors and
generate short recursive forecasts.

> **Disclaimer:** experimental portfolio project, **not financial advice**. Do not use it for trading.

## Pages
| Page | Purpose |
|---|---|
| Overview (`app.py`) | Summary, latest price, key metrics, dataset info |
| Market Explorer | Live/snapshot OHLCV, candlesticks, volume, SMA/EMA |
| Model Comparison | RF vs Linear Regression: R², MAE, RMSE, residuals |
| Prediction Lab | 1-7 trading-day recursive forecast |
| Model Evaluation | Train vs test, residuals, feature importance, limitations |

## Methodology (version 1: original notebook)
Chronological 80/20 split, `StandardScaler` fitted on training data, 21 input features (raw OHLCV columns plus engineered ones),
best model per stock chosen by test RMSE. See the limitations section on the *Model Evaluation* page.
Version 2 (next-day targets, walk-forward validation, naive baseline) is planned.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train_all.py      # downloads data, trains models, writes data/ and models/
streamlit run app.py
```

## Project layout
```
app.py                  Overview page
pages/                  Market Explorer, Model Comparison, Prediction Lab, Model Evaluation
src/                    config, data_loader, preprocessing, model_training, model_loader,
                        prediction, evaluation, ui
scripts/train_all.py    offline training
data/                   CSV snapshots, metrics.json, test-set predictions
models/                 trained .joblib files (a few MB in total)
.streamlit/config.toml  theme
```

## Deploy on Streamlit Community Cloud
1. Run `python scripts/train_all.py` locally, then commit `data/` and `models/`.
2. Push the repo to GitHub (public).
3. Go to share.streamlit.io, sign in with GitHub, click **Create app**.
4. Pick the repo, branch `main`, main file `app.py`, then **Deploy**.
