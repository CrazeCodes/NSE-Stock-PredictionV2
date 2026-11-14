"""Central configuration. All paths are relative to the repository root."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
PRED_DIR = DATA_DIR / "predictions"
METRICS_FILE = DATA_DIR / "metrics.json"
SNAPSHOT_META_FILE = DATA_DIR / "snapshot_meta.json"

APP_NAME = "NSE Predict"
APP_TAGLINE = "Stock Market Analytics & Forecasting"

# Same five stocks and ticker alternatives as the original notebook.
STOCK_SYMBOLS = {
    "RELIANCE": ["RELIANCE.NS", "RELIANCE.BO"],
    "TCS": ["TCS.NS", "TCS.BO"],
    "HDFCBANK": ["HDFCBANK.NS", "HDFCBANK.BO"],
    "INFY": ["INFY.NS", "INFY.BO"],
    "ITC": ["ITC.NS", "ITC.BO"],
}
STOCK_LABELS = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "HDFCBANK": "HDFC Bank",
    "INFY": "Infosys",
    "ITC": "ITC Limited",
}
STOCKS = list(STOCK_SYMBOLS.keys())

# Original notebook settings
DEFAULT_PERIOD = "2y"
MIN_DATA_POINTS = 100
TRAIN_TEST_SPLIT = 0.8
MAX_HORIZON = 7

MODEL_NAMES = ["Random_Forest", "Linear_Regression"]
MODEL_LABELS = {"Random_Forest": "Random Forest", "Linear_Regression": "Linear Regression"}
RF_PARAMS = dict(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

BASE_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume",
                "Dividends", "Stock Splits", "Symbol"]

DISCLAIMER = (
    "Experimental portfolio project. Predictions come from simple machine-learning "
    "models trained on historical data and are **not financial advice**. "
    "Do not use this app for trading or investment decisions."
)
