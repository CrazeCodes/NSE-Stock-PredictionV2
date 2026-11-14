"""Fetch data, train all models, write models/ and data/. Run from the repo root:

    python scripts/train_all.py            # fetch live data, then train
    python scripts/train_all.py --offline  # reuse existing data/*.csv snapshots
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_training import train_all  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="do not call yfinance")
    args = parser.parse_args()
    train_all(use_live=not args.offline)
    print("Done. Commit data/ and models/ to GitHub.")
