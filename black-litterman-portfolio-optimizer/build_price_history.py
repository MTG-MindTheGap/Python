"""
Regenerates sp500_stocks.csv — the long-format daily OHLCV price history
that the main notebook reads. This file is intentionally not checked into
the repo (it's ~300MB for the full 2000-2024 history), so run this once
before opening the notebook.

Usage:
    pip install yfinance pandas tqdm
    python build_price_history.py                  # full history since 2000
    python build_price_history.py --start 2015-01-01  # shorter history, faster

Output schema matches what the notebook expects:
    date, open, high, low, close, volume, symbol

Alternative: if you'd rather not re-download 500+ tickers from yfinance,
the same schema is available as a static download from the Kaggle dataset
"S&P 500 Stocks (daily updated)" — https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks
Just rename the downloaded file to sp500_stocks.csv and place it in this folder.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import pandas as pd
import yfinance as yf


def load_universe(companies_csv: str = "sp500_companies.csv") -> list[str]:
    companies = pd.read_csv(companies_csv)
    return sorted(companies["symbol"].unique().tolist())


def download_all(tickers: list[str], start: str, end: Optional[str]) -> pd.DataFrame:
    frames = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}", end="\r")
        try:
            hist = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        except Exception as e:
            print(f"\n  skipped {ticker}: {e}")
            continue
        if hist.empty:
            continue
        hist = hist.reset_index()
        hist.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in hist.columns]
        hist["symbol"] = ticker
        frames.append(hist[["date", "open", "high", "low", "close", "volume", "symbol"]])
        time.sleep(0.05)  # be polite to Yahoo Finance
    print()
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2000-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), default = today")
    parser.add_argument("--companies-csv", default="sp500_companies.csv")
    parser.add_argument("--out", default="sp500_stocks.csv")
    args = parser.parse_args()

    tickers = load_universe(args.companies_csv)
    print(f"Downloading {len(tickers)} tickers from {args.start} to {args.end or 'today'}...")

    df = download_all(tickers, args.start, args.end)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
