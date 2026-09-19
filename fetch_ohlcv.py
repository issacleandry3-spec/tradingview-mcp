#!/usr/bin/env python3
"""
Download public OHLCV data via yfinance and save in hyperview's CSV format.

Output: data/{interval}-regular-YAHOO-{SYMBOL}.csv
Columns: time (Unix seconds), open, high, low, close, volume

Usage:
    python3 fetch_ohlcv.py
    python3 fetch_ohlcv.py --symbols BTC-USD ETH-USD --interval 4h --days 730
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"

# yfinance supports 4h only for the last 730 days; 1h for last 730 days
VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "1h",
                   "90m", "4h", "1d", "5d", "1wk", "1mo", "3mo"]

DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD"]


def download(ticker: str, interval: str, days: int) -> pd.DataFrame:
    period = f"{min(days, 730)}d" if interval in ("1h", "4h", "60m", "90m") else f"{days}d"
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns produced by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df.rename(columns={"open": "open", "high": "high", "low": "low",
                             "close": "close", "volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].copy()

    # Convert DatetimeIndex → Unix seconds (Timestamp.value is always nanoseconds)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("UTC")
    df.insert(0, "time", [int(t.value // 10**9) for t in df.index])

    df = df.dropna(subset=["close"])
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def save(df: pd.DataFrame, ticker: str, interval: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("-", "").replace("/", "")
    fname = f"{interval}-regular-YAHOO-{safe}.csv"
    path = DATA_DIR / fname
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OHLCV via yfinance → hyperview CSV")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="Yahoo Finance tickers, e.g. BTC-USD ETH-USD SOL-USD LINK-USD")
    parser.add_argument("--interval", default="4h", choices=VALID_INTERVALS)
    parser.add_argument("--days", type=int, default=730,
                        help="History depth in days (max 730 for sub-daily intervals)")
    args = parser.parse_args()

    for ticker in args.symbols:
        print(f"  {ticker} {args.interval} ...", end=" ", flush=True)
        df = download(ticker, args.interval, args.days)
        if df.empty:
            print("no data returned")
            continue
        path = save(df, ticker, args.interval)
        start = pd.Timestamp(int(df["time"].iloc[0]),  unit="s")
        end   = pd.Timestamp(int(df["time"].iloc[-1]), unit="s")
        print(f"{len(df):,} bars  {start.date()} → {end.date()}  → {path.name}")


if __name__ == "__main__":
    main()
