#!/usr/bin/env python3
"""
Fetch OHLCV from Coinbase Exchange public REST API, resample to 4h,
and save in both flat CSV and hyperview cache format.

Coinbase supports granularities: 60 / 300 / 900 / 3600 / 21600 / 86400
No native 4h — we fetch 1h (granularity=3600) and resample.
Max 300 bars per request → paginate with start/end window params.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

GRANULARITY  = 3600          # 1h in seconds
BARS_PER_REQ = 300           # Coinbase hard limit
RESAMPLE_TO  = "4h"
DATA_DIR     = Path(__file__).parent / "data"
HEADERS      = {"User-Agent": "Mozilla/5.0"}


def fetch_window(symbol: str, start: int, end: int) -> list:
    url = (
        f"https://api.exchange.coinbase.com/products/{symbol}/candles"
        f"?granularity={GRANULARITY}&start={start}&end={end}"
    )
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()          # [[time, low, high, open, close, volume], ...]


def fetch_all(symbol: str, days: int = 730) -> pd.DataFrame:
    now       = int(time.time())
    oldest    = now - days * 86_400
    window    = BARS_PER_REQ * GRANULARITY   # seconds covered per request

    all_rows: list = []
    end = now
    while end > oldest:
        start = max(end - window, oldest)
        rows  = fetch_window(symbol, start, end)
        if rows:
            all_rows.extend(rows)
        end = start - 1
        time.sleep(0.2)         # stay under rate limit

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["time", "low", "high", "open", "close", "volume"])
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1h bars into 4h OHLCV, anchored to midnight UTC."""
    ts = pd.to_datetime(df["time"], unit="s", utc=True)
    ohlcv = df.set_index(ts)[["open", "high", "low", "close", "volume"]]
    r = ohlcv.resample("4h", origin="epoch")
    agg = pd.DataFrame({
        "open":   r["open"].first(),
        "high":   r["high"].max(),
        "low":    r["low"].min(),
        "close":  r["close"].last(),
        "volume": r["volume"].sum(),
    }).dropna(subset=["close"])
    agg.insert(0, "time", [int(t.value // 10**9) for t in agg.index])
    return agg.reset_index(drop=True)


def save(df: pd.DataFrame, symbol: str) -> tuple[Path, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("-", "")

    # flat file (user-friendly name)
    flat_path = DATA_DIR / f"{safe}_4h.csv"
    df.to_csv(flat_path, index=False)

    # hyperview cache format
    hv_path = DATA_DIR / f"4h-regular-COINBASE-{safe}.csv"
    df.to_csv(hv_path, index=False)

    return flat_path, hv_path


def main() -> None:
    pairs = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD"]
    for symbol in pairs:
        print(f"  {symbol} 1h→4h ...", end=" ", flush=True)
        raw = fetch_all(symbol, days=730)
        if raw.empty:
            print("no data")
            continue
        df = resample_4h(raw)
        flat_path, hv_path = save(df, symbol)
        start = pd.Timestamp(int(df["time"].iloc[0]),  unit="s")
        end   = pd.Timestamp(int(df["time"].iloc[-1]), unit="s")
        print(f"{len(df):,} bars  {start.date()} → {end.date()}")
        print(f"    → {flat_path.name}")
        print(f"    → {hv_path.name}")


if __name__ == "__main__":
    main()
