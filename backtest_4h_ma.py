#!/usr/bin/env python3
"""
Tier-1 SMA/EMA grid search on 4h BTC/USD data (Coinbase, 5yr).
Gate: N≥100 trades AND Δsharpe≥+0.02 AND Δmax_dd≥+0.5pp vs baseline.
Results appended to strategy_journal.csv.
"""
from __future__ import annotations

import csv
import json
import warnings
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).parent
DATA_FILE    = ROOT / "data" / "4h-regular-COINBASE-BTCUSD.csv"
BASELINE_FILE= ROOT / "baseline.json"
JOURNAL_FILE = ROOT / "strategy_journal.csv"

FEE          = 0.006      # 0.6% round-trip taker
CAPITAL      = 100_000.0
DEPLOY       = 0.95

# Gate thresholds (CLAUDE.md §2 + §3)
MIN_TRADES   = 100
DELTA_SHARPE = 0.02
DELTA_DD     = 0.5        # pp less negative

SHORT_WINDOWS = [5, 10, 15, 20, 30, 50]
LONG_WINDOWS  = [15, 20, 30, 50, 100, 200]
MA_TYPES      = ["sma", "ema"]


def ma(series: pd.Series, window: int, kind: str) -> pd.Series:
    if kind == "ema":
        return series.ewm(span=window, adjust=False).mean()
    return series.rolling(window).mean()


def run_backtest(df: pd.DataFrame, sw: int, lw: int, kind: str) -> dict:
    close  = df["close"].values
    times  = df["time"].values
    n      = len(close)

    sma_s = ma(pd.Series(close), sw, kind).values
    sma_l = ma(pd.Series(close), lw, kind).values

    cash  = CAPITAL
    qty   = 0.0
    entry = 0.0
    in_pos= False
    equity= []
    n_trades = 0

    for i in range(lw, n):
        prev_cross = sma_s[i-1] > sma_l[i-1]
        curr_cross = sma_s[i]   > sma_l[i]

        if not in_pos and not prev_cross and curr_cross:       # golden cross
            spent = cash * DEPLOY
            qty   = spent / (close[i] * (1 + FEE/2))
            cash -= spent
            entry = close[i]
            in_pos= True
            n_trades += 1

        elif in_pos and prev_cross and not curr_cross:          # death cross
            cash  += qty * close[i] * (1 - FEE/2)
            qty    = 0.0
            in_pos = False

        equity.append(cash + qty * close[i])

    if in_pos:
        cash += qty * close[-1] * (1 - FEE/2)

    eq = pd.Series(equity, index=pd.to_datetime(times[lw:], unit="s", utc=True))
    ret = eq.pct_change().dropna()
    if ret.empty or n_trades == 0:
        return {}

    cagr    = float(qs.stats.cagr(ret))   * 100
    sharpe  = float(qs.stats.sharpe(ret))
    max_dd  = float(qs.stats.max_drawdown(ret)) * 100
    sortino = float(qs.stats.sortino(ret))
    calmar  = float(qs.stats.calmar(ret))

    years   = (times[-1] - times[lw]) / (365.25 * 86_400)
    win_year= sum(
        eq.resample("YE").last().pct_change().dropna() > 0
    ) / max(1, len(eq.resample("YE").last().pct_change().dropna()))

    return dict(
        strategy="btc_4h_ma_crossover",
        ma_type=kind, short_window=sw, long_window=lw,
        n_trades=n_trades, cagr=round(cagr,4),
        sharpe=round(sharpe,4), max_dd=round(max_dd,4),
        sortino=round(sortino,4), calmar=round(calmar,4),
        win_year_pct=round(win_year*100,2),
        recorded_at=datetime.now(timezone.utc).date().isoformat(),
        notes=f"4h coinbase 5yr {kind.upper()} {sw}/{lw}",
    )


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    bl = json.loads(BASELINE_FILE.read_text())
    bl_sharpe = bl["sharpe"]
    bl_dd     = bl["max_dd"]

    print(f"Baseline Sharpe={bl_sharpe:.4f}  MaxDD={bl_dd:.2f}%")
    print(f"Data: {len(df)} bars  "
          f"{pd.Timestamp(df['time'].iloc[0], unit='s').date()} → "
          f"{pd.Timestamp(df['time'].iloc[-1], unit='s').date()}")
    print(f"Grid: {len(SHORT_WINDOWS)}×{len(LONG_WINDOWS)}×{len(MA_TYPES)} = "
          f"{len(SHORT_WINDOWS)*len(LONG_WINDOWS)*len(MA_TYPES)} combos\n")

    # load existing journal columns
    existing_cols: list[str] = []
    if JOURNAL_FILE.exists():
        existing_cols = pd.read_csv(JOURNAL_FILE, nrows=0).columns.tolist()

    results = []
    passes  = []

    for kind, sw, lw in product(MA_TYPES, SHORT_WINDOWS, LONG_WINDOWS):
        if sw >= lw:
            continue
        r = run_backtest(df, sw, lw, kind)
        if not r:
            continue
        results.append(r)

        n_ok   = r["n_trades"] >= MIN_TRADES
        sh_ok  = r["sharpe"] - bl_sharpe >= DELTA_SHARPE
        dd_ok  = r["max_dd"] - bl_dd     >= DELTA_DD
        gate   = n_ok and sh_ok and dd_ok
        marker = "✓ PASS" if gate else "·"
        if gate:
            passes.append(r)

        print(f"  {marker} {kind.upper()} {sw:>2}/{lw:<3} "
              f"N={r['n_trades']:>3}  "
              f"Sharpe={r['sharpe']:>7.4f} (Δ{r['sharpe']-bl_sharpe:+.4f})  "
              f"MaxDD={r['max_dd']:>7.2f}% (Δ{r['max_dd']-bl_dd:+.2f}pp)  "
              f"CAGR={r['cagr']:>7.2f}%")

    # append to journal
    all_cols = list({c for r in results for c in r} | set(existing_cols))
    write_header = not JOURNAL_FILE.exists()
    with open(JOURNAL_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\n{'─'*70}")
    print(f"Gate passes: {len(passes)} / {len(results)}")

    if passes:
        best = max(passes, key=lambda r: r["sharpe"])
        print(f"\nBest: {best['ma_type'].upper()} {best['short_window']}/{best['long_window']}  "
              f"Sharpe={best['sharpe']:.4f}  CAGR={best['cagr']:.2f}%  MaxDD={best['max_dd']:.2f}%  "
              f"N={best['n_trades']}")


if __name__ == "__main__":
    main()
