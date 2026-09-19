#!/usr/bin/env python3
"""
Tier-2 filter search on top of SMA 10/200 4h baseline.
Tests RSI filter, volume confirmation, and ATR position sizing.
Gate: N≥100 AND Δsharpe≥+0.02 AND Δmax_dd≥+0.5pp vs baseline_4h.json
"""
from __future__ import annotations

import csv
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs
import ta

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).parent
DATA_FILE     = ROOT / "data" / "4h-regular-COINBASE-BTCUSD.csv"
BASELINE_FILE = ROOT / "baseline_4h.json"
JOURNAL_FILE  = ROOT / "strategy_journal.csv"

FEE     = 0.006
CAPITAL = 100_000.0
DEPLOY  = 0.95
SW, LW  = 10, 200          # fixed from baseline

MIN_TRADES   = 100
DELTA_SHARPE = 0.02
DELTA_DD     = 0.5


# ── core metrics ─────────────────────────────────────────────────────────────

def calc_metrics(equity: np.ndarray, times: np.ndarray) -> dict:
    eq  = pd.Series(equity, index=pd.to_datetime(times, unit="s", utc=True))
    ret = eq.pct_change().dropna()
    if ret.empty:
        return {}
    return dict(
        cagr    = round(float(qs.stats.cagr(ret))          * 100, 4),
        sharpe  = round(float(qs.stats.sharpe(ret)),               4),
        max_dd  = round(float(qs.stats.max_drawdown(ret))  * 100, 4),
        sortino = round(float(qs.stats.sortino(ret)),              4),
        calmar  = round(float(qs.stats.calmar(ret)),               4),
    )


# ── backtest engine ───────────────────────────────────────────────────────────

def backtest(
    df: pd.DataFrame,
    rsi_filter: bool  = False,
    rsi_thresh: int   = 60,
    rsi_window: int   = 14,
    vol_confirm: bool = False,
    vol_mult: float   = 1.5,
    vol_window: int   = 20,
    atr_sizing: bool  = False,
    risk_pct: float   = 0.02,
    atr_mult: float   = 2.0,
) -> dict:
    close  = df["close"].values
    volume = df["volume"].values
    times  = df["time"].values
    n      = len(close)

    sma_s  = pd.Series(close).rolling(SW).mean().values
    sma_l  = pd.Series(close).rolling(LW).mean().values
    rsi_v  = ta.momentum.rsi(pd.Series(close), window=rsi_window).values if rsi_filter else None
    vol_ma = pd.Series(volume).rolling(vol_window).mean().values          if vol_confirm else None
    atr_v  = ta.volatility.average_true_range(
                 pd.Series(df["high"].values),
                 pd.Series(df["low"].values),
                 pd.Series(close), window=14).values                      if atr_sizing  else None

    cash = CAPITAL; qty = 0.0; in_pos = False
    equity = []; n_trades = 0

    for i in range(LW, n):
        golden = (not (sma_s[i-1] > sma_l[i-1])) and (sma_s[i] > sma_l[i])
        death  = (sma_s[i-1] > sma_l[i-1]) and (not (sma_s[i] > sma_l[i]))

        if not in_pos and golden:
            if rsi_filter  and (np.isnan(rsi_v[i]) or rsi_v[i] >= rsi_thresh):
                pass
            elif vol_confirm and (np.isnan(vol_ma[i]) or volume[i] <= vol_mult * vol_ma[i]):
                pass
            else:
                if atr_sizing and not np.isnan(atr_v[i]) and atr_v[i] > 0:
                    risk_amt = cash * risk_pct
                    qty_atr  = risk_amt / (atr_v[i] * atr_mult)
                    max_qty  = (cash * DEPLOY) / (close[i] * (1 + FEE/2))
                    qty      = min(qty_atr, max_qty)
                else:
                    qty = (cash * DEPLOY) / (close[i] * (1 + FEE/2))
                cash -= qty * close[i] * (1 + FEE/2)
                in_pos = True
                n_trades += 1

        elif in_pos and death:
            cash  += qty * close[i] * (1 - FEE/2)
            qty    = 0.0
            in_pos = False

        equity.append(cash + qty * close[i])

    if in_pos:
        cash += qty * close[-1] * (1 - FEE/2)

    if n_trades == 0:
        return {}

    m = calc_metrics(np.array(equity), times[LW:])
    if not m:
        return {}

    return dict(n_trades=n_trades, **m,
                rsi_filter=rsi_filter, rsi_threshold=rsi_thresh, rsi_window=rsi_window,
                vol_confirm=vol_confirm, vol_mult=vol_mult,
                atr_sizing=atr_sizing, risk_pct=risk_pct, atr_mult=atr_mult)


# ── journal ───────────────────────────────────────────────────────────────────

FIXED_FIELDS = [
    "strategy","timeframe","ma_type","short_window","long_window","tier",
    "rsi_filter","rsi_threshold","rsi_window",
    "vol_confirm","vol_mult",
    "atr_sizing","risk_pct","atr_mult",
    "n_trades","cagr","sharpe","max_dd","sortino","calmar",
    "recorded_at","notes",
]

def write_row(row: dict) -> None:
    exists = JOURNAL_FILE.exists()
    # merge with any extra cols already in file
    if exists:
        existing = pd.read_csv(JOURNAL_FILE, nrows=0).columns.tolist()
        cols = list(dict.fromkeys(existing + FIXED_FIELDS))
    else:
        cols = FIXED_FIELDS
    with open(JOURNAL_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_and_print(label: str, bl: dict, r: dict, extra: dict) -> dict | None:
    if not r:
        print(f"  {label:<45}  no trades")
        return None

    n_ok  = r["n_trades"] >= MIN_TRADES
    sh_ok = r["sharpe"] - bl["sharpe"] >= DELTA_SHARPE
    dd_ok = r["max_dd"]  - bl["max_dd"] >= DELTA_DD
    gate  = n_ok and sh_ok and dd_ok
    mark  = "✓ PASS" if gate else "·     "
    dn    = "N<100 " if not n_ok else ("Sh    " if not sh_ok else ("DD    " if not dd_ok else ""))

    print(f"  {mark} {label:<45} "
          f"N={r['n_trades']:>3}  "
          f"Sharpe={r['sharpe']:>7.4f} (Δ{r['sharpe']-bl['sharpe']:+.4f})  "
          f"MaxDD={r['max_dd']:>7.2f}% (Δ{r['max_dd']-bl['max_dd']:+.2f}pp)  "
          f"CAGR={r['cagr']:>7.2f}%"
          + (f"  [{dn.strip()}]" if not gate else ""))

    row = dict(
        strategy="btc_4h_sma10_200_tier2",
        timeframe="4h", ma_type="sma",
        short_window=SW, long_window=LW,
        tier=2,
        recorded_at=datetime.now(timezone.utc).date().isoformat(),
        **r, **extra,
    )
    write_row(row)
    return row if gate else None


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(DATA_FILE)
    bl = json.loads(BASELINE_FILE.read_text())

    print(f"Baseline SMA {SW}/{LW} 4h: "
          f"Sharpe={bl['sharpe']:.4f}  MaxDD={bl['max_dd']:.2f}%  N={bl['n_trades']}")
    print(f"Data: {len(df)} bars  "
          f"{pd.Timestamp(df['time'].iloc[0],  unit='s').date()} → "
          f"{pd.Timestamp(df['time'].iloc[-1], unit='s').date()}\n")

    passes = []

    # ── 2A: RSI filter ────────────────────────────────────────────────────────
    print("── Tier-2A: RSI entry filter ────────────────────────────────────────")
    for thresh in [50, 55, 60, 65, 70]:
        r = backtest(df, rsi_filter=True, rsi_thresh=thresh)
        p = run_and_print(f"RSI<{thresh}", bl, r,
                          {"notes": f"4h SMA10/200 + RSI<{thresh}"})
        if p:
            passes.append(p)

    # ── 2B: volume confirmation ───────────────────────────────────────────────
    print("\n── Tier-2B: Volume confirmation ─────────────────────────────────────")
    for mult in [1.2, 1.5, 2.0, 2.5]:
        r = backtest(df, vol_confirm=True, vol_mult=mult)
        p = run_and_print(f"vol>{mult}×20ma", bl, r,
                          {"notes": f"4h SMA10/200 + vol>{mult}x"})
        if p:
            passes.append(p)

    # ── 2C: ATR position sizing ───────────────────────────────────────────────
    print("\n── Tier-2C: ATR position sizing ─────────────────────────────────────")
    for risk, mult in [(0.01,1.5),(0.01,2.0),(0.02,1.5),(0.02,2.0),(0.03,2.0)]:
        r = backtest(df, atr_sizing=True, risk_pct=risk, atr_mult=mult)
        p = run_and_print(f"ATR risk={risk*100:.0f}% mult={mult}", bl, r,
                          {"notes": f"4h SMA10/200 ATR risk={risk} mult={mult}"})
        if p:
            passes.append(p)

    # ── 2D: combined best combos ──────────────────────────────────────────────
    print("\n── Tier-2D: RSI + Volume combined ───────────────────────────────────")
    for thresh in [55, 60]:
        for mult in [1.5, 2.0]:
            r = backtest(df, rsi_filter=True, rsi_thresh=thresh,
                         vol_confirm=True, vol_mult=mult)
            p = run_and_print(f"RSI<{thresh} + vol>{mult}×", bl, r,
                              {"notes": f"4h SMA10/200 RSI<{thresh}+vol>{mult}x"})
            if p:
                passes.append(p)

    print(f"\n{'─'*75}")
    print(f"Gate passes: {len(passes)}")

    if passes:
        best = max(passes, key=lambda r: r["sharpe"])
        print(f"\nBest: {best.get('notes','')}  "
              f"Sharpe={best['sharpe']:.4f}  MaxDD={best['max_dd']:.2f}%  "
              f"CAGR={best['cagr']:.2f}%  N={best['n_trades']}")

        # update 4h baseline
        bl.update({
            "rsi_filter":    best.get("rsi_filter", False),
            "rsi_threshold": best.get("rsi_threshold"),
            "vol_confirm":   best.get("vol_confirm", False),
            "vol_mult":      best.get("vol_mult"),
            "atr_sizing":    best.get("atr_sizing", False),
            "risk_pct":      best.get("risk_pct"),
            "atr_mult":      best.get("atr_mult"),
            "n_trades":      best["n_trades"],
            "cagr":          best["cagr"],
            "sharpe":        best["sharpe"],
            "max_dd":        best["max_dd"],
            "sortino":       best.get("sortino", 0),
            "calmar":        best.get("calmar", 0),
            "recorded_at":   datetime.now(timezone.utc).date().isoformat(),
            "notes":         f"Tier-2 best: {best.get('notes','')}",
        })
        BASELINE_FILE.write_text(json.dumps(bl, indent=2))
        print("baseline_4h.json updated.")
    else:
        print("No passes — baseline_4h.json unchanged.")


if __name__ == "__main__":
    main()
