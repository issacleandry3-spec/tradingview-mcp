#!/usr/bin/env python3
"""
Run the Tier-2 best config (SMA 10/200 + ATR 1%/2.0×) across ETH, SOL, LINK.
Position size = min(ATR-based, MAX_POS_PCT × equity) to cap compounding blow-up.
Compares each asset against the BTC baseline and reports cross-asset consistency.
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
BASELINE_FILE = ROOT / "baseline_4h.json"
JOURNAL_FILE  = ROOT / "strategy_journal.csv"

FEE         = 0.006
CAPITAL     = 100_000.0
SW, LW      = 10, 200
ATR_RISK    = 0.01
ATR_MULT    = 2.0
MAX_POS_PCT = 0.25    # max 25% of current equity per position

ASSETS = {
    "BTC": ROOT / "data" / "4h-regular-COINBASE-BTCUSD.csv",
    "ETH": ROOT / "data" / "4h-regular-COINBASE-ETHUSD.csv",
    "SOL": ROOT / "data" / "4h-regular-COINBASE-SOLUSD.csv",
    "LINK": ROOT / "data" / "4h-regular-COINBASE-LINKUSD.csv",
}


def backtest(df: pd.DataFrame) -> dict:
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    times = df["time"].values
    n     = len(close)

    sma_s = pd.Series(close).rolling(SW).mean().values
    sma_l = pd.Series(close).rolling(LW).mean().values
    atr_v = ta.volatility.average_true_range(
                pd.Series(high), pd.Series(low), pd.Series(close), window=14).values

    cash = CAPITAL; qty = 0.0; in_pos = False
    equity: list[float] = []
    n_trades = 0
    trade_rets: list[float] = []
    entry_price = 0.0

    for i in range(LW, n):
        golden = (not (sma_s[i-1] > sma_l[i-1])) and (sma_s[i] > sma_l[i])
        death  = (sma_s[i-1] > sma_l[i-1]) and (not (sma_s[i] > sma_l[i]))

        if not in_pos and golden:
            atr      = atr_v[i]
            equity_now = cash                          # qty == 0 at entry
            cap_qty  = (equity_now * MAX_POS_PCT) / (close[i] * (1 + FEE/2))
            if not np.isnan(atr) and atr > 0:
                atr_qty = (equity_now * ATR_RISK) / (atr * ATR_MULT)
                qty = min(atr_qty, cap_qty)
            else:
                qty = cap_qty
            cash -= qty * close[i] * (1 + FEE/2)
            entry_price = close[i]
            in_pos = True
            n_trades += 1

        elif in_pos and death:
            proceeds = qty * close[i] * (1 - FEE/2)
            trade_rets.append((close[i] - entry_price) / entry_price)
            cash += proceeds
            qty = 0.0
            in_pos = False

        equity.append(cash + qty * close[i])

    if in_pos:
        cash += qty * close[-1] * (1 - FEE/2)

    if n_trades == 0 or not equity:
        return {}

    eq  = pd.Series(equity, index=pd.to_datetime(times[LW:], unit="s", utc=True))
    ret = eq.pct_change().dropna()

    win_trades = sum(1 for r in trade_rets if r > 0)
    win_rate   = win_trades / len(trade_rets) if trade_rets else 0.0
    avg_win    = float(np.mean([r for r in trade_rets if r > 0])) if any(r > 0 for r in trade_rets) else 0.0
    avg_loss   = float(np.mean([r for r in trade_rets if r < 0])) if any(r < 0 for r in trade_rets) else 0.0

    years = (times[-1] - times[LW]) / (365.25 * 86_400)

    return dict(
        n_trades   = n_trades,
        years      = round(years, 2),
        trades_yr  = round(n_trades / max(years, 1), 1),
        cagr       = round(float(qs.stats.cagr(ret))          * 100, 4),
        sharpe     = round(float(qs.stats.sharpe(ret)),               4),
        max_dd     = round(float(qs.stats.max_drawdown(ret))  * 100, 4),
        sortino    = round(float(qs.stats.sortino(ret)),              4),
        calmar     = round(float(qs.stats.calmar(ret)),               4),
        win_rate   = round(win_rate * 100, 1),
        avg_win_pct= round(avg_win  * 100, 2),
        avg_loss_pct=round(avg_loss * 100, 2),
        final_equity=round(cash, 2),
    )


JOURNAL_COLS = [
    "strategy","timeframe","asset","ma_type","short_window","long_window",
    "atr_sizing","risk_pct","atr_mult",
    "n_trades","years","trades_yr","cagr","sharpe","max_dd","sortino","calmar",
    "win_rate","avg_win_pct","avg_loss_pct","final_equity",
    "recorded_at","notes",
]

def write_row(row: dict) -> None:
    exists = JOURNAL_FILE.exists()
    if exists:
        existing = pd.read_csv(JOURNAL_FILE, nrows=0).columns.tolist()
        cols = list(dict.fromkeys(existing + JOURNAL_COLS))
    else:
        cols = JOURNAL_COLS
    with open(JOURNAL_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    bl = json.loads(BASELINE_FILE.read_text())
    print(f"Strategy: SMA {SW}/{LW} + ATR {ATR_RISK*100:.0f}%/{ATR_MULT}×  (4h Coinbase)")
    print(f"BTC baseline: Sharpe={bl['sharpe']:.4f}  MaxDD={bl['max_dd']:.2f}%  N={bl['n_trades']}\n")

    results = {}

    for asset, path in ASSETS.items():
        if not path.exists():
            print(f"  {asset}: file not found ({path.name})")
            continue
        df = pd.read_csv(path)
        start = pd.Timestamp(df["time"].iloc[0],  unit="s").date()
        end   = pd.Timestamp(df["time"].iloc[-1], unit="s").date()
        print(f"  {asset:<5} {len(df):>6} bars  {start} → {end} ...", end=" ", flush=True)

        r = backtest(df)
        if not r:
            print("no trades")
            continue

        results[asset] = r
        print(f"N={r['n_trades']:>3}  "
              f"Sharpe={r['sharpe']:>7.4f}  "
              f"MaxDD={r['max_dd']:>7.2f}%  "
              f"CAGR={r['cagr']:>7.2f}%  "
              f"WR={r['win_rate']:>5.1f}%")

        write_row(dict(
            strategy="4h_sma10_200_atr_multiasset",
            timeframe="4h", asset=asset,
            ma_type="sma", short_window=SW, long_window=LW,
            atr_sizing=True, risk_pct=ATR_RISK, atr_mult=ATR_MULT,
            recorded_at=datetime.now(timezone.utc).date().isoformat(),
            notes=f"4h SMA10/200 ATR1%/2x on {asset}",
            **r,
        ))

    # ── cross-asset summary table ─────────────────────────────────────────────
    if len(results) < 2:
        return

    print(f"\n{'═'*85}")
    print(f"  {'Asset':<6} {'N':>4} {'Yrs':>4} {'N/yr':>5}  "
          f"{'Sharpe':>7}  {'MaxDD%':>8}  {'CAGR%':>7}  {'WR%':>6}  "
          f"{'AvgW%':>6}  {'AvgL%':>7}  {'Final $':>10}")
    print(f"  {'─'*6} {'─'*4} {'─'*4} {'─'*5}  "
          f"{'─'*7}  {'─'*8}  {'─'*7}  {'─'*6}  "
          f"{'─'*6}  {'─'*7}  {'─'*10}")

    for asset, r in results.items():
        flag = " ← BTC baseline" if asset == "BTC" else ""
        print(f"  {asset:<6} {r['n_trades']:>4} {r['years']:>4.1f} {r['trades_yr']:>5.1f}  "
              f"{r['sharpe']:>7.4f}  {r['max_dd']:>8.2f}  {r['cagr']:>7.2f}  "
              f"{r['win_rate']:>6.1f}  {r['avg_win_pct']:>6.2f}  "
              f"{r['avg_loss_pct']:>7.2f}  {r['final_equity']:>10,.0f}{flag}")

    # ── consistency metrics ───────────────────────────────────────────────────
    sharpes  = [r["sharpe"]  for r in results.values()]
    max_dds  = [r["max_dd"]  for r in results.values()]
    cagrs    = [r["cagr"]    for r in results.values()]
    n_pos_sharpe = sum(1 for s in sharpes if s > 0)
    n_pos_cagr   = sum(1 for c in cagrs   if c > 0)

    print(f"\n  Positive Sharpe: {n_pos_sharpe}/{len(results)}  |  "
          f"Positive CAGR: {n_pos_cagr}/{len(results)}")
    print(f"  Sharpe  avg={np.mean(sharpes):.4f}  min={min(sharpes):.4f}  max={max(sharpes):.4f}")
    print(f"  MaxDD   avg={np.mean(max_dds):.2f}%  worst={min(max_dds):.2f}%  best={max(max_dds):.2f}%")
    print(f"  CAGR    avg={np.mean(cagrs):.2f}%   min={min(cagrs):.2f}%   max={max(cagrs):.2f}%")


if __name__ == "__main__":
    main()
