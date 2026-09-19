#!/usr/bin/env python3
"""
Tier-3 on 4h SMA 10/200 + ATR baseline.
3A — EMA vs SMA on nearby window grid (SW 7-20, LW 100-250) with ATR sizing
3B — Regime filter (expanding vol-percentile blocks / halves entries)
Gate: N≥100 AND Δsharpe≥+0.02 AND Δmax_dd≥+0.5pp vs baseline_4h.json
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
import ta

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).parent
DATA_FILE     = ROOT / "data" / "4h-regular-COINBASE-BTCUSD.csv"
BASELINE_FILE = ROOT / "baseline_4h.json"
JOURNAL_FILE  = ROOT / "strategy_journal.csv"

FEE     = 0.006
CAPITAL = 100_000.0

# fixed ATR params from Tier-2 best
ATR_RISK  = 0.01
ATR_MULT  = 2.0
DEPLOY    = 0.95   # fallback cap

MIN_TRADES   = 100
DELTA_SHARPE = 0.02
DELTA_DD     = 0.5


# ── helpers ───────────────────────────────────────────────────────────────────

def calc_metrics(equity: np.ndarray, times: np.ndarray) -> dict:
    eq  = pd.Series(equity, index=pd.to_datetime(times, unit="s", utc=True))
    ret = eq.pct_change().dropna()
    if ret.empty:
        return {}
    return dict(
        cagr    = round(float(qs.stats.cagr(ret))         * 100, 4),
        sharpe  = round(float(qs.stats.sharpe(ret)),              4),
        max_dd  = round(float(qs.stats.max_drawdown(ret)) * 100, 4),
        sortino = round(float(qs.stats.sortino(ret)),             4),
        calmar  = round(float(qs.stats.calmar(ret)),              4),
    )


def ma(series: pd.Series, w: int, kind: str) -> np.ndarray:
    if kind == "ema":
        return series.ewm(span=w, adjust=False).mean().values
    return series.rolling(w).mean().values


def atr_qty(cash: float, atr: float, price: float) -> float:
    if np.isnan(atr) or atr <= 0:
        return (cash * DEPLOY) / (price * (1 + FEE / 2))
    return min(
        (cash * ATR_RISK) / (atr * ATR_MULT),
        (cash * DEPLOY)   / (price * (1 + FEE / 2)),
    )


# ── backtest engine ───────────────────────────────────────────────────────────

def backtest(
    df: pd.DataFrame,
    sw: int, lw: int, kind: str = "sma",
    regime: bool = False,
    entry_block: float = 0.85,
    half_size_thresh: float = 0.70,
) -> dict:
    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    times  = df["time"].values
    n      = len(close)

    sma_s = ma(pd.Series(close), sw, kind)
    sma_l = ma(pd.Series(close), lw, kind)
    atr_v = ta.volatility.average_true_range(
                pd.Series(high), pd.Series(low), pd.Series(close), window=14).values

    vol_pct: np.ndarray | None = None
    if regime:
        vol30 = pd.Series(close).pct_change().rolling(30).std().values
        vol_pct = pd.Series(vol30).expanding().rank(pct=True).values

    cash = CAPITAL; qty = 0.0; in_pos = False
    equity: list[float] = []; n_trades = 0
    start = max(lw, 30)

    for i in range(start, n):
        golden = (not (sma_s[i-1] > sma_l[i-1])) and (sma_s[i] > sma_l[i])
        death  = (sma_s[i-1] > sma_l[i-1]) and (not (sma_s[i] > sma_l[i]))

        if not in_pos and golden:
            vp = vol_pct[i] if vol_pct is not None else 0.0
            if regime and not np.isnan(vp) and vp >= entry_block:
                pass                        # block entry in crisis regime
            else:
                base_qty = atr_qty(cash, atr_v[i], close[i])
                if regime and not np.isnan(vp) and vp >= half_size_thresh:
                    base_qty *= 0.5         # half size in elevated vol
                qty   = base_qty
                cash -= qty * close[i] * (1 + FEE / 2)
                in_pos = True
                n_trades += 1

        elif in_pos and death:
            cash  += qty * close[i] * (1 - FEE / 2)
            qty    = 0.0
            in_pos = False

        equity.append(cash + qty * close[i])

    if in_pos:
        cash += qty * close[-1] * (1 - FEE / 2)

    if n_trades == 0:
        return {}
    m = calc_metrics(np.array(equity), times[start:])
    return dict(n_trades=n_trades, sw=sw, lw=lw, kind=kind, **m) if m else {}


# ── reporting ─────────────────────────────────────────────────────────────────

JOURNAL_COLS = [
    "strategy","timeframe","ma_type","short_window","long_window","tier",
    "atr_sizing","risk_pct","atr_mult",
    "regime_filter","entry_block","half_size_thresh",
    "n_trades","cagr","sharpe","max_dd","sortino","calmar",
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


def report(label: str, bl: dict, r: dict, extra: dict) -> dict | None:
    if not r:
        print(f"  {'·':6} {label:<48}  no trades")
        return None
    n_ok  = r["n_trades"] >= MIN_TRADES
    sh_ok = r["sharpe"]  - bl["sharpe"]  >= DELTA_SHARPE
    dd_ok = r["max_dd"]  - bl["max_dd"]  >= DELTA_DD
    gate  = n_ok and sh_ok and dd_ok
    mark  = "✓ PASS" if gate else "·"
    why   = "" if gate else (
        " [N<100]"  if not n_ok  else
        " [Sh]"     if not sh_ok else " [DD]")
    print(f"  {mark:<6} {label:<48} "
          f"N={r['n_trades']:>3}  "
          f"Sharpe={r['sharpe']:>7.4f} (Δ{r['sharpe']-bl['sharpe']:+.4f})  "
          f"MaxDD={r['max_dd']:>7.2f}% (Δ{r['max_dd']-bl['max_dd']:+.2f}pp)  "
          f"CAGR={r['cagr']:>7.2f}%{why}")
    row = dict(
        strategy="btc_4h_tier3",
        timeframe="4h", tier=3,
        atr_sizing=True, risk_pct=ATR_RISK, atr_mult=ATR_MULT,
        recorded_at=datetime.now(timezone.utc).date().isoformat(),
        **r, **extra,
    )
    row["ma_type"]      = r.get("kind", "sma")
    row["short_window"] = r.get("sw")
    row["long_window"]  = r.get("lw")
    write_row(row)
    return row if gate else None


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(DATA_FILE)
    bl = json.loads(BASELINE_FILE.read_text())

    print(f"Baseline: SMA 10/200 + ATR {ATR_RISK*100:.0f}%/{ATR_MULT}×  "
          f"Sharpe={bl['sharpe']:.4f}  MaxDD={bl['max_dd']:.2f}%  N={bl['n_trades']}")
    print(f"Data: {len(df)} bars  "
          f"{pd.Timestamp(df['time'].iloc[0],  unit='s').date()} → "
          f"{pd.Timestamp(df['time'].iloc[-1], unit='s').date()}\n")

    passes = []

    # ── 3A: MA type × window grid ─────────────────────────────────────────────
    print("── Tier-3A: EMA vs SMA window grid (+ ATR sizing) ───────────────────")
    short_wins = [7, 8, 10, 12, 15, 20]
    long_wins  = [100, 150, 175, 200, 250]
    for kind, sw, lw in product(["sma", "ema"], short_wins, long_wins):
        if sw >= lw:
            continue
        r = backtest(df, sw, lw, kind)
        p = report(f"{kind.upper()} {sw}/{lw}", bl, r,
                   {"notes": f"4h {kind.upper()} {sw}/{lw} ATR"})
        if p:
            passes.append(p)

    # ── 3B: regime filter on baseline window ──────────────────────────────────
    print("\n── Tier-3B: Regime (vol-percentile) filter on SMA 10/200 ────────────")
    for eb, hs in [(0.85, 0.70), (0.90, 0.75), (0.80, 0.65), (0.95, 0.80)]:
        r = backtest(df, 10, 200, "sma", regime=True, entry_block=eb, half_size_thresh=hs)
        p = report(f"regime block={eb} half={hs}", bl, r,
                   {"notes": f"4h SMA10/200 ATR regime eb={eb} hs={hs}",
                    "regime_filter": True, "entry_block": eb, "half_size_thresh": hs})
        if p:
            passes.append(p)

    # ── 3C: regime on best-window EMA/SMA found in 3A ────────────────────────
    # (only run if 3A produced passes)
    if passes:
        top = max(passes, key=lambda x: x["sharpe"])
        tsw, tlw, tkind = top.get("sw", 10), top.get("lw", 200), top.get("kind", "sma")
        if (tsw, tlw, tkind) != (10, 200, "sma"):
            print(f"\n── Tier-3C: Regime filter on best 3A combo "
                  f"({tkind.upper()} {tsw}/{tlw}) ───────")
            for eb, hs in [(0.85, 0.70), (0.90, 0.75)]:
                r = backtest(df, tsw, tlw, tkind,
                             regime=True, entry_block=eb, half_size_thresh=hs)
                p = report(f"{tkind.upper()} {tsw}/{tlw} regime eb={eb}", bl, r,
                           {"notes": f"4h {tkind.upper()} {tsw}/{tlw} ATR regime eb={eb}",
                            "regime_filter": True, "entry_block": eb,
                            "half_size_thresh": hs})
                if p:
                    passes.append(p)

    print(f"\n{'─'*80}")
    print(f"Gate passes: {len(passes)}")

    if passes:
        best = max(passes, key=lambda r: r["sharpe"])
        print(f"\nBest: {best.get('notes','')}  "
              f"Sharpe={best['sharpe']:.4f}  MaxDD={best['max_dd']:.2f}%  "
              f"CAGR={best['cagr']:.2f}%  N={best['n_trades']}")

        prev_sharpe = bl["sharpe"]
        bl.update({
            "short_window":      best.get("sw", bl["short_window"]),
            "long_window":       best.get("lw", bl["long_window"]),
            "ma_type":           best.get("kind", bl["ma_type"]),
            "regime_filter":     best.get("regime_filter", False),
            "entry_block":       best.get("entry_block"),
            "half_size_thresh":  best.get("half_size_thresh"),
            "n_trades":          best["n_trades"],
            "cagr":              best["cagr"],
            "sharpe":            best["sharpe"],
            "max_dd":            best["max_dd"],
            "sortino":           best.get("sortino", 0),
            "calmar":            best.get("calmar", 0),
            "recorded_at":       datetime.now(timezone.utc).date().isoformat(),
            "notes":             f"Tier-3 best: {best.get('notes','')}",
        })
        BASELINE_FILE.write_text(json.dumps(bl, indent=2))
        print(f"baseline_4h.json updated  "
              f"(Sharpe {prev_sharpe:.4f} → {best['sharpe']:.4f})")
    else:
        print("No passes — baseline_4h.json unchanged.")


if __name__ == "__main__":
    main()
