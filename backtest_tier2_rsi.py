#!/usr/bin/env python3
"""
Tier-2 RSI filter — apply rsi<threshold entry gate to top-3 Tier-1 combos.
Top-3 (by Sharpe from Tier-1 auto-loop):
  SW=20 / LW=300  Sharpe=0.80  CAGR=25.89  MaxDD=-52.86
  SW=75 / LW=300  Sharpe=0.73  CAGR=21.98  MaxDD=-52.78
  SW=30 / LW=100  Sharpe=0.73  CAGR=21.59  MaxDD=-53.59

Grid: RSI window=14, thresholds [55, 60, 65, 70]  => 12 runs.
Gate vs baseline (30.25 / 0.82 / -64.32):
  Δcagr ≥ +0.5pp  AND  Δsharpe ≥ +0.02  AND  Δmax_dd ≥ +0.5pp (less negative)
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import quantstats as qs
import ta

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
JOURNAL_PATH  = ROOT / "strategy_journal.csv"
BASELINE_PATH = ROOT / "baseline.json"

# ── exchange / data config ────────────────────────────────────────────────────
EXCHANGE_ID      = "coinbase"
SYMBOL           = "BTC/USD"
TIMEFRAME        = "1d"
INITIAL_CAPITAL  = 100_000.0
TAKER_FEE        = 0.006
START            = datetime(2018, 1, 1, tzinfo=timezone.utc)
END              = datetime(2026, 9, 17, tzinfo=timezone.utc)

# ── Tier-2 grid ───────────────────────────────────────────────────────────────
TOP3_COMBOS = [
    (20, 300),
    (75, 300),
    (30, 100),
]
RSI_WINDOW      = 14
RSI_THRESHOLDS  = [55, 60, 65, 70]

# ── journal columns (match existing CSV header) ───────────────────────────────
JOURNAL_COLS = [
    "run_id","timestamp","short_window","long_window","ma_type",
    "rsi_filter","vol_confirm","atr_sizing","regime_filter",
    "cagr","sharpe","max_dd","sortino","calmar",
    "breakout_window","take_profit","stop_loss","vol_mult","max_hold_bars",
    "gate_passed","notes","strategy","asset",
    "n_trades","win_rate","net_pnl","total_fees",
    "rsi_threshold","rsi_window",
    "win_year_pct","regime","tier",
]


def _next_run_id() -> int:
    if not JOURNAL_PATH.exists():
        return 1
    with open(JOURNAL_PATH) as f:
        rows = list(csv.reader(f))
    return len(rows)   # header + data rows = next id


def fetch_ohlcv(exchange: ccxt.Exchange) -> pd.DataFrame:
    since_ms = int(START.timestamp() * 1000)
    end_ms   = int(END.timestamp()   * 1000)
    limit    = 300
    rows: list[list] = []
    print(f"Fetching {SYMBOL} {TIMEFRAME} from {EXCHANGE_ID} …", flush=True)
    while since_ms < end_ms:
        batch = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since_ms, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        since_ms = batch[-1][0] + 1
        print(f"  up to {datetime.fromtimestamp(batch[-1][0]/1000, tz=timezone.utc).date()}", end="\r")
        time.sleep(exchange.rateLimit / 1000)
    print()
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[df.index < pd.Timestamp(END)]
    df = df[~df.index.duplicated(keep="last")]
    print(f"Loaded {len(df)} candles  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def build_features(df: pd.DataFrame, sw: int, lw: int) -> pd.DataFrame:
    out = df.copy()
    close = df["close"]
    out["sma_s"]  = ta.trend.sma_indicator(close, window=sw)
    out["sma_l"]  = ta.trend.sma_indicator(close, window=lw)
    out["rsi"]    = ta.momentum.rsi(close, window=RSI_WINDOW)
    return out


def run_backtest(feat: pd.DataFrame, sw: int, lw: int,
                 rsi_thresh: float) -> tuple[pd.Series, list[dict], dict]:
    """Bar-by-bar simulation with RSI entry gate. Returns (returns, trades, stats)."""
    close  = feat["close"].values
    sma_s  = feat["sma_s"].values
    sma_l  = feat["sma_l"].values
    rsi    = feat["rsi"].values
    idx    = feat.index

    cash      = INITIAL_CAPITAL
    btc_held  = 0.0
    trades: list[dict] = []
    equity: list[tuple] = []

    for i in range(1, len(feat)):
        price   = float(close[i])
        s_now   = sma_s[i];   s_prev = sma_s[i-1]
        l_now   = sma_l[i];   l_prev = sma_l[i-1]
        rsi_now = rsi[i]

        if np.isnan(s_now) or np.isnan(l_now) or np.isnan(s_prev) or np.isnan(l_prev):
            equity.append((idx[i], cash + btc_held * price))
            continue

        golden = (s_prev <= l_prev) and (s_now > l_now)
        death  = (s_prev >= l_prev) and (s_now < l_now)

        if golden and btc_held == 0 and (not np.isnan(rsi_now)) and rsi_now < rsi_thresh:
            spend    = cash * 0.95
            cost     = spend * (1 + TAKER_FEE)
            if cost <= cash:
                btc_held  = spend / price
                cash     -= cost
                trades.append({"side":"BUY","price":price,"date":idx[i].date()})

        elif death and btc_held > 0:
            proceeds  = btc_held * price * (1 - TAKER_FEE)
            cash     += proceeds
            trades.append({"side":"SELL","price":price,"date":idx[i].date()})
            btc_held  = 0.0

        equity.append((idx[i], cash + btc_held * price))

    eq = pd.Series({ts: v for ts, v in equity})
    eq.index = pd.DatetimeIndex(eq.index).tz_localize(None)
    returns = eq.pct_change().dropna()

    # ── metrics ───────────────────────────────────────────────────────────────
    cagr    = float(qs.stats.cagr(returns))   * 100
    sharpe  = float(qs.stats.sharpe(returns))
    max_dd  = float(qs.stats.max_drawdown(returns)) * 100
    sortino = float(qs.stats.sortino(returns))
    calmar  = float(qs.stats.calmar(returns))

    # win-year: fraction of calendar years with positive return
    ann = returns.resample("YE").apply(lambda r: (1+r).prod() - 1)
    win_year_pct = float((ann > 0).mean()) * 100

    # per-trade stats
    buy_prices: list[float] = []
    wins = 0; n_complete = 0; total_fees = 0.0; net_pnl = 0.0
    open_buy = None
    for t in trades:
        if t["side"] == "BUY":
            open_buy = t["price"]
            total_fees += open_buy * 0.95 * TAKER_FEE
        elif t["side"] == "SELL" and open_buy is not None:
            sell_val   = t["price"] * (1 - TAKER_FEE)
            total_fees += t["price"] * TAKER_FEE
            if sell_val > open_buy:
                wins += 1
            net_pnl  += sell_val - open_buy
            n_complete += 1
            open_buy = None
    win_rate = wins / n_complete if n_complete else float("nan")

    stats = dict(
        cagr=cagr, sharpe=sharpe, max_dd=max_dd,
        sortino=sortino, calmar=calmar, win_year_pct=win_year_pct,
        n_trades=n_complete, win_rate=win_rate,
        net_pnl=net_pnl, total_fees=total_fees,
    )
    return returns, trades, stats


def gate_check(stats: dict, baseline: dict) -> bool:
    ok_cagr   = stats["cagr"]   - baseline["cagr"]   >= 0.5
    ok_sharpe = stats["sharpe"] - baseline["sharpe"]  >= 0.02
    ok_dd     = stats["max_dd"] - baseline["max_dd"]  >= 0.5   # less negative ↑
    return ok_cagr and ok_sharpe and ok_dd


def append_journal(rows: list[dict]) -> None:
    write_header = not JOURNAL_PATH.exists()
    with open(JOURNAL_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_COLS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    baseline = json.loads(BASELINE_PATH.read_text())
    print(f"Baseline — CAGR={baseline['cagr']:.2f}%  Sharpe={baseline['sharpe']:.4f}  MaxDD={baseline['max_dd']:.2f}%\n")

    exchange = ccxt.coinbase({"enableRateLimit": True})
    raw_df   = fetch_ohlcv(exchange)

    run_id    = _next_run_id()
    journal_rows: list[dict] = []
    best_by_sharpe = {"sharpe": -999.0, "row": None}

    header = (
        f"{'SW':>4} {'LW':>4} {'RSI_T':>6}  "
        f"{'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8}  "
        f"{'Trades':>7} {'WR%':>6}  Gate"
    )
    print(header)
    print("-" * len(header))

    for sw, lw in TOP3_COMBOS:
        feat = build_features(raw_df, sw, lw)
        for rsi_t in RSI_THRESHOLDS:
            _, trades, s = run_backtest(feat, sw, lw, rsi_t)
            passed = gate_check(s, baseline)

            print(
                f"{sw:>4} {lw:>4} {rsi_t:>6}  "
                f"{s['cagr']:>8.2f} {s['sharpe']:>8.4f} {s['max_dd']:>8.2f}  "
                f"{s['n_trades']:>7} {s['win_rate']*100:>6.1f}  "
                f"{'PASS ✓' if passed else 'fail'}"
            )

            row = {
                "run_id":        run_id,
                "timestamp":     datetime.utcnow().isoformat(),
                "short_window":  sw,
                "long_window":   lw,
                "ma_type":       "sma",
                "rsi_filter":    True,
                "vol_confirm":   False,
                "atr_sizing":    False,
                "regime_filter": False,
                "cagr":          round(s["cagr"], 4),
                "sharpe":        round(s["sharpe"], 4),
                "max_dd":        round(s["max_dd"], 4),
                "sortino":       round(s["sortino"], 4),
                "calmar":        round(s["calmar"], 4),
                "win_year_pct":  round(s["win_year_pct"], 4),
                "gate_passed":   passed,
                "notes":         f"Tier2 RSI filter rsi<{rsi_t}",
                "strategy":      "btc_sma_crossover",
                "asset":         "BTC/USD",
                "n_trades":      s["n_trades"],
                "win_rate":      round(s["win_rate"], 4) if not np.isnan(s["win_rate"]) else "",
                "net_pnl":       round(s["net_pnl"], 4),
                "total_fees":    round(s["total_fees"], 4),
                "rsi_threshold": rsi_t,
                "rsi_window":    RSI_WINDOW,
                "tier":          2,
            }
            journal_rows.append(row)
            run_id += 1

            if s["sharpe"] > best_by_sharpe["sharpe"]:
                best_by_sharpe = {"sharpe": s["sharpe"], "row": row, "stats": s}

    print()
    append_journal(journal_rows)
    print(f"Appended {len(journal_rows)} rows to {JOURNAL_PATH}")

    # ── gate summary ──────────────────────────────────────────────────────────
    passes = [r for r in journal_rows if r["gate_passed"]]
    print(f"\nGate passes: {len(passes)} / {len(journal_rows)}")

    if passes:
        best_pass = max(passes, key=lambda r: r["sharpe"])
        print(f"\nBest gate-pass: SW={best_pass['short_window']} LW={best_pass['long_window']}"
              f" RSI<{best_pass['rsi_threshold']}"
              f"  CAGR={best_pass['cagr']:.2f}%  Sharpe={best_pass['sharpe']:.4f}"
              f"  MaxDD={best_pass['max_dd']:.2f}%")

        # update baseline
        baseline.update({
            "short_window":  best_pass["short_window"],
            "long_window":   best_pass["long_window"],
            "ma_type":       "sma",
            "rsi_filter":    True,
            "rsi_threshold": best_pass["rsi_threshold"],
            "rsi_window":    RSI_WINDOW,
            "vol_confirm":   False,
            "atr_sizing":    False,
            "regime_filter": False,
            "cagr":          best_pass["cagr"],
            "sharpe":        best_pass["sharpe"],
            "max_dd":        best_pass["max_dd"],
            "sortino":       best_pass["sortino"],
            "calmar":        best_pass["calmar"],
            "win_year_pct":  best_pass["win_year_pct"],
            "recorded_at":   datetime.utcnow().date().isoformat(),
            "notes":         f"Tier2 RSI<{best_pass['rsi_threshold']} on SW{best_pass['short_window']}/LW{best_pass['long_window']}",
        })
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
        print(f"baseline.json updated  →  Sharpe {baseline['sharpe']:.4f}  CAGR {baseline['cagr']:.2f}%")
    else:
        # report best overall even without gate pass
        br = best_by_sharpe["row"]
        print(f"\nNo gate passes. Best Tier-2 Sharpe: {br['sharpe']:.4f}"
              f" (SW={br['short_window']} LW={br['long_window']} RSI<{br['rsi_threshold']})")
        print(f"Δsharpe vs baseline: {br['sharpe'] - baseline['sharpe']:+.4f}"
              f"  Δcagr: {br['cagr'] - baseline['cagr']:+.2f}pp"
              f"  Δmax_dd: {br['max_dd'] - baseline['max_dd']:+.2f}pp")
        print("baseline.json unchanged.")


if __name__ == "__main__":
    main()
