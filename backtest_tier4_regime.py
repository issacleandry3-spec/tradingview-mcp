#!/usr/bin/env python3
"""
Tier-4 regime conditioning — rolling volatility percentile filter.

vol_pct = expanding percentile rank of 30-day rolling return std (no lookahead).
  - No new entries when vol_pct > entry_block  (crisis regime)
  - Halve position size when vol_pct > half_size_threshold (0.70, fixed)

Base combos (best from prior tiers):
  SW=50 / LW=200  SMA  — original baseline
  SW=20 / LW=300  SMA  — best Tier-1 MaxDD
  SW=30 / LW=100  SMA  — best Tier-1 Sharpe

Entry-block grid: [0.80, 0.85, 0.90]  (CLAUDE.md specifies 0.85)
=> 3 combos × 3 thresholds = 9 runs

Gate vs baseline (cagr=30.25, sharpe=0.82, max_dd=-64.32):
  Δcagr ≥ +0.5pp  AND  Δsharpe ≥ +0.02  AND  Δmax_dd ≥ +0.5pp
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

ROOT          = Path(__file__).parent
JOURNAL_PATH  = ROOT / "strategy_journal.csv"
BASELINE_PATH = ROOT / "baseline.json"

EXCHANGE_ID     = "coinbase"
SYMBOL          = "BTC/USD"
TIMEFRAME       = "1d"
INITIAL_CAPITAL = 100_000.0
TAKER_FEE       = 0.006
START           = datetime(2018, 1, 1, tzinfo=timezone.utc)
END             = datetime(2026, 9, 17, tzinfo=timezone.utc)

VOL_ROLL_WINDOW   = 30
HALF_SIZE_THRESH  = 0.70          # fixed per CLAUDE.md
ENTRY_BLOCK_GRID  = [0.80, 0.85, 0.90]

BASE_COMBOS = [
    (50, 200, "sma"),   # original baseline
    (20, 300, "sma"),   # best Tier-1 MaxDD / Sharpe
    (30, 100, "sma"),   # best Tier-1 raw Sharpe (0.924 with RSI)
]

JOURNAL_COLS = [
    "run_id","timestamp","short_window","long_window","ma_type",
    "rsi_filter","vol_confirm","atr_sizing","regime_filter",
    "cagr","sharpe","max_dd","sortino","calmar",
    "breakout_window","take_profit","stop_loss","vol_mult","max_hold_bars",
    "gate_passed","notes","strategy","asset",
    "n_trades","win_rate","net_pnl","total_fees",
    "rsi_threshold","rsi_window","win_year_pct","regime","tier",
    "risk_pct","atr_mult","atr_window",
    "vol_roll_window","entry_block","half_size_thresh",
]


def _next_run_id() -> int:
    if not JOURNAL_PATH.exists():
        return 1
    with open(JOURNAL_PATH) as f:
        return sum(1 for _ in f)


def fetch_ohlcv(exchange: ccxt.Exchange) -> pd.DataFrame:
    since_ms = int(START.timestamp() * 1000)
    end_ms   = int(END.timestamp()   * 1000)
    rows: list[list] = []
    print(f"Fetching {SYMBOL} {TIMEFRAME} …", flush=True)
    while since_ms < end_ms:
        batch = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since_ms, limit=300)
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
    print(f"Loaded {len(df)} candles  ({df.index[0].date()} → {df.index[-1].date()})\n")
    return df


def build_features(df: pd.DataFrame, sw: int, lw: int, ma_type: str) -> pd.DataFrame:
    out   = df.copy()
    close = df["close"]
    if ma_type == "ema":
        out["ma_s"] = ta.trend.ema_indicator(close, window=sw)
        out["ma_l"] = ta.trend.ema_indicator(close, window=lw)
    else:
        out["ma_s"] = ta.trend.sma_indicator(close, window=sw)
        out["ma_l"] = ta.trend.sma_indicator(close, window=lw)

    # vol percentile — expanding rank avoids lookahead bias
    ret          = close.pct_change()
    vol_30       = ret.rolling(VOL_ROLL_WINDOW).std()
    # expanding rank: at bar i, rank vol_30[i] against vol_30[:i]
    out["vol_pct"] = vol_30.expanding().rank(pct=True)
    return out


def run_backtest(feat: pd.DataFrame, entry_block: float) -> dict:
    ma_s    = feat["ma_s"].values
    ma_l    = feat["ma_l"].values
    px      = feat["close"].values
    vol_pct = feat["vol_pct"].values
    idx     = feat.index

    cash     = INITIAL_CAPITAL
    btc_held = 0.0
    trades: list[dict] = []
    equity: list[tuple] = []

    for i in range(1, len(feat)):
        price    = float(px[i])
        s_now    = ma_s[i];   s_prev = ma_s[i-1]
        l_now    = ma_l[i];   l_prev = ma_l[i-1]
        vp       = vol_pct[i]

        if np.isnan(s_now) or np.isnan(l_now) or np.isnan(s_prev) or np.isnan(l_prev):
            equity.append((idx[i], cash + btc_held * price))
            continue

        golden = (s_prev <= l_prev) and (s_now > l_now)
        death  = (s_prev >= l_prev) and (s_now < l_now)

        if golden and btc_held == 0:
            # regime gate: skip entry in crisis
            if not np.isnan(vp) and vp > entry_block:
                equity.append((idx[i], cash + btc_held * price))
                continue

            # position sizing: halve in elevated-vol regime
            size_frac = 0.95
            if not np.isnan(vp) and vp > HALF_SIZE_THRESH:
                size_frac = 0.475

            spend = cash * size_frac
            cost  = spend * (1 + TAKER_FEE)
            if cost <= cash:
                btc_held  = spend / price
                cash     -= cost
                trades.append({"side":"BUY","price":price,"qty":btc_held,
                                "vp":vp,"regime":"crisis" if vp>entry_block else
                                        "elevated" if vp>HALF_SIZE_THRESH else "calm"})

        elif death and btc_held > 0:
            proceeds  = btc_held * price * (1 - TAKER_FEE)
            cash     += proceeds
            trades.append({"side":"SELL","price":price,"qty":btc_held})
            btc_held  = 0.0

        equity.append((idx[i], cash + btc_held * price))

    eq = pd.Series({ts: v for ts, v in equity})
    eq.index = pd.DatetimeIndex(eq.index).tz_localize(None)
    returns  = eq.pct_change().dropna()

    if len(returns) < 2 or returns.std() == 0:
        return dict(cagr=0.0, sharpe=0.0, max_dd=0.0, sortino=0.0,
                    calmar=0.0, win_year_pct=0.0, n_trades=0,
                    win_rate=float("nan"), net_pnl=0.0, total_fees=0.0)

    cagr    = float(qs.stats.cagr(returns))         * 100
    sharpe  = float(qs.stats.sharpe(returns))
    max_dd  = float(qs.stats.max_drawdown(returns)) * 100
    sortino = float(qs.stats.sortino(returns))
    calmar  = float(qs.stats.calmar(returns))
    ann     = returns.resample("YE").apply(lambda r: (1+r).prod() - 1)
    win_year_pct = float((ann > 0).mean()) * 100

    open_buy = None; wins = 0; n_complete = 0
    total_fees = net_pnl = 0.0
    for t in trades:
        if t["side"] == "BUY":
            open_buy   = t["price"]
            total_fees += t["price"] * t["qty"] * TAKER_FEE
        elif t["side"] == "SELL" and open_buy is not None:
            total_fees += t["price"] * t["qty"] * TAKER_FEE
            gross       = t["price"] * (1 - TAKER_FEE) - open_buy
            net_pnl    += gross * t["qty"]
            if gross > 0:
                wins += 1
            n_complete += 1
            open_buy = None
    win_rate = wins / n_complete if n_complete else float("nan")

    return dict(cagr=cagr, sharpe=sharpe, max_dd=max_dd, sortino=sortino,
                calmar=calmar, win_year_pct=win_year_pct, n_trades=n_complete,
                win_rate=win_rate, net_pnl=net_pnl, total_fees=total_fees)


def gate_check(s: dict, baseline: dict) -> bool:
    return (
        s["cagr"]   - baseline["cagr"]   >= 0.5  and
        s["sharpe"] - baseline["sharpe"]  >= 0.02 and
        s["max_dd"] - baseline["max_dd"]  >= 0.5
    )


def append_journal(rows: list[dict]) -> None:
    write_header = not JOURNAL_PATH.exists()
    with open(JOURNAL_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    baseline = json.loads(BASELINE_PATH.read_text())
    print(f"Baseline — CAGR={baseline['cagr']:.2f}%  "
          f"Sharpe={baseline['sharpe']:.4f}  MaxDD={baseline['max_dd']:.2f}%")
    print(f"Regime filter: block entries vol_pct > [0.80|0.85|0.90]  "
          f"| halve size vol_pct > {HALF_SIZE_THRESH}\n")

    exchange = ccxt.coinbase({"enableRateLimit": True})
    raw_df   = fetch_ohlcv(exchange)

    run_id       = _next_run_id()
    journal_rows = []
    best         = {"sharpe": -999.0, "row": None}

    hdr = (f"{'SW':>4} {'LW':>4} {'MA':>4} {'Block':>6}  "
           f"{'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8}  "
           f"{'Trades':>7} {'WR%':>6}  Gate")
    print(hdr)
    print("-" * len(hdr))

    for sw, lw, ma_type in BASE_COMBOS:
        feat = build_features(raw_df, sw, lw, ma_type)
        for entry_block in ENTRY_BLOCK_GRID:
            s      = run_backtest(feat, entry_block)
            passed = gate_check(s, baseline)
            wr_str = f"{s['win_rate']*100:6.1f}" if not np.isnan(s["win_rate"]) else "   n/a"

            print(f"{sw:>4} {lw:>4} {ma_type:>4} {entry_block:>6.2f}  "
                  f"{s['cagr']:>8.2f} {s['sharpe']:>8.4f} {s['max_dd']:>8.2f}  "
                  f"{s['n_trades']:>7} {wr_str}  "
                  f"{'PASS ✓' if passed else 'fail'}")

            row = {
                "run_id":           run_id,
                "timestamp":        datetime.now(timezone.utc).isoformat(),
                "short_window":     sw,
                "long_window":      lw,
                "ma_type":          ma_type,
                "rsi_filter":       False,
                "vol_confirm":      False,
                "atr_sizing":       False,
                "regime_filter":    True,
                "cagr":             round(s["cagr"], 4),
                "sharpe":           round(s["sharpe"], 4),
                "max_dd":           round(s["max_dd"], 4),
                "sortino":          round(s["sortino"], 4),
                "calmar":           round(s["calmar"], 4),
                "win_year_pct":     round(s["win_year_pct"], 4),
                "gate_passed":      passed,
                "notes":            f"Tier4 regime block={entry_block} half={HALF_SIZE_THRESH}",
                "strategy":         "btc_sma_crossover",
                "asset":            "BTC/USD",
                "n_trades":         s["n_trades"],
                "win_rate":         round(s["win_rate"], 4) if not np.isnan(s["win_rate"]) else "",
                "net_pnl":          round(s["net_pnl"], 4),
                "total_fees":       round(s["total_fees"], 4),
                "vol_roll_window":  VOL_ROLL_WINDOW,
                "entry_block":      entry_block,
                "half_size_thresh": HALF_SIZE_THRESH,
                "tier":             4,
            }
            journal_rows.append(row)
            run_id += 1
            if s["sharpe"] > best["sharpe"]:
                best = {"sharpe": s["sharpe"], "row": row, "stats": s}

    print()
    append_journal(journal_rows)
    print(f"Appended {len(journal_rows)} rows → {JOURNAL_PATH}")

    passes = [r for r in journal_rows if r["gate_passed"]]
    print(f"\nGate passes: {len(passes)} / {len(journal_rows)}")

    if passes:
        bp = max(passes, key=lambda r: r["sharpe"])
        print(f"\nBest gate-pass: {bp['ma_type'].upper()} {bp['short_window']}/{bp['long_window']}  "
              f"block={bp['entry_block']}")
        print(f"  CAGR={bp['cagr']:.2f}%  Sharpe={bp['sharpe']:.4f}  "
              f"MaxDD={bp['max_dd']:.2f}%  Trades={bp['n_trades']}")
        print(f"  Δcagr={bp['cagr']-baseline['cagr']:+.2f}pp  "
              f"Δsharpe={bp['sharpe']-baseline['sharpe']:+.4f}  "
              f"Δmax_dd={bp['max_dd']-baseline['max_dd']:+.2f}pp")

        baseline.update({
            "short_window":     bp["short_window"],
            "long_window":      bp["long_window"],
            "ma_type":          bp["ma_type"],
            "rsi_filter":       False,
            "vol_confirm":      False,
            "atr_sizing":       False,
            "regime_filter":    True,
            "vol_roll_window":  VOL_ROLL_WINDOW,
            "entry_block":      bp["entry_block"],
            "half_size_thresh": HALF_SIZE_THRESH,
            "cagr":             bp["cagr"],
            "sharpe":           bp["sharpe"],
            "max_dd":           bp["max_dd"],
            "sortino":          bp["sortino"],
            "calmar":           bp["calmar"],
            "win_year_pct":     bp["win_year_pct"],
            "recorded_at":      datetime.now(timezone.utc).date().isoformat(),
            "notes":            (f"Tier4 regime {bp['ma_type'].upper()} "
                                 f"{bp['short_window']}/{bp['long_window']} "
                                 f"block={bp['entry_block']}"),
        })
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
        print(f"\nbaseline.json updated → Sharpe {baseline['sharpe']:.4f}  "
              f"CAGR {baseline['cagr']:.2f}%  MaxDD {baseline['max_dd']:.2f}%")

        if len(passes) > 1:
            print(f"\nAll {len(passes)} gate-passes:")
            for r in sorted(passes, key=lambda r: r["sharpe"], reverse=True):
                print(f"  {r['ma_type'].upper()} {r['short_window']:>3}/{r['long_window']:>3} "
                      f"block={r['entry_block']}  "
                      f"CAGR={r['cagr']:.2f}%  Sharpe={r['sharpe']:.4f}  "
                      f"MaxDD={r['max_dd']:.2f}%")
    else:
        br = best["row"]
        print(f"\nNo gate passes. Best Sharpe: {br['sharpe']:.4f}  "
              f"[{br['ma_type'].upper()} {br['short_window']}/{br['long_window']} "
              f"block={br['entry_block']}]")
        print(f"  Δsharpe={br['sharpe']-baseline['sharpe']:+.4f}  "
              f"Δcagr={br['cagr']-baseline['cagr']:+.2f}pp  "
              f"Δmax_dd={br['max_dd']-baseline['max_dd']:+.2f}pp")
        print("baseline.json unchanged.")


if __name__ == "__main__":
    main()
