#!/usr/bin/env python3
"""
Tier-2 ATR position sizing — SW=50 / LW=200 on BTC/USD.
Position size: qty = (cash * risk_pct) / (atr * atr_mult)
  risk_pct  in {0.005, 0.01, 0.02}
  atr_mult  in {1.5, 2.0, 2.5}
  => 9 runs

Same entry/exit signals as baseline (golden/death cross), only position
sizing changes. Capped at 95% of cash to avoid over-leverage.

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

SW, LW          = 50, 200
ATR_WINDOW      = 14
RISK_PCTS       = [0.005, 0.01, 0.02]
ATR_MULTS       = [1.5, 2.0, 2.5]

JOURNAL_COLS = [
    "run_id","timestamp","short_window","long_window","ma_type",
    "rsi_filter","vol_confirm","atr_sizing","regime_filter",
    "cagr","sharpe","max_dd","sortino","calmar",
    "breakout_window","take_profit","stop_loss","vol_mult","max_hold_bars",
    "gate_passed","notes","strategy","asset",
    "n_trades","win_rate","net_pnl","total_fees",
    "rsi_threshold","rsi_window","win_year_pct","regime","tier",
    "risk_pct","atr_mult","atr_window",
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
    print(f"Loaded {len(df)} candles  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = df["close"]
    out["sma_s"] = ta.trend.sma_indicator(close, window=SW)
    out["sma_l"] = ta.trend.sma_indicator(close, window=LW)
    out["atr"]   = ta.volatility.average_true_range(
        df["high"], df["low"], close, window=ATR_WINDOW
    )
    return out


def run_backtest(feat: pd.DataFrame, risk_pct: float,
                 atr_mult: float) -> tuple[pd.Series, dict]:
    close  = feat["close"].values
    sma_s  = feat["sma_s"].values
    sma_l  = feat["sma_l"].values
    atr_v  = feat["atr"].values
    idx    = feat.index

    cash     = INITIAL_CAPITAL
    btc_held = 0.0
    trades: list[dict] = []
    equity: list[tuple] = []

    for i in range(1, len(feat)):
        price   = float(close[i])
        s_now   = sma_s[i];  s_prev = sma_s[i-1]
        l_now   = sma_l[i];  l_prev = sma_l[i-1]
        atr_now = atr_v[i]

        if np.isnan(s_now) or np.isnan(l_now) or np.isnan(s_prev) or np.isnan(l_prev):
            equity.append((idx[i], cash + btc_held * price))
            continue

        golden = (s_prev <= l_prev) and (s_now > l_now)
        death  = (s_prev >= l_prev) and (s_now < l_now)

        if golden and btc_held == 0 and not np.isnan(atr_now) and atr_now > 0:
            # ATR sizing: risk a fixed fraction of capital per ATR unit
            qty_atr = (cash * risk_pct) / (atr_now * atr_mult)
            # Cap: don't spend more than 95% of cash
            qty_cap = (cash * 0.95) / (price * (1 + TAKER_FEE))
            qty     = min(qty_atr, qty_cap)
            cost    = qty * price * (1 + TAKER_FEE)
            if cost <= cash and qty > 0:
                btc_held  = qty
                cash     -= cost
                trades.append({"side":"BUY","price":price,"qty":qty,"date":idx[i].date()})

        elif death and btc_held > 0:
            proceeds  = btc_held * price * (1 - TAKER_FEE)
            cash     += proceeds
            trades.append({"side":"SELL","price":price,"qty":btc_held,"date":idx[i].date()})
            btc_held  = 0.0

        equity.append((idx[i], cash + btc_held * price))

    eq = pd.Series({ts: v for ts, v in equity})
    eq.index = pd.DatetimeIndex(eq.index).tz_localize(None)
    returns  = eq.pct_change().dropna()

    if len(returns) < 2 or returns.std() == 0:
        return returns, dict(
            cagr=0.0, sharpe=0.0, max_dd=0.0, sortino=0.0, calmar=0.0,
            win_year_pct=0.0, n_trades=0, win_rate=float("nan"),
            net_pnl=0.0, total_fees=0.0,
        )

    cagr    = float(qs.stats.cagr(returns))           * 100
    sharpe  = float(qs.stats.sharpe(returns))
    max_dd  = float(qs.stats.max_drawdown(returns))   * 100
    sortino = float(qs.stats.sortino(returns))
    calmar  = float(qs.stats.calmar(returns))
    ann     = returns.resample("YE").apply(lambda r: (1+r).prod() - 1)
    win_year_pct = float((ann > 0).mean()) * 100

    open_buy = None; wins = 0; n_complete = 0
    total_fees = net_pnl = 0.0
    for t in trades:
        if t["side"] == "BUY":
            open_buy = t["price"]
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

    return returns, dict(
        cagr=cagr, sharpe=sharpe, max_dd=max_dd, sortino=sortino,
        calmar=calmar, win_year_pct=win_year_pct,
        n_trades=n_complete, win_rate=win_rate,
        net_pnl=net_pnl, total_fees=total_fees,
    )


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
    print(f"Strategy: SW={SW} / LW={LW}  ATR window={ATR_WINDOW}\n")

    exchange = ccxt.coinbase({"enableRateLimit": True})
    raw_df   = fetch_ohlcv(exchange)
    feat     = build_features(raw_df)

    run_id       = _next_run_id()
    journal_rows = []
    best         = {"sharpe": -999.0, "row": None, "stats": None}

    hdr = (f"{'Risk%':>6} {'AtrMult':>7}  "
           f"{'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8}  "
           f"{'Trades':>7} {'WR%':>6}  Gate")
    print(hdr)
    print("-" * len(hdr))

    for risk_pct in RISK_PCTS:
        for atr_mult in ATR_MULTS:
            _, s = run_backtest(feat, risk_pct, atr_mult)
            passed = gate_check(s, baseline)
            wr_str = f"{s['win_rate']*100:6.1f}" if not np.isnan(s["win_rate"]) else "   n/a"

            print(f"{risk_pct*100:>5.1f}% {atr_mult:>7.1f}  "
                  f"{s['cagr']:>8.2f} {s['sharpe']:>8.4f} {s['max_dd']:>8.2f}  "
                  f"{s['n_trades']:>7} {wr_str}  "
                  f"{'PASS ✓' if passed else 'fail'}")

            row = {
                "run_id":        run_id,
                "timestamp":     datetime.now(timezone.utc).isoformat(),
                "short_window":  SW,
                "long_window":   LW,
                "ma_type":       "sma",
                "rsi_filter":    False,
                "vol_confirm":   False,
                "atr_sizing":    True,
                "regime_filter": False,
                "cagr":          round(s["cagr"], 4),
                "sharpe":        round(s["sharpe"], 4),
                "max_dd":        round(s["max_dd"], 4),
                "sortino":       round(s["sortino"], 4),
                "calmar":        round(s["calmar"], 4),
                "win_year_pct":  round(s["win_year_pct"], 4),
                "gate_passed":   passed,
                "notes":         f"Tier2 ATR sizing risk={risk_pct*100:.1f}% mult={atr_mult}",
                "strategy":      "btc_sma_crossover",
                "asset":         "BTC/USD",
                "n_trades":      s["n_trades"],
                "win_rate":      round(s["win_rate"], 4) if not np.isnan(s["win_rate"]) else "",
                "net_pnl":       round(s["net_pnl"], 4),
                "total_fees":    round(s["total_fees"], 4),
                "risk_pct":      risk_pct,
                "atr_mult":      atr_mult,
                "atr_window":    ATR_WINDOW,
                "tier":          2,
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
        print(f"\nBest gate-pass: risk={bp['risk_pct']*100:.1f}%  atr_mult={bp['atr_mult']}")
        print(f"  CAGR={bp['cagr']:.2f}%  Sharpe={bp['sharpe']:.4f}  MaxDD={bp['max_dd']:.2f}%  "
              f"Trades={bp['n_trades']}  WR={float(bp['win_rate'])*100:.1f}%")
        print(f"  Δcagr={bp['cagr']-baseline['cagr']:+.2f}pp  "
              f"Δsharpe={bp['sharpe']-baseline['sharpe']:+.4f}  "
              f"Δmax_dd={bp['max_dd']-baseline['max_dd']:+.2f}pp")

        baseline.update({
            "short_window":  SW,
            "long_window":   LW,
            "ma_type":       "sma",
            "rsi_filter":    False,
            "vol_confirm":   False,
            "atr_sizing":    True,
            "atr_window":    ATR_WINDOW,
            "risk_pct":      bp["risk_pct"],
            "atr_mult":      bp["atr_mult"],
            "regime_filter": False,
            "cagr":          bp["cagr"],
            "sharpe":        bp["sharpe"],
            "max_dd":        bp["max_dd"],
            "sortino":       bp["sortino"],
            "calmar":        bp["calmar"],
            "win_year_pct":  bp["win_year_pct"],
            "recorded_at":   datetime.now(timezone.utc).date().isoformat(),
            "notes":         f"Tier2 ATR risk={bp['risk_pct']*100:.1f}% mult={bp['atr_mult']} SW{SW}/LW{LW}",
        })
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
        print(f"\nbaseline.json updated → Sharpe {baseline['sharpe']:.4f}  CAGR {baseline['cagr']:.2f}%")
    else:
        br = best["row"]
        print(f"\nNo gate passes. Best Sharpe: {br['sharpe']:.4f}  "
              f"[risk={br['risk_pct']*100:.1f}%  atr_mult={br['atr_mult']}]")
        print(f"  Δsharpe={br['sharpe']-baseline['sharpe']:+.4f}  "
              f"Δcagr={br['cagr']-baseline['cagr']:+.2f}pp  "
              f"Δmax_dd={br['max_dd']-baseline['max_dd']:+.2f}pp")
        print("baseline.json unchanged.")


if __name__ == "__main__":
    main()
