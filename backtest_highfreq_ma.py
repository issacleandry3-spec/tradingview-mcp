#!/usr/bin/env python3
"""
High-frequency MA crossover search — short windows requiring ≥100 trades
over the 8-year window (~12 trades/year minimum for statistical significance).

Search space:
  short_window : {3, 5, 7, 10, 15}
  long_window  : {10, 15, 20, 30, 50, 75}
  ma_type      : {sma, ema}
  skip pairs where short_window >= long_window
  => 46 combos total

Hard floor:  n_trades >= 100  (statistically significant Sharpe estimate)
Gate (post-floor):
  Δsharpe ≥ +0.02   vs baseline 0.82
  Δmax_dd ≥ +0.5pp  vs baseline -64.32%
  Δcagr   ≥ -2.0pp  vs baseline 30.25% (user-approved relaxation)

Baseline: SMA 50/200, CAGR=30.25, Sharpe=0.82, MaxDD=-64.32
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

SHORT_WINDOWS = [3, 5, 7, 10, 15]
LONG_WINDOWS  = [10, 15, 20, 30, 50, 75]
MA_TYPES      = ["sma", "ema"]
MIN_TRADES    = 100

DELTA_CAGR_MIN   = -2.0
DELTA_SHARPE_MIN = +0.02
DELTA_DD_MIN     = +0.5

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


def run_backtest(df: pd.DataFrame, sw: int, lw: int,
                 ma_type: str) -> dict:
    close = df["close"]
    if ma_type == "ema":
        ma_s = ta.trend.ema_indicator(close, window=sw).values
        ma_l = ta.trend.ema_indicator(close, window=lw).values
    else:
        ma_s = ta.trend.sma_indicator(close, window=sw).values
        ma_l = ta.trend.sma_indicator(close, window=lw).values

    px  = close.values
    idx = df.index

    cash     = INITIAL_CAPITAL
    btc_held = 0.0
    trades: list[dict] = []
    equity: list[tuple] = []

    for i in range(1, len(df)):
        price  = float(px[i])
        s_now  = ma_s[i];  s_prev = ma_s[i-1]
        l_now  = ma_l[i];  l_prev = ma_l[i-1]

        if np.isnan(s_now) or np.isnan(l_now) or np.isnan(s_prev) or np.isnan(l_prev):
            equity.append((idx[i], cash + btc_held * price))
            continue

        golden = (s_prev <= l_prev) and (s_now > l_now)
        death  = (s_prev >= l_prev) and (s_now < l_now)

        if golden and btc_held == 0:
            spend    = cash * 0.95
            cost     = spend * (1 + TAKER_FEE)
            if cost <= cash:
                btc_held  = spend / price
                cash     -= cost
                trades.append({"side":"BUY","price":price,"qty":btc_held})
        elif death and btc_held > 0:
            proceeds  = btc_held * price * (1 - TAKER_FEE)
            cash     += proceeds
            trades.append({"side":"SELL","price":price,"qty":btc_held})
            btc_held  = 0.0

        equity.append((idx[i], cash + btc_held * price))

    # count complete round-trips
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

    # return early stub if below trade floor (avoids quantstats on degenerate series)
    if n_complete < MIN_TRADES:
        return dict(cagr=float("nan"), sharpe=float("nan"), max_dd=float("nan"),
                    sortino=float("nan"), calmar=float("nan"), win_year_pct=float("nan"),
                    n_trades=n_complete, win_rate=win_rate,
                    net_pnl=net_pnl, total_fees=total_fees,
                    below_floor=True)

    eq = pd.Series({ts: v for ts, v in equity})
    eq.index = pd.DatetimeIndex(eq.index).tz_localize(None)
    returns  = eq.pct_change().dropna()

    if len(returns) < 2 or returns.std() == 0:
        return dict(cagr=0.0, sharpe=0.0, max_dd=0.0, sortino=0.0, calmar=0.0,
                    win_year_pct=0.0, n_trades=n_complete, win_rate=win_rate,
                    net_pnl=net_pnl, total_fees=total_fees, below_floor=False)

    cagr    = float(qs.stats.cagr(returns))         * 100
    sharpe  = float(qs.stats.sharpe(returns))
    max_dd  = float(qs.stats.max_drawdown(returns)) * 100
    sortino = float(qs.stats.sortino(returns))
    calmar  = float(qs.stats.calmar(returns))
    ann     = returns.resample("YE").apply(lambda r: (1+r).prod() - 1)
    win_year_pct = float((ann > 0).mean()) * 100

    return dict(cagr=cagr, sharpe=sharpe, max_dd=max_dd, sortino=sortino,
                calmar=calmar, win_year_pct=win_year_pct, n_trades=n_complete,
                win_rate=win_rate, net_pnl=net_pnl, total_fees=total_fees,
                below_floor=False)


def gate_check(s: dict, bl: dict) -> bool:
    if s.get("below_floor") or any(np.isnan(s[k]) for k in ["cagr","sharpe","max_dd"]):
        return False
    return (
        s["cagr"]   - bl["cagr"]   >= DELTA_CAGR_MIN   and
        s["sharpe"] - bl["sharpe"]  >= DELTA_SHARPE_MIN and
        s["max_dd"] - bl["max_dd"]  >= DELTA_DD_MIN
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
    bl_cagr   = baseline["cagr"]
    bl_sharpe = baseline["sharpe"]
    bl_dd     = baseline["max_dd"]
    print(f"Baseline — CAGR={bl_cagr:.2f}%  Sharpe={bl_sharpe:.4f}  MaxDD={bl_dd:.2f}%")
    print(f"Trade floor: N ≥ {MIN_TRADES}  |  Gate: Δcagr≥{DELTA_CAGR_MIN:+}pp  "
          f"Δsharpe≥{DELTA_SHARPE_MIN:+.2f}  Δmax_dd≥{DELTA_DD_MIN:+.1f}pp\n")

    exchange = ccxt.coinbase({"enableRateLimit": True})
    raw_df   = fetch_ohlcv(exchange)

    combos = [
        (sw, lw, ma)
        for ma in MA_TYPES
        for sw in SHORT_WINDOWS
        for lw in LONG_WINDOWS
        if sw < lw
    ]
    print(f"Running {len(combos)} combos …\n")

    run_id       = _next_run_id()
    journal_rows = []
    qualifying   = []   # rows that cleared the trade floor

    hdr = (f"{'MA':>4} {'SW':>3} {'LW':>3}  "
           f"{'N':>5}  {'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8}  "
           f"{'WR%':>6}  Gate")
    print(hdr)
    print("-" * len(hdr))

    for sw, lw, ma_type in combos:
        s      = run_backtest(raw_df, sw, lw, ma_type)
        passed = gate_check(s, baseline)

        if s.get("below_floor"):
            n_str = f"{'<'+str(MIN_TRADES):>5}"
            print(f"{ma_type.upper():>4} {sw:>3} {lw:>3}  "
                  f"{n_str}  {'—':>8} {'—':>8} {'—':>8}  "
                  f"{'—':>6}  skip (N<{MIN_TRADES})")
        else:
            wr_s = f"{s['win_rate']*100:6.1f}" if not np.isnan(s["win_rate"]) else "   n/a"
            print(f"{ma_type.upper():>4} {sw:>3} {lw:>3}  "
                  f"{s['n_trades']:>5}  "
                  f"{s['cagr']:>8.2f} {s['sharpe']:>8.4f} {s['max_dd']:>8.2f}  "
                  f"{wr_s}  "
                  f"{'PASS ✓' if passed else 'fail'}")
            qualifying.append((sw, lw, ma_type, s, passed))

        row = {
            "run_id":       run_id,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "short_window": sw,
            "long_window":  lw,
            "ma_type":      ma_type,
            "rsi_filter":   False,
            "vol_confirm":  False,
            "atr_sizing":   False,
            "regime_filter":False,
            "cagr":         "" if s.get("below_floor") else round(s["cagr"], 4),
            "sharpe":       "" if s.get("below_floor") else round(s["sharpe"], 4),
            "max_dd":       "" if s.get("below_floor") else round(s["max_dd"], 4),
            "sortino":      "" if s.get("below_floor") else round(s["sortino"], 4),
            "calmar":       "" if s.get("below_floor") else round(s["calmar"], 4),
            "win_year_pct": "" if s.get("below_floor") else round(s["win_year_pct"], 4),
            "gate_passed":  passed,
            "notes":        f"HiFreq N-floor={MIN_TRADES}" + (" below_floor" if s.get("below_floor") else ""),
            "strategy":     "btc_sma_crossover",
            "asset":        "BTC/USD",
            "n_trades":     s["n_trades"],
            "win_rate":     "" if np.isnan(s["win_rate"]) else round(s["win_rate"], 4),
            "net_pnl":      round(s["net_pnl"], 4),
            "total_fees":   round(s["total_fees"], 4),
            "tier":         "HF",
        }
        journal_rows.append(row)
        run_id += 1

    print()
    append_journal(journal_rows)
    print(f"Appended {len(journal_rows)} rows → {JOURNAL_PATH}")

    passes = [(sw,lw,ma,s) for sw,lw,ma,s,p in qualifying if p]
    print(f"\nQualifying (N≥{MIN_TRADES}): {len(qualifying)} / {len(combos)}")
    print(f"Gate passes:                {len(passes)} / {len(qualifying)}")

    if passes:
        passes.sort(key=lambda x: x[3]["sharpe"], reverse=True)
        best_sw, best_lw, best_ma, best_s = passes[0]
        print(f"\nAll gate-passes (ranked by Sharpe):")
        print(f"{'MA':>4} {'SW':>3} {'LW':>3}  "
              f"{'N':>5}  {'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8}  "
              f"{'Δcagr':>7} {'Δsharpe':>8} {'Δdd':>7}")
        print("-" * 80)
        for sw, lw, ma, s in passes:
            print(f"{ma.upper():>4} {sw:>3} {lw:>3}  "
                  f"{s['n_trades']:>5}  "
                  f"{s['cagr']:>8.2f} {s['sharpe']:>8.4f} {s['max_dd']:>8.2f}  "
                  f"{s['cagr']-bl_cagr:>+7.2f} {s['sharpe']-bl_sharpe:>+8.4f} "
                  f"{s['max_dd']-bl_dd:>+7.2f}")

        print(f"\nUpdating baseline.json → {best_ma.upper()} {best_sw}/{best_lw}")
        baseline.update({
            "short_window":  best_sw,
            "long_window":   best_lw,
            "ma_type":       best_ma,
            "rsi_filter":    False,
            "vol_confirm":   False,
            "atr_sizing":    False,
            "regime_filter": False,
            "cagr":          round(best_s["cagr"], 4),
            "sharpe":        round(best_s["sharpe"], 4),
            "max_dd":        round(best_s["max_dd"], 4),
            "sortino":       round(best_s["sortino"], 4),
            "calmar":        round(best_s["calmar"], 4),
            "win_year_pct":  round(best_s["win_year_pct"], 4),
            "n_trades":      best_s["n_trades"],
            "recorded_at":   datetime.now(timezone.utc).date().isoformat(),
            "notes":         f"HiFreq {best_ma.upper()} {best_sw}/{best_lw} N={best_s['n_trades']}",
            "gate_relaxed":  True,
            "delta_cagr_gate": DELTA_CAGR_MIN,
            "min_trades":    MIN_TRADES,
        })
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
        print(f"  CAGR {bl_cagr:.2f}% → {best_s['cagr']:.2f}%  "
              f"({best_s['cagr']-bl_cagr:+.2f}pp)")
        print(f"  Sharpe {bl_sharpe:.4f} → {best_s['sharpe']:.4f}  "
              f"({best_s['sharpe']-bl_sharpe:+.4f})")
        print(f"  MaxDD  {bl_dd:.2f}% → {best_s['max_dd']:.2f}%  "
              f"({best_s['max_dd']-bl_dd:+.2f}pp)")
        print(f"  Trades  8 → {best_s['n_trades']}")
    else:
        if qualifying:
            qualifying.sort(key=lambda x: x[3]["sharpe"], reverse=True)
            print(f"\nTop-5 qualifying combos by Sharpe (no gate passes):")
            print(f"{'MA':>4} {'SW':>3} {'LW':>3}  "
                  f"{'N':>5}  {'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8}  "
                  f"{'Δcagr':>7} {'Δsharpe':>8} {'Δdd':>7}")
            print("-" * 80)
            for sw, lw, ma, s, _ in qualifying[:5]:
                print(f"{ma.upper():>4} {sw:>3} {lw:>3}  "
                      f"{s['n_trades']:>5}  "
                      f"{s['cagr']:>8.2f} {s['sharpe']:>8.4f} {s['max_dd']:>8.2f}  "
                      f"{s['cagr']-bl_cagr:>+7.2f} {s['sharpe']-bl_sharpe:>+8.4f} "
                      f"{s['max_dd']-bl_dd:>+7.2f}")
        print("\nbaseline.json unchanged.")


if __name__ == "__main__":
    main()
