#!/usr/bin/env python3
"""
Memecoin Scalper Backtest — 1-Minute High-Frequency Momentum
=============================================================
Strategy  : N-bar rolling-high breakout entry + volume confirmation.
            Exit on M-bar rolling-low breakdown, take-profit, stop-loss,
            or timeout.

Execution rules (fixed — not optimised):
  1. 1.5% fixed slippage on every entry AND every exit
  2. $0.50 network/priority fee deducted per transaction ($1.00/round-trip)
  3. Position size capped at 1% of estimated liquidity-pool depth

Data      : ccxt Binance PEPE/USDT 1-min OHLCV (live).
            Falls back to jump-diffusion synthetic data when offline.

Metrics   : quantstats — net CAGR, Sharpe, Sortino, Max DD,
            plus a custom gas-vs-profit breakdown table.

Usage:
    python backtest_memecoin_scalper.py            # single run (default params)
    python backtest_memecoin_scalper.py --optimize # grid search + journal
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import quantstats as qs

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ── fixed execution constants (never optimised) ────────────────────────────────
SLIPPAGE        = 0.015     # 1.5% per side
GAS_FEE         = 0.50      # USD per transaction
POOL_CAP_PCT    = 0.01      # max position = 1% of pool TVL
POOL_TURNOVER   = 8.0       # pool TVL ≈ rolling-24h-vol-USD / POOL_TURNOVER

# ── default strategy parameters (overridden by optimizer) ─────────────────────
BREAKOUT_WINDOW = 15
BREAKDOWN_WINDOW= 10
VOL_WINDOW      = 20
VOL_MULT        = 1.5
TAKE_PROFIT     = 0.04
STOP_LOSS       = 0.02
MAX_HOLD_BARS   = 60
RISK_PCT        = 0.02

# ── backtest config ────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
REPORT_FILE     = "backtest_report_memecoin.html"
LOOKBACK_DAYS   = 45

# ── ccxt config ────────────────────────────────────────────────────────────────
EXCHANGE_ID     = "binance"
SYMBOL          = "PEPE/USDT"
TIMEFRAME       = "1m"

# ── optimizer grid ─────────────────────────────────────────────────────────────
OPT_GRID = {
    "breakout_window": [10, 20, 30, 45],
    "take_profit":     [0.05, 0.07, 0.09, 0.12, 0.15],
    "stop_loss":       [0.02, 0.04, 0.06],
    "vol_mult":        [1.5, 2.0, 2.5, 3.0],
    "max_hold_bars":   [60, 120, 180, 240],
}
# fixed inside the optimizer
OPT_FIXED = {"breakdown_window": 10, "vol_window": 20, "risk_pct": 0.025}
SCALPER_BASELINE_FILE = "scalper_baseline.json"
SCALPER_JOURNAL_FILE  = "scalper_journal.csv"


# ── 1. data layer ──────────────────────────────────────────────────────────────

def fetch_live_ohlcv(symbol: str = SYMBOL, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Paginate Binance 1-min OHLCV via ccxt."""
    import ccxt

    ex = ccxt.binance({"enableRateLimit": True})
    since_ms = ex.milliseconds() - lookback_days * 86_400_000
    end_ms   = ex.milliseconds()
    rows: list = []

    print(f"Fetching {symbol} 1-min OHLCV from Binance …", flush=True)
    while since_ms < end_ms:
        batch = ex.fetch_ohlcv(symbol, TIMEFRAME, since=since_ms, limit=1000,
                               params={"paginate": False})
        if not batch:
            break
        rows.extend(batch)
        since_ms = batch[-1][0] + 60_000
        print(f"  → {datetime.fromtimestamp(batch[-1][0]/1000, tz=timezone.utc).date()}", end="\r")
        time.sleep(ex.rateLimit / 1000)

    print()
    return _rows_to_df(rows)


def generate_synthetic_ohlcv(days: int = LOOKBACK_DAYS, seed: int = 42) -> pd.DataFrame:
    """
    Jump-diffusion synthetic 1-min OHLCV mimicking memecoin microstructure:
    GBM (20% annual vol) + Poisson pump events (λ≈4/day, avg +6%) + volume spikes.
    """
    rng    = np.random.default_rng(seed)
    n      = days * 1440
    idx    = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")

    sig    = 0.20 / np.sqrt(252 * 1440)
    lam    = 4.0 / 1440
    jumps  = rng.normal(0.06, 0.03, n) * rng.poisson(lam, n)
    lr     = sig * rng.standard_normal(n) + jumps
    prices = 0.000012 * np.exp(np.cumsum(lr))

    rng_range = np.abs(rng.normal(0, sig * 2, n)) * prices
    highs  = prices + rng_range * 0.6
    lows   = prices - rng_range * 0.4
    opens  = np.concatenate([[prices[0]], prices[:-1]])

    base_vol = 5_000_000 / 1440 / prices
    volumes  = base_vol * (1 + 20 * np.abs(lr)) * rng.lognormal(0, 0.3, n)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes},
        index=idx,
    )


def _rows_to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").sort_index()[~pd.to_datetime(
        df.set_index("ts").sort_index().index).duplicated(keep="last")]


def load_ohlcv() -> tuple[pd.DataFrame, str]:
    try:
        df = fetch_live_ohlcv()
        if df.empty:
            raise ValueError("empty")
        return df, "Binance live"
    except Exception as exc:
        print(f"[offline] {type(exc).__name__}: {exc}")
        print("[fallback] generating synthetic jump-diffusion PEPE/USDT data …")
        return generate_synthetic_ohlcv(), "synthetic (jump-diffusion)"


# ── 2. feature engineering ─────────────────────────────────────────────────────

def add_features(
    raw_df: pd.DataFrame,
    breakout_window: int  = BREAKOUT_WINDOW,
    breakdown_window: int = BREAKDOWN_WINDOW,
    vol_window: int       = VOL_WINDOW,
) -> pd.DataFrame:
    df = raw_df.copy()
    c  = df["close"].shift(1)
    df["rolling_high"]   = c.rolling(breakout_window).max()
    df["rolling_low"]    = c.rolling(breakdown_window).min()
    df["avg_volume"]     = df["volume"].shift(1).rolling(vol_window).mean()
    df["vol_usd"]        = df["volume"] * df["close"]
    df["pool_depth_usd"] = df["vol_usd"].rolling(1440, min_periods=60).sum() / POOL_TURNOVER
    return df.dropna(subset=["rolling_high", "rolling_low", "avg_volume", "pool_depth_usd"])


# ── 3a. original bar-by-bar engine (used for single run + full Trade objects) ──

@dataclass
class Trade:
    entry_bar: int; entry_mid: float; entry_exec: float
    qty: float; position_usd: float; pool_depth_usd: float; max_pos_usd: float
    exit_bar: int = -1; exit_mid: float = 0.0; exit_exec: float = 0.0
    exit_reason: str = ""

    @property
    def gross_pnl(self)     -> float: return (self.exit_mid - self.entry_mid) * self.qty
    @property
    def slippage_cost(self) -> float: return (self.entry_mid + self.exit_mid) * self.qty * SLIPPAGE
    @property
    def gas_cost(self)      -> float: return 2 * GAS_FEE
    @property
    def net_pnl(self)       -> float: return self.gross_pnl - self.slippage_cost - self.gas_cost


def run_backtest(df: pd.DataFrame, p: dict | None = None) -> tuple[pd.Series, list[Trade]]:
    tp  = (p or {}).get("take_profit",  TAKE_PROFIT)
    sl  = (p or {}).get("stop_loss",    STOP_LOSS)
    mh  = (p or {}).get("max_hold_bars",MAX_HOLD_BARS)
    vm  = (p or {}).get("vol_mult",     VOL_MULT)
    rp  = (p or {}).get("risk_pct",     RISK_PCT)

    cash = INITIAL_CAPITAL
    in_pos = False
    cur: Trade | None = None
    trades: list[Trade] = []
    equity_records = []

    for i, (ts, row) in enumerate(df.iterrows()):
        price = float(row["close"])
        held  = cur.qty * price if in_pos else 0.0
        equity = cash + held
        equity_records.append((ts, equity))

        if in_pos and cur is not None:
            pct = (price - cur.entry_mid) / cur.entry_mid
            reason = ""
            if   pct >= tp:                  reason = "take_profit"
            elif pct <= -sl:                 reason = "stop_loss"
            elif price < row["rolling_low"]: reason = "breakdown"
            elif i - cur.entry_bar >= mh:    reason = "timeout"
            if reason:
                cur.exit_bar = i; cur.exit_mid = price
                cur.exit_exec = price * (1 - SLIPPAGE)
                cur.exit_reason = reason
                cash += cur.qty * cur.exit_exec - GAS_FEE
                trades.append(cur); in_pos = False; cur = None

        if not in_pos:
            if (float(row["close"]) > float(row["rolling_high"])
                    and float(row["volume"]) > float(row["avg_volume"]) * vm
                    and float(row["pool_depth_usd"]) > 0):
                max_pos  = float(row["pool_depth_usd"]) * POOL_CAP_PCT
                pos_usd  = min(equity * rp, max_pos)
                exec_p   = price * (1 + SLIPPAGE)
                if pos_usd + GAS_FEE <= cash:
                    cur = Trade(i, price, exec_p, pos_usd / exec_p, pos_usd,
                                float(row["pool_depth_usd"]), max_pos)
                    cash -= pos_usd + GAS_FEE
                    in_pos = True

    if in_pos and cur is not None:
        lp = float(df["close"].iloc[-1])
        cur.exit_bar = len(df)-1; cur.exit_mid = lp
        cur.exit_exec = lp * (1 - SLIPPAGE); cur.exit_reason = "end_of_data"
        cash += cur.qty * cur.exit_exec - GAS_FEE
        trades.append(cur)

    eq = pd.Series({ts: v for ts, v in equity_records}, name="Memecoin Scalper")
    return eq, trades


# ── 3b. numpy fast-path engine (used by optimizer) ────────────────────────────

def _run_fast(df: pd.DataFrame, p: dict) -> tuple[pd.Series, list[dict]]:
    """
    Same logic as run_backtest() but operates on pre-extracted numpy arrays
    for ~40× speed-up over iterrows. Returns (equity_series, list_of_trade_dicts).
    """
    tp  = p["take_profit"]
    sl  = p["stop_loss"]
    mh  = p["max_hold_bars"]
    vm  = p["vol_mult"]
    rp  = p.get("risk_pct", RISK_PCT)

    prices = df["close"].values.astype(np.float64)
    r_high = df["rolling_high"].values.astype(np.float64)
    r_low  = df["rolling_low"].values.astype(np.float64)
    avgvol = df["avg_volume"].values.astype(np.float64)
    pdepth = df["pool_depth_usd"].values.astype(np.float64)
    vols   = df["volume"].values.astype(np.float64)
    n      = len(prices)

    equity_arr = np.empty(n, dtype=np.float64)
    trades: list[dict] = []

    cash     = INITIAL_CAPITAL
    in_pos   = False
    qty      = 0.0
    emid     = 0.0
    ebar     = 0

    for i in range(n):
        price  = prices[i]
        equity = cash + qty * price
        equity_arr[i] = equity

        if in_pos:
            pct    = (price - emid) / emid
            reason = ""
            if   pct >= tp:              reason = "take_profit"
            elif pct <= -sl:             reason = "stop_loss"
            elif price < r_low[i]:       reason = "breakdown"
            elif i - ebar >= mh:         reason = "timeout"

            if reason:
                exit_exec  = price * (1 - SLIPPAGE)
                gross      = (price - emid) * qty
                slip       = (emid + price) * qty * SLIPPAGE
                gas        = 2 * GAS_FEE
                cash      += qty * exit_exec - GAS_FEE
                trades.append({"gross_pnl": gross, "slippage_cost": slip,
                                "gas_cost": gas, "net_pnl": gross - slip - gas,
                                "exit_reason": reason})
                in_pos = False; qty = 0.0

        if not in_pos and not np.isnan(r_high[i]) and not np.isnan(avgvol[i]):
            if price > r_high[i] and vols[i] > avgvol[i] * vm and pdepth[i] > 0:
                max_pos = pdepth[i] * POOL_CAP_PCT
                pos_usd = min(equity * rp, max_pos)
                exec_p  = price * (1 + SLIPPAGE)
                if pos_usd + GAS_FEE <= cash:
                    qty    = pos_usd / exec_p
                    cash  -= pos_usd + GAS_FEE
                    in_pos = True; emid = price; ebar = i

    if in_pos:
        lp    = prices[-1]
        gross = (lp - emid) * qty
        slip  = (emid + lp) * qty * SLIPPAGE
        gas   = 2 * GAS_FEE
        cash += qty * lp * (1 - SLIPPAGE) - GAS_FEE
        equity_arr[-1] = cash
        trades.append({"gross_pnl": gross, "slippage_cost": slip,
                        "gas_cost": gas, "net_pnl": gross - slip - gas,
                        "exit_reason": "end_of_data"})

    eq = pd.Series(equity_arr, index=df.index, name="Memecoin Scalper")
    return eq, trades


# ── 4. metrics helpers ─────────────────────────────────────────────────────────

def _daily_returns(equity_s: pd.Series) -> pd.Series:
    daily = equity_s.resample("1D").last().dropna()
    daily.index = daily.index.tz_localize(None)
    ret = daily.pct_change().dropna()
    ret.name = equity_s.name
    return ret


def _benchmark_returns(df: pd.DataFrame) -> pd.Series:
    daily = df["close"].resample("1D").last().dropna()
    daily.index = daily.index.tz_localize(None)
    ret = daily.pct_change().dropna()
    ret.name = "PEPE buy & hold"
    return ret


def _fast_metrics(equity_s: pd.Series) -> dict:
    """Compute CAGR/Sharpe/MaxDD/Sortino without printing anything."""
    ret = _daily_returns(equity_s)
    if len(ret) < 5 or ret.std() == 0:
        return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan, "sortino": np.nan}
    try:
        return {
            "cagr":    float(qs.stats.cagr(ret)         * 100),
            "sharpe":  float(qs.stats.sharpe(ret)),
            "max_dd":  float(qs.stats.max_drawdown(ret) * 100),
            "sortino": float(qs.stats.sortino(ret)),
        }
    except Exception:
        return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan, "sortino": np.nan}


def _trade_summary(trades: list) -> dict:
    """Works with both Trade objects and plain dicts."""
    def g(t, k): return getattr(t, k)() if hasattr(t, k) and callable(getattr(t, k)) \
        else (getattr(t, k) if hasattr(t, k) else t[k])

    if not trades:
        return {}

    gross = sum(t.gross_pnl  if isinstance(t, Trade) else t["gross_pnl"]  for t in trades)
    slip  = sum(t.slippage_cost if isinstance(t, Trade) else t["slippage_cost"] for t in trades)
    gas   = sum(t.gas_cost   if isinstance(t, Trade) else t["gas_cost"]   for t in trades)
    net   = sum(t.net_pnl    if isinstance(t, Trade) else t["net_pnl"]    for t in trades)
    n_win = sum(1 for t in trades
                if (t.net_pnl if isinstance(t, Trade) else t["net_pnl"]) > 0)
    reasons = pd.Series(
        [t.exit_reason if isinstance(t, Trade) else t["exit_reason"] for t in trades]
    ).value_counts()
    return {"gross": gross, "slip": slip, "gas": gas, "net": net,
            "n": len(trades), "n_win": n_win, "reasons": reasons}


def print_gas_profit_table(trades: list) -> None:
    s = _trade_summary(trades)
    if not s:
        print("No closed trades."); return
    w = 56
    gross, slip, gas, net = s["gross"], s["slip"], s["gas"], s["net"]
    print(f"\n{'='*w}")
    print("  GAS vs GROSS PROFIT BREAKDOWN")
    print(f"{'='*w}")
    print(f"  Total trades            : {s['n']:>10}")
    print(f"  Win rate (net)          : {s['n_win']/s['n']*100:>9.1f}%")
    print(f"{'-'*w}")
    ref = max(abs(gross), 1)
    print(f"  Gross P&L (mid-market)  : ${gross:>12,.2f}")
    print(f"  Slippage cost (1.5%×2)  : ${slip:>12,.2f}  ({slip/ref*100:.1f}% of gross)")
    print(f"  Gas / network fees      : ${gas:>12,.2f}  ({gas/ref*100:.1f}% of gross)")
    print(f"  Net P&L                 : ${net:>12,.2f}")
    print(f"  Net return on capital   : {net/INITIAL_CAPITAL*100:>9.2f}%")
    print(f"{'-'*w}")
    print(f"  Per-trade avg gross     : ${gross/s['n']:>10.4f}")
    print(f"  Per-trade avg gas       : ${gas/s['n']:>10.4f}")
    print(f"  Break-even gross/trade  : ${(slip+gas)/s['n']:>10.4f}  (slip + gas)")
    print(f"{'-'*w}")
    print("  Exit reasons:")
    for reason, cnt in s["reasons"].items():
        print(f"    {reason:<22}: {cnt:>5}  ({cnt/s['n']*100:.1f}%)")
    print(f"{'='*w}")


def print_quantstats(returns: pd.Series, benchmark: pd.Series) -> None:
    qs.extend_pandas()
    print(f"\n{'='*56}")
    print("  QUANTSTATS PERFORMANCE METRICS")
    print("  Benchmark: buy-and-hold on same price series")
    print(f"{'='*56}")
    qs.reports.metrics(returns, benchmark=benchmark, mode="full")


def save_tearsheet(returns: pd.Series, benchmark: pd.Series,
                   title: str = "Memecoin Scalper — 1-min Momentum") -> None:
    qs.reports.html(returns, benchmark=benchmark, output=REPORT_FILE,
                    title=title, download_filename=REPORT_FILE)
    print(f"\nHTML tearsheet  →  {REPORT_FILE}")


# ── 5. optimizer ───────────────────────────────────────────────────────────────

def _load_scalper_baseline() -> dict:
    if os.path.exists(SCALPER_BASELINE_FILE):
        with open(SCALPER_BASELINE_FILE) as f:
            return json.load(f)
    # seed from initial run results
    return {
        "breakout_window": 15, "take_profit": 0.04, "stop_loss": 0.02,
        "vol_mult": 1.5,       "max_hold_bars": 60,
        "net_pnl": -3426.41,   "cagr": -90.58,   "sharpe": -67.9,
        "max_dd": -34.26,      "n_trades": 637,
        "recorded_at": "2026-09-18",
        "notes": "seed from initial run",
    }


def _save_scalper_baseline(b: dict) -> None:
    with open(SCALPER_BASELINE_FILE, "w") as f:
        json.dump(b, f, indent=2)


def _append_journal(rows: list[dict]) -> None:
    df_new = pd.DataFrame(rows)
    if os.path.exists(SCALPER_JOURNAL_FILE):
        existing = pd.read_csv(SCALPER_JOURNAL_FILE)
        df_new   = pd.concat([existing, df_new], ignore_index=True)
    df_new.to_csv(SCALPER_JOURNAL_FILE, index=False)


def optimize(raw_df: pd.DataFrame) -> None:
    """
    Grid-search over OPT_GRID parameters.
    Uses the numpy fast-path engine and precomputes features per unique
    breakout_window to avoid redundant pandas rolling operations.
    Records every result to scalper_journal.csv; overwrites
    scalper_baseline.json whenever all three gates pass simultaneously.
    """
    baseline = _load_scalper_baseline()
    print(f"\nBaseline to beat  →  net_pnl=${baseline['net_pnl']:,.2f}  "
          f"cagr={baseline['cagr']:.1f}%  sharpe={baseline['sharpe']:.2f}\n")

    # ── pre-compute pool_depth once (doesn't depend on any strategy param)
    base = raw_df.copy()
    base["vol_usd"]       = base["volume"] * base["close"]
    base["pool_depth_usd"] = base["vol_usd"].rolling(1440, min_periods=60).sum() / POOL_TURNOVER

    # ── pre-compute features for each unique breakout_window
    bw_fixed    = OPT_FIXED["breakdown_window"]
    vw_fixed    = OPT_FIXED["vol_window"]
    feature_map: dict[int, pd.DataFrame] = {}
    for bw in OPT_GRID["breakout_window"]:
        df = base.copy()
        df["rolling_high"] = df["close"].shift(1).rolling(bw).max()
        df["rolling_low"]  = df["close"].shift(1).rolling(bw_fixed).min()
        df["avg_volume"]   = df["volume"].shift(1).rolling(vw_fixed).mean()
        feature_map[bw]    = df.dropna(
            subset=["rolling_high", "rolling_low", "avg_volume", "pool_depth_usd"])

    bench_df  = next(iter(feature_map.values()))
    bench_ret = _benchmark_returns(bench_df)

    # ── build full grid
    combos = list(itertools.product(
        OPT_GRID["breakout_window"],
        OPT_GRID["take_profit"],
        OPT_GRID["stop_loss"],
        OPT_GRID["vol_mult"],
        OPT_GRID["max_hold_bars"],
    ))
    total = len(combos)

    hdr = (f"{'#':>4} {'BW':>4} {'TP%':>5} {'SL%':>5} {'VM':>5} {'MH':>4}"
           f" | {'#Tr':>5} {'Win%':>6} {'Gross':>9} {'Net':>9}"
           f" | {'CAGR%':>7} {'Sharpe':>7} {'MaxDD%':>7} | Gate")
    sep = "─" * len(hdr)
    print(f"Grid search: {total} combinations  "
          f"({len(OPT_GRID['breakout_window'])} BW × "
          f"{len(OPT_GRID['take_profit'])} TP × "
          f"{len(OPT_GRID['stop_loss'])} SL × "
          f"{len(OPT_GRID['vol_mult'])} VM × "
          f"{len(OPT_GRID['max_hold_bars'])} MH)\n")
    print(hdr); print(sep)

    journal_rows: list[dict] = []
    all_results:  list[dict] = []

    for idx, (bw, tp, sl, vm, mh) in enumerate(combos, 1):
        p = {"breakout_window": bw, "take_profit": tp, "stop_loss": sl,
             "vol_mult": vm, "max_hold_bars": mh,
             "risk_pct": OPT_FIXED["risk_pct"]}

        df = feature_map[bw]
        equity_s, tdicts = _run_fast(df, p)

        if not tdicts:
            continue

        s    = _trade_summary(tdicts)
        m    = _fast_metrics(equity_s)
        n, nw, gross, net = s["n"], s["n_win"], s["gross"], s["net"]
        win  = nw / n * 100

        # gate: all three metrics must beat current baseline
        gate = (
            net    > baseline["net_pnl"]
            and not np.isnan(m["cagr"])    and m["cagr"]   > baseline["cagr"]
            and not np.isnan(m["sharpe"])  and m["sharpe"] > baseline["sharpe"]
        )

        flag = "✓ PASS" if gate else "      "
        print(
            f"{idx:>4} {bw:>4} {tp:>4.0%} {sl:>4.0%} {vm:>5.1f} {mh:>4}"
            f" | {n:>5} {win:>5.1f}% ${gross:>8,.0f} ${net:>8,.0f}"
            f" | {m['cagr']:>6.1f}% {m['sharpe']:>7.2f} {m['max_dd']:>6.1f}% | {flag}"
        )

        row = dict(
            iteration=idx, timestamp=datetime.now().isoformat(),
            breakout_window=bw, take_profit=tp, stop_loss=sl,
            vol_mult=vm, max_hold_bars=mh,
            n_trades=n, win_rate=round(win, 2),
            gross_pnl=round(gross, 2),
            total_slippage=round(s["slip"], 2),
            total_gas=round(s["gas"], 2),
            net_pnl=round(net, 2),
            cagr=round(m["cagr"], 2), sharpe=round(m["sharpe"], 2),
            max_dd=round(m["max_dd"], 2), sortino=round(m["sortino"], 2),
            gate_passed=gate, notes="",
        )
        journal_rows.append(row)
        all_results.append(row)

        if gate:
            baseline.update(
                breakout_window=bw, take_profit=tp, stop_loss=sl,
                vol_mult=vm, max_hold_bars=mh,
                net_pnl=round(net, 2), cagr=round(m["cagr"], 2),
                sharpe=round(m["sharpe"], 2), max_dd=round(m["max_dd"], 2),
                n_trades=n, recorded_at=datetime.now().strftime("%Y-%m-%d"),
                notes=f"iter {idx}",
            )
            _save_scalper_baseline(baseline)

    print(sep)

    # ── persist journal
    _append_journal(journal_rows)
    print(f"\nJournal  →  {SCALPER_JOURNAL_FILE}  ({len(journal_rows)} rows appended)")
    print(f"Baseline →  {SCALPER_BASELINE_FILE}")

    # ── rank and display top 10 by Sharpe
    if not all_results:
        print("No results recorded."); return

    res_df = pd.DataFrame(all_results)
    res_df = res_df[res_df["sharpe"].notna()].copy()
    top10  = (res_df
              .nlargest(10, "sharpe")
              [["breakout_window","take_profit","stop_loss","vol_mult",
                "max_hold_bars","n_trades","win_rate","net_pnl",
                "cagr","sharpe","max_dd","gate_passed"]]
              .reset_index(drop=True))

    print(f"\n{'='*90}")
    print("  TOP 10 PARAMETER SETS BY SHARPE  (of all gate-passing and non-passing combos)")
    print(f"{'='*90}")
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(top10.to_string(index=True))

    # ── run full quantstats on best combo
    best_row = res_df.loc[res_df["sharpe"].idxmax()]
    best_p = {
        "breakout_window": int(best_row["breakout_window"]),
        "take_profit":     float(best_row["take_profit"]),
        "stop_loss":       float(best_row["stop_loss"]),
        "vol_mult":        float(best_row["vol_mult"]),
        "max_hold_bars":   int(best_row["max_hold_bars"]),
        "risk_pct":        OPT_FIXED["risk_pct"],
    }
    print(f"\n{'='*90}")
    print(f"  FULL REPORT — BEST COMBO: {best_p}")
    print(f"{'='*90}")

    best_df  = feature_map[best_p["breakout_window"]]
    best_eq, best_td = _run_fast(best_df, best_p)
    print_gas_profit_table(best_td)

    best_ret = _daily_returns(best_eq)
    best_ret, bench_ret_a = best_ret.align(bench_ret, join="inner")
    print_quantstats(best_ret, bench_ret_a)
    save_tearsheet(
        best_ret, bench_ret_a,
        title=(f"Memecoin Scalper OPTIMISED — BW={best_p['breakout_window']} "
               f"TP={best_p['take_profit']:.0%} SL={best_p['stop_loss']:.0%} "
               f"VM={best_p['vol_mult']} MH={best_p['max_hold_bars']}")
    )

    # ── persist best params back to the module-level constants as a comment
    print(f"\nBest params (update constants to use):")
    for k, v in best_p.items():
        print(f"  {k.upper():<20} = {v}")


# ── 6. main ────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    optimize_mode = "--optimize" in sys.argv

    print("Memecoin Scalper Backtest — 1-Minute PEPE/USDT")
    print(f"Capital: ${INITIAL_CAPITAL:,.0f}  Slippage: {SLIPPAGE*100:.1f}%/side  "
          f"Gas: ${GAS_FEE}/tx  Pool cap: {POOL_CAP_PCT*100:.0f}%\n")

    raw_df, source = load_ohlcv()
    print(f"Source : {source}")
    print(f"Bars   : {len(raw_df):,}  ({raw_df.index[0].date()} → {raw_df.index[-1].date()})\n")

    if optimize_mode:
        optimize(raw_df)
        return

    # ── single run with default constants
    df = add_features(raw_df)
    print(f"Usable bars: {len(df):,}\n")
    equity_s, trades = run_backtest(df)
    print(f"Final equity: ${equity_s.iloc[-1]:,.2f}  |  Trades: {len(trades)}\n")

    daily_ret = _daily_returns(equity_s)
    bench_ret = _benchmark_returns(df)
    daily_ret, bench_ret = daily_ret.align(bench_ret, join="inner")

    print_gas_profit_table(trades)
    print_quantstats(daily_ret, bench_ret)
    save_tearsheet(daily_ret, bench_ret)


if __name__ == "__main__":
    main()
