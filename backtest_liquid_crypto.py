#!/usr/bin/env python3
"""
CEX Liquid-Crypto Momentum Backtest — BTC/USDT · ETH/USDT · SOL/USDT
======================================================================
Strategy  : N-bar rolling-high breakout + volume confirmation.
            Exit on M-bar rolling-low breakdown, take-profit,
            stop-loss, or timeout.

CEX execution rules (fixed — not subject to optimisation):
  1. 0.075 % taker fee per side  (Binance/OKX VIP-0 rate)
  2. 0.050 % max slippage per side  (conservative for top-5 books)
  3. $0.00 gas / network fee
  ──────────────────────────────────────────────────────────────────
  Round-trip cost: 2 × (0.075 % + 0.050 %) = 0.25 %

Data      : OKX 1-min OHLCV via ccxt (primary).
            Coinbase fallback. Per-asset jump-diffusion synthetic
            fallback when both exchanges are unreachable.

Outputs   : strategy_journal.csv  (appended, all three assets)
            cex_baseline.json     (best per-asset params)
            backtest_report_cex_<ASSET>.html  (quantstats tearsheet)

Run:
    python backtest_liquid_crypto.py               # default params, all assets
    python backtest_liquid_crypto.py --optimize    # full grid search
    python backtest_liquid_crypto.py --asset BTC   # single asset
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import quantstats as qs

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ── CEX execution constants (FIXED — never optimised) ─────────────────────────
TAKER_FEE      = 0.00075   # 0.075 % per side
SLIPPAGE       = 0.0005    # 0.050 % per side
GAS_FEE        = 0.00      # $0.00 on CEX
COST_PER_SIDE  = TAKER_FEE + SLIPPAGE   # 0.125 % per side
ROUND_TRIP_PCT = 2 * COST_PER_SIDE      # 0.250 % total

# ── capital & backtest config ──────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
LOOKBACK_DAYS   = 14        # 14 × 1440 = 20 160 bars per asset
RISK_PCT        = 0.02      # 2 % of equity per trade

# ── default strategy params (overridden per combination in the optimizer) ─────
DEFAULT_PARAMS = dict(
    breakout_window  = 15,
    breakdown_window = 10,
    vol_window       = 20,
    vol_mult         = 2.0,
    take_profit      = 0.010,   # 1.0 %
    stop_loss        = 0.005,   # 0.5 %
    max_hold_bars    = 60,
    risk_pct         = RISK_PCT,
)

# ── assets ─────────────────────────────────────────────────────────────────────
ASSET_CONFIG = {
    "BTC/USDT": dict(start_price=42_000, annual_vol=0.55, annual_drift=0.35,
                     jump_lam=0.3/1440, jump_mu=0.020, jump_sig=0.010, seed=1),
    "ETH/USDT": dict(start_price=2_200,  annual_vol=0.70, annual_drift=0.40,
                     jump_lam=0.5/1440, jump_mu=0.030, jump_sig=0.015, seed=2),
    "SOL/USDT": dict(start_price=100,    annual_vol=0.90, annual_drift=0.60,
                     jump_lam=1.0/1440, jump_mu=0.050, jump_sig=0.025, seed=3),
}

# ── optimizer grid ─────────────────────────────────────────────────────────────
OPT_GRID = dict(
    breakout_window  = [10, 15, 20, 30],
    take_profit      = [0.004, 0.007, 0.010, 0.015, 0.020],
    stop_loss        = [0.003, 0.006, 0.010],
    vol_mult         = [1.5, 2.0, 2.5, 3.0],
    max_hold_bars    = [30, 60, 90, 120],
)
OPT_FIXED = dict(breakdown_window=10, vol_window=20, risk_pct=RISK_PCT)

# ── persistence files ──────────────────────────────────────────────────────────
CEX_BASELINE_FILE = "cex_baseline.json"
JOURNAL_FILE      = "strategy_journal.csv"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _rows_to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df[~df.index.duplicated(keep="last")]


def fetch_ohlcv_ccxt(
    exchange_name: str,
    symbol: str,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Paginate 1-min OHLCV from the given ccxt exchange."""
    import ccxt
    ex       = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    since_ms = ex.milliseconds() - lookback_days * 86_400_000
    end_ms   = ex.milliseconds()
    rows: list = []

    while since_ms < end_ms:
        batch = ex.fetch_ohlcv(symbol, "1m", since=since_ms, limit=300)
        if not batch:
            break
        rows.extend(batch)
        since_ms = batch[-1][0] + 60_000
        time.sleep(ex.rateLimit / 1000)

    df = _rows_to_df(rows)
    # trim to requested window
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
    return df[df.index >= cutoff]


def generate_synthetic(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Per-asset jump-diffusion synthetic 1-min OHLCV."""
    cfg  = ASSET_CONFIG[symbol]
    rng  = np.random.default_rng(cfg["seed"])
    n    = days * 1440
    idx  = pd.date_range("2024-09-01", periods=n, freq="1min", tz="UTC")

    sig_min   = cfg["annual_vol"]   / np.sqrt(252 * 1440)
    drift_min = cfg["annual_drift"] / (252 * 1440)
    lam       = cfg["jump_lam"]

    jumps  = rng.normal(cfg["jump_mu"], cfg["jump_sig"], n) * rng.poisson(lam, n)
    lr     = drift_min + sig_min * rng.standard_normal(n) + jumps
    prices = cfg["start_price"] * np.exp(np.cumsum(lr))

    rng2   = np.abs(rng.normal(0, sig_min * 2, n)) * prices
    highs  = prices + rng2 * 0.6
    lows   = prices - rng2 * 0.4
    opens  = np.concatenate([[prices[0]], prices[:-1]])

    base_vol = max(cfg["start_price"] * 1e4, 1e6) / (1440 * prices)
    vols     = base_vol * (1 + 15 * np.abs(lr)) * rng.lognormal(0, 0.25, n)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices, "volume": vols},
        index=idx,
    )


def load_asset(symbol: str) -> tuple[pd.DataFrame, str]:
    """Try OKX → Coinbase → synthetic per asset."""
    for ex_name in ("okx", "coinbase"):
        try:
            df = fetch_ohlcv_ccxt(ex_name, symbol)
            if len(df) > 1000:
                return df, ex_name
        except Exception as e:
            print(f"  [{ex_name}] {type(e).__name__}: {str(e)[:70]}")
    print(f"  [synthetic] using jump-diffusion for {symbol}")
    return generate_synthetic(symbol), "synthetic"


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def add_features(
    raw: pd.DataFrame,
    breakout_window: int  = 15,
    breakdown_window: int = 10,
    vol_window: int       = 20,
) -> pd.DataFrame:
    df = raw.copy()
    c  = df["close"].shift(1)
    df["rolling_high"] = c.rolling(breakout_window).max()
    df["rolling_low"]  = c.rolling(breakdown_window).min()
    df["avg_volume"]   = df["volume"].shift(1).rolling(vol_window).mean()
    return df.dropna(subset=["rolling_high", "rolling_low", "avg_volume"])


def precompute_features(
    raw: pd.DataFrame, breakout_windows: list[int]
) -> dict[int, pd.DataFrame]:
    """Cache a feature DataFrame per unique breakout_window."""
    cache: dict[int, pd.DataFrame] = {}
    bw_fixed = OPT_FIXED["breakdown_window"]
    vw_fixed = OPT_FIXED["vol_window"]
    for bw in breakout_windows:
        cache[bw] = add_features(raw, bw, bw_fixed, vw_fixed)
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# 3. BACKTEST ENGINE  (numpy fast-path — CEX fee model)
# ══════════════════════════════════════════════════════════════════════════════

def run_fast(df: pd.DataFrame, p: dict) -> tuple[pd.Series, list[dict]]:
    """
    Bar-by-bar numpy loop with CEX execution model.
    Entry fill  = close × (1 + SLIPPAGE),  taker fee on position value.
    Exit fill   = close × (1 - SLIPPAGE),  taker fee on proceeds.
    Gas         = $0.
    Returns (equity_series, list_of_trade_dicts).
    """
    tp   = p["take_profit"]
    sl   = p["stop_loss"]
    mh   = p["max_hold_bars"]
    vm   = p["vol_mult"]
    rp   = p.get("risk_pct", RISK_PCT)

    prices  = df["close"].values.astype(np.float64)
    r_high  = df["rolling_high"].values.astype(np.float64)
    r_low   = df["rolling_low"].values.astype(np.float64)
    avgvol  = df["avg_volume"].values.astype(np.float64)
    vols    = df["volume"].values.astype(np.float64)
    n       = len(prices)

    equity_arr = np.empty(n, dtype=np.float64)
    trades: list[dict] = []

    cash    = INITIAL_CAPITAL
    in_pos  = False
    qty     = 0.0
    emid    = 0.0
    pos_usd = 0.0
    ebar    = 0

    for i in range(n):
        price  = prices[i]
        equity = cash + qty * price
        equity_arr[i] = equity

        # ── exit check ────────────────────────────────────────────────────
        if in_pos:
            pct    = (price - emid) / emid
            reason = ""
            if   pct >= tp:          reason = "take_profit"
            elif pct <= -sl:         reason = "stop_loss"
            elif price < r_low[i]:   reason = "breakdown"
            elif i - ebar >= mh:     reason = "timeout"

            if reason:
                exit_exec  = price * (1 - SLIPPAGE)
                proceeds   = qty * exit_exec
                exit_fee   = proceeds * TAKER_FEE
                cash      += proceeds - exit_fee

                gross  = (price  - emid)  * qty
                s_cost = (emid   + price) * qty * SLIPPAGE
                f_cost = pos_usd * TAKER_FEE + proceeds * TAKER_FEE
                trades.append(dict(
                    gross_pnl=gross, slip_cost=s_cost, fee_cost=f_cost,
                    net_pnl=gross - s_cost - f_cost,
                    exit_reason=reason,
                ))
                in_pos = False; qty = 0.0; pos_usd = 0.0

        # ── entry check ───────────────────────────────────────────────────
        if not in_pos and not np.isnan(r_high[i]) and not np.isnan(avgvol[i]):
            if price > r_high[i] and vols[i] > avgvol[i] * vm:
                pos_usd   = equity * rp
                entry_exec = price * (1 + SLIPPAGE)
                entry_fee  = pos_usd * TAKER_FEE
                total_cost = pos_usd + entry_fee    # gas = $0

                if total_cost <= cash:
                    qty    = pos_usd / entry_exec
                    cash  -= total_cost
                    in_pos = True; emid = price; ebar = i

    # ── force-close at last bar ───────────────────────────────────────────
    if in_pos:
        lp        = prices[-1]
        exit_exec = lp * (1 - SLIPPAGE)
        proceeds  = qty * exit_exec
        exit_fee  = proceeds * TAKER_FEE
        cash     += proceeds - exit_fee
        equity_arr[-1] = cash

        gross  = (lp   - emid) * qty
        s_cost = (emid + lp)   * qty * SLIPPAGE
        f_cost = pos_usd * TAKER_FEE + proceeds * TAKER_FEE
        trades.append(dict(
            gross_pnl=gross, slip_cost=s_cost, fee_cost=f_cost,
            net_pnl=gross - s_cost - f_cost,
            exit_reason="end_of_data",
        ))

    eq = pd.Series(equity_arr, index=df.index, name="CEX Momentum")
    return eq, trades


# ══════════════════════════════════════════════════════════════════════════════
# 4. METRICS
# ══════════════════════════════════════════════════════════════════════════════

def daily_returns(equity_s: pd.Series) -> pd.Series:
    d = equity_s.resample("1D").last().dropna()
    d.index = d.index.tz_localize(None)
    r = d.pct_change().dropna()
    r.name = equity_s.name
    return r


def benchmark_returns(df: pd.DataFrame, symbol: str) -> pd.Series:
    d = df["close"].resample("1D").last().dropna()
    d.index = d.index.tz_localize(None)
    r = d.pct_change().dropna()
    r.name = f"{symbol} buy & hold"
    return r


def fast_metrics(equity_s: pd.Series) -> dict:
    ret = daily_returns(equity_s)
    if len(ret) < 3 or ret.std() == 0:
        return dict(cagr=np.nan, sharpe=np.nan, max_dd=np.nan, sortino=np.nan)
    try:
        return dict(
            cagr    = float(qs.stats.cagr(ret)         * 100),
            sharpe  = float(qs.stats.sharpe(ret)),
            max_dd  = float(qs.stats.max_drawdown(ret) * 100),
            sortino = float(qs.stats.sortino(ret)),
        )
    except Exception:
        return dict(cagr=np.nan, sharpe=np.nan, max_dd=np.nan, sortino=np.nan)


def trade_summary(trades: list[dict]) -> dict:
    if not trades:
        return {}
    gross  = sum(t["gross_pnl"] for t in trades)
    slip   = sum(t["slip_cost"] for t in trades)
    fees   = sum(t["fee_cost"]  for t in trades)
    net    = sum(t["net_pnl"]   for t in trades)
    n_win  = sum(1 for t in trades if t["net_pnl"] > 0)
    reasons = pd.Series([t["exit_reason"] for t in trades]).value_counts()
    return dict(n=len(trades), n_win=n_win, gross=gross,
                slip=slip, fees=fees, net=net, reasons=reasons)


def print_cost_breakdown(symbol: str, s: dict) -> None:
    if not s:
        return
    w    = 58
    ref  = max(abs(s["gross"]), 1)
    n, g, sl, fe, net = s["n"], s["gross"], s["slip"], s["fees"], s["net"]
    rt   = (sl + fe) / max(n, 1)
    print(f"\n{'='*w}")
    print(f"  CEX COST BREAKDOWN  — {symbol}")
    print(f"{'='*w}")
    print(f"  Total trades          : {n:>8}")
    print(f"  Win rate (net P&L>0)  : {s['n_win']/n*100:>7.1f}%")
    print(f"  Round-trip cost (cfg) : {ROUND_TRIP_PCT*100:>7.3f}%")
    print(f"{'-'*w}")
    print(f"  Gross P&L (mid-mkt)   : ${g:>10,.2f}")
    print(f"  Slippage cost 0.05%×2 : ${sl:>10,.2f}  ({sl/ref*100:.2f}% of gross)")
    print(f"  Taker fees  0.075%×2  : ${fe:>10,.2f}  ({fe/ref*100:.2f}% of gross)")
    print(f"  Gas / network fees    : $      0.00   (CEX)")
    print(f"  Net P&L               : ${net:>10,.2f}")
    print(f"  Net return on capital : {net/INITIAL_CAPITAL*100:>7.2f}%")
    print(f"{'-'*w}")
    print(f"  Avg gross per trade   : ${g/n:>10.4f}")
    print(f"  Avg cost per trade    : ${rt:>10.4f}  (slip + fee)")
    print(f"  Break-even gross/trade: ${rt:>10.4f}")
    print(f"{'-'*w}")
    print("  Exit reasons:")
    for reason, cnt in s["reasons"].items():
        print(f"    {reason:<22}: {cnt:>4}  ({cnt/n*100:.1f}%)")
    print(f"{'='*w}")


def print_quantstats(returns: pd.Series, benchmark: pd.Series) -> None:
    qs.extend_pandas()
    print(f"\n{'='*58}")
    print(f"  QUANTSTATS — {returns.name}")
    print(f"  Benchmark: {benchmark.name}")
    print(f"{'='*58}")
    qs.reports.metrics(returns, benchmark=benchmark, mode="full")


def save_tearsheet(returns: pd.Series, benchmark: pd.Series, symbol: str) -> None:
    slug = symbol.replace("/", "")
    out  = f"backtest_report_cex_{slug}.html"
    qs.reports.html(returns, benchmark=benchmark, output=out,
                    title=f"CEX Momentum — {symbol}  (0.25% round-trip)",
                    download_filename=out)
    print(f"HTML tearsheet → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. BASELINE PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_baseline() -> dict:
    if os.path.exists(CEX_BASELINE_FILE):
        with open(CEX_BASELINE_FILE) as f:
            return json.load(f)
    # seed: conservative "unoptimised" reference per asset
    seed = {}
    for sym in ASSET_CONFIG:
        seed[sym] = dict(
            breakout_window=15, take_profit=0.010, stop_loss=0.005,
            vol_mult=2.0, max_hold_bars=60,
            net_pnl=-500.0, cagr=-50.0, sharpe=-5.0,
            win_rate=20.0, max_dd=-20.0,
            recorded_at="2026-09-18", notes="seed",
        )
    return seed


def save_baseline(b: dict) -> None:
    with open(CEX_BASELINE_FILE, "w") as f:
        json.dump(b, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 6. OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def optimize_asset(
    symbol: str,
    raw_df: pd.DataFrame,
    baseline: dict,
) -> tuple[list[dict], dict]:
    """
    Run the full OPT_GRID over a single asset.
    Returns (journal_rows, updated_baseline_for_this_asset).
    """
    asset_bl = baseline.get(symbol, {
        "net_pnl": -500.0, "cagr": -50.0, "sharpe": -5.0
    })

    # Pre-compute features per unique breakout_window
    feat_cache = precompute_features(raw_df, OPT_GRID["breakout_window"])

    combos = list(itertools.product(
        OPT_GRID["breakout_window"],
        OPT_GRID["take_profit"],
        OPT_GRID["stop_loss"],
        OPT_GRID["vol_mult"],
        OPT_GRID["max_hold_bars"],
    ))

    hdr = (f"  {'#':>4} {'BW':>3} {'TP':>6} {'SL':>5} {'VM':>5} {'MH':>4}"
           f" │ {'#Tr':>5} {'Win%':>6} {'Gross':>9} {'Net':>9}"
           f" │ {'CAGR%':>7} {'Sharpe':>7} {'MaxDD%':>7} │ Gate")
    sep = "  " + "─" * (len(hdr) - 2)
    print(hdr); print(sep)

    journal_rows: list[dict] = []
    best_sharpe  = asset_bl["sharpe"]

    for idx, (bw, tp, sl, vm, mh) in enumerate(combos, 1):
        # skip structurally invalid: SL must be < TP and TP must cover round-trip
        if sl >= tp or tp < ROUND_TRIP_PCT:
            continue

        p = dict(breakout_window=bw, take_profit=tp, stop_loss=sl,
                 vol_mult=vm, max_hold_bars=mh,
                 breakdown_window=OPT_FIXED["breakdown_window"],
                 vol_window=OPT_FIXED["vol_window"],
                 risk_pct=OPT_FIXED["risk_pct"])

        df       = feat_cache[bw]
        equity_s, tdicts = run_fast(df, p)

        if not tdicts:
            continue

        s = trade_summary(tdicts)
        m = fast_metrics(equity_s)

        n, nw, gross, net = s["n"], s["n_win"], s["gross"], s["net"]
        win = nw / n * 100 if n > 0 else 0.0

        # Gate: beat baseline on all three metrics simultaneously
        gate = (
            not np.isnan(m["sharpe"])
            and net    > asset_bl["net_pnl"]
            and m["cagr"]   > asset_bl["cagr"]
            and m["sharpe"] > asset_bl["sharpe"]
        )

        flag = "✓" if gate else " "
        print(
            f"  {idx:>4} {bw:>3} {tp:>5.1%} {sl:>4.1%} {vm:>5.1f} {mh:>4}"
            f" │ {n:>5} {win:>5.1f}% ${gross:>8,.1f} ${net:>8,.1f}"
            f" │ {m['cagr']:>6.1f}% {m['sharpe']:>7.2f} {m['max_dd']:>6.1f}% │ {flag}"
        )

        row = dict(
            strategy="cex_momentum", asset=symbol,
            iteration=idx, timestamp=datetime.now().isoformat(),
            breakout_window=bw, take_profit=tp, stop_loss=sl,
            vol_mult=vm, max_hold_bars=mh,
            n_trades=n, win_rate=round(win, 2),
            gross_pnl=round(gross, 4),
            total_fees=round(s["fees"], 4),
            total_slippage=round(s["slip"], 4),
            net_pnl=round(net, 4),
            cagr=round(m["cagr"],    4),
            sharpe=round(m["sharpe"], 4),
            max_dd=round(m["max_dd"], 4),
            sortino=round(m["sortino"], 4),
            gate_passed=gate, notes="",
        )
        journal_rows.append(row)

        if gate and m["sharpe"] > best_sharpe:
            best_sharpe = m["sharpe"]
            asset_bl.update(
                breakout_window=bw, take_profit=tp, stop_loss=sl,
                vol_mult=vm, max_hold_bars=mh,
                net_pnl=round(net, 4), cagr=round(m["cagr"], 4),
                sharpe=round(m["sharpe"], 4), win_rate=round(win, 2),
                max_dd=round(m["max_dd"], 4),
                recorded_at=datetime.now().strftime("%Y-%m-%d"),
                notes=f"iter {idx}",
            )

    print(sep)
    return journal_rows, asset_bl


def append_journal(new_rows: list[dict]) -> None:
    df_new = pd.DataFrame(new_rows)
    if os.path.exists(JOURNAL_FILE):
        existing = pd.read_csv(JOURNAL_FILE)
        df_new   = pd.concat([existing, df_new], ignore_index=True)
    df_new.to_csv(JOURNAL_FILE, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> tuple[bool, list[str]]:
    optimize = "--optimize" in sys.argv
    asset_flag = None
    if "--asset" in sys.argv:
        i = sys.argv.index("--asset")
        if i + 1 < len(sys.argv):
            asset_flag = sys.argv[i + 1].upper()
    symbols = [s for s in ASSET_CONFIG if asset_flag is None or asset_flag in s]
    return optimize, symbols


def main() -> None:
    optimize_mode, symbols = _parse_args()

    print("CEX Liquid-Crypto Momentum Backtest")
    print(f"Fee model : taker {TAKER_FEE*100:.3f}%/side  "
          f"slippage {SLIPPAGE*100:.3f}%/side  gas $0.00")
    print(f"Round-trip: {ROUND_TRIP_PCT*100:.3f}%  |  "
          f"Capital: ${INITIAL_CAPITAL:,.0f}  |  "
          f"Lookback: {LOOKBACK_DAYS}d\n")

    # ── fetch data ─────────────────────────────────────────────────────────
    data: dict[str, tuple[pd.DataFrame, str]] = {}
    for sym in symbols:
        print(f"Loading {sym} …")
        df, source = load_asset(sym)
        data[sym] = (df, source)
        print(f"  {source}: {len(df):,} bars  "
              f"({df.index[0].date()} → {df.index[-1].date()})")

    print()

    if not optimize_mode:
        # ── single-run mode ────────────────────────────────────────────────
        for sym, (raw_df, source) in data.items():
            print(f"\n{'═'*60}")
            print(f"  {sym}  [{source}]")
            print(f"{'═'*60}")
            p  = DEFAULT_PARAMS.copy()
            df = add_features(raw_df, p["breakout_window"],
                              p["breakdown_window"], p["vol_window"])
            equity_s, tdicts = run_fast(df, p)
            s  = trade_summary(tdicts)
            if s:
                print_cost_breakdown(sym, s)
            m  = fast_metrics(equity_s)
            print(f"\n  Net P&L : ${s.get('net',0):,.2f}  |  "
                  f"CAGR: {m['cagr']:.1f}%  |  "
                  f"Sharpe: {m['sharpe']:.2f}  |  "
                  f"MaxDD: {m['max_dd']:.1f}%")
        return

    # ── optimizer mode ─────────────────────────────────────────────────────
    baseline    = load_baseline()
    all_journal: list[dict] = []
    summary_rows: list[dict] = []

    total_combos = (
        len(OPT_GRID["breakout_window"]) *
        len(OPT_GRID["take_profit"])     *
        len(OPT_GRID["stop_loss"])       *
        len(OPT_GRID["vol_mult"])        *
        len(OPT_GRID["max_hold_bars"])
    )

    for sym, (raw_df, source) in data.items():
        print(f"\n{'═'*70}")
        print(f"  OPTIMISING  {sym}  [{source}]  |  "
              f"{len(raw_df):,} bars  |  {total_combos} combos")
        print(f"  Baseline to beat: "
              f"net_pnl=${baseline[sym]['net_pnl']:,.2f}  "
              f"sharpe={baseline[sym]['sharpe']:.2f}")
        print(f"{'═'*70}")

        rows, updated_bl = optimize_asset(sym, raw_df, baseline)
        baseline[sym]    = updated_bl
        all_journal.extend(rows)

        # per-asset summary
        if rows:
            rdf    = pd.DataFrame(rows)
            rdf    = rdf[rdf["sharpe"].notna()]
            best   = rdf.loc[rdf["sharpe"].idxmax()] if len(rdf) else None
            passes = rdf["gate_passed"].sum()
            print(f"\n  {sym} — {len(rows)} valid combos  |  {passes} gate-passing")
            if best is not None:
                print(f"  Best Sharpe : {best['sharpe']:.2f}  "
                      f"(BW={int(best['breakout_window'])} "
                      f"TP={best['take_profit']:.1%} "
                      f"SL={best['stop_loss']:.1%} "
                      f"VM={best['vol_mult']:.1f} "
                      f"MH={int(best['max_hold_bars'])})")
                print(f"  Best net P&L: ${best['net_pnl']:,.2f}  "
                      f"Win rate: {best['win_rate']:.1f}%  "
                      f"CAGR: {best['cagr']:.1f}%")
                summary_rows.append(dict(
                    asset=sym, source=source,
                    **{k: best[k] for k in
                       ["breakout_window","take_profit","stop_loss","vol_mult",
                        "max_hold_bars","n_trades","win_rate","net_pnl",
                        "cagr","sharpe","max_dd","gate_passed"]},
                ))

        # full quantstats + tearsheet for best combo
        if rows:
            rdf  = pd.DataFrame(rows)
            rdf  = rdf[rdf["sharpe"].notna()]
            if len(rdf) == 0:
                continue
            best = rdf.loc[rdf["sharpe"].idxmax()]
            bp   = dict(
                breakout_window  = int(best["breakout_window"]),
                breakdown_window = OPT_FIXED["breakdown_window"],
                vol_window       = OPT_FIXED["vol_window"],
                take_profit      = float(best["take_profit"]),
                stop_loss        = float(best["stop_loss"]),
                vol_mult         = float(best["vol_mult"]),
                max_hold_bars    = int(best["max_hold_bars"]),
                risk_pct         = OPT_FIXED["risk_pct"],
            )
            feat_df   = add_features(raw_df, bp["breakout_window"],
                                     bp["breakdown_window"], bp["vol_window"])
            best_eq, best_td = run_fast(feat_df, bp)

            print_cost_breakdown(sym, trade_summary(best_td))

            dr  = daily_returns(best_eq)
            br  = benchmark_returns(feat_df, sym)
            dr, br = dr.align(br, join="inner")
            print_quantstats(dr, br)
            save_tearsheet(dr, br, sym)

    # ── persist journal + baseline ─────────────────────────────────────────
    save_baseline(baseline)
    append_journal(all_journal)
    print(f"\nJournal  → {JOURNAL_FILE}  ({len(all_journal)} rows appended)")
    print(f"Baseline → {CEX_BASELINE_FILE}")

    # ── cross-asset summary table ──────────────────────────────────────────
    if summary_rows:
        print(f"\n{'═'*80}")
        print("  CROSS-ASSET RESULTS — BEST COMBO PER ASSET (by Sharpe)")
        print(f"{'═'*80}")
        sdf = pd.DataFrame(summary_rows)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", "{:.4f}".format)
        cols = ["asset","breakout_window","take_profit","stop_loss","vol_mult",
                "max_hold_bars","n_trades","win_rate","net_pnl","cagr","sharpe",
                "max_dd","gate_passed"]
        print(sdf[cols].to_string(index=False))

        print(f"\n{'─'*50}")
        print("  NET P&L / SHARPE / WIN RATE  (required by spec)")
        print(f"{'─'*50}")
        for _, r in sdf.iterrows():
            print(f"  {r['asset']:<12}  "
                  f"Net P&L: ${r['net_pnl']:>9,.2f}  "
                  f"Sharpe: {r['sharpe']:>6.2f}  "
                  f"Win Rate: {r['win_rate']:>5.1f}%")
        print(f"{'─'*50}")


if __name__ == "__main__":
    main()
