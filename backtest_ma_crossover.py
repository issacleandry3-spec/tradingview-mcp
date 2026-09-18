#!/usr/bin/env python3
"""
Moving Average Crossover Backtest — BTC/USD on Coinbase
Strategy : Buy when 50-day SMA crosses above 200-day SMA (golden cross).
           Sell (go to cash) when it crosses below (death cross).
Stack     : ccxt (Coinbase OHLCV, equivalent to lumibot CcxtBacktesting data layer)
            ta (SMA indicators)
            quantstats (metrics + HTML tearsheet)

lumibot ships a CcxtBacktesting class that wraps this exact flow — it passes
ccxt.fetch_ohlcv bars into the same Strategy.on_trading_iteration loop.  Here
we implement the data layer and event loop ourselves because lumibot's full
import chain requires polars + pyarrow, which cannot build from source on
ARM/Android (Termux). The backtesting semantics are identical.

Run:
    python backtest_ma_crossover.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import ccxt
import pandas as pd
import quantstats as qs
import ta

# ── config ────────────────────────────────────────────────────────────────────
EXCHANGE_ID   = "coinbase"          # ccxt exchange id (Coinbase Advanced Trade)
SYMBOL        = "BTC/USD"
TIMEFRAME     = "1d"
SHORT_WINDOW  = 50
LONG_WINDOW   = 200
INITIAL_CAPITAL = 100_000.0
START         = datetime(2018, 1, 1, tzinfo=timezone.utc)
END           = datetime(2024, 12, 31, tzinfo=timezone.utc)
REPORT_FILE   = "backtest_report_btc.html"
TAKER_FEE     = 0.006               # Coinbase taker fee (0.6 %)


# ── data layer (CcxtBacktesting equivalent) ───────────────────────────────────
def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Paginate ccxt.fetch_ohlcv to cover the full date range.
    Returns a DataFrame with DatetimeIndex (UTC) and columns:
    open, high, low, close, volume.
    This replicates what lumibot's CcxtBacktestingData.get_historical_prices
    does internally before handing bars to Strategy.on_trading_iteration.
    """
    since_ms = int(start.timestamp() * 1000)
    end_ms   = int(end.timestamp()   * 1000)
    limit    = 300                          # Coinbase max per request
    rows: list[list] = []

    print(f"Fetching {symbol} {timeframe} OHLCV from Coinbase …", flush=True)
    while since_ms < end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        since_ms = batch[-1][0] + 1         # advance past last received candle
        print(f"  fetched up to {datetime.fromtimestamp(batch[-1][0]/1000, tz=timezone.utc).date()}", end="\r")
        time.sleep(exchange.rateLimit / 1000)

    print()
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[df.index < pd.Timestamp(end)]
    df = df[~df.index.duplicated(keep="last")]
    return df


# ── strategy (mirrors lumibot Strategy.on_trading_iteration logic) ────────────
class MACrossoverBacktest:
    """
    Event-driven backtest engine matching lumibot CcxtBacktesting semantics.
    Each daily bar triggers _on_bar(), which runs the same crossover logic
    that lumibot's Strategy.on_trading_iteration would execute.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        short_window: int = SHORT_WINDOW,
        long_window: int  = LONG_WINDOW,
        capital: float    = INITIAL_CAPITAL,
        fee: float        = TAKER_FEE,
    ) -> None:
        self.df           = df.copy()
        self.short_window = short_window
        self.long_window  = long_window
        self.capital      = capital
        self.fee          = fee

        # Compute indicators up-front (same result as running ta inside the loop)
        close = df["close"]
        self.df["sma_short"] = ta.trend.sma_indicator(close, window=short_window)
        self.df["sma_long"]  = ta.trend.sma_indicator(close, window=long_window)

    # ------------------------------------------------------------------
    def run(self) -> pd.Series:
        """
        Simulate the strategy bar by bar.
        Returns a daily returns Series (float) indexed by date.
        """
        cash      = self.capital
        btc_held  = 0.0
        trades: list[dict] = []
        equity: list[tuple[pd.Timestamp, float]] = []

        df = self.df
        idx = df.index

        for i in range(1, len(df)):
            row      = df.iloc[i]
            prev_row = df.iloc[i - 1]

            price      = float(row["close"])
            s_now      = row["sma_short"]
            l_now      = row["sma_long"]
            s_prev     = prev_row["sma_short"]
            l_prev     = prev_row["sma_long"]

            # Skip until both MAs are fully formed
            if pd.isna(s_now) or pd.isna(l_now) or pd.isna(s_prev) or pd.isna(l_prev):
                portfolio_val = cash + btc_held * price
                equity.append((idx[i], portfolio_val))
                continue

            golden_cross = (s_prev <= l_prev) and (s_now > l_now)
            death_cross  = (s_prev >= l_prev) and (s_now < l_now)

            if golden_cross and btc_held == 0:
                # Buy: spend 95 % of cash after fee
                spend    = cash * 0.95
                cost     = spend * (1 + self.fee)
                if cost <= cash:
                    btc_held  = spend / price
                    cash     -= cost
                    trades.append({
                        "date":   idx[i].date(),
                        "side":   "BUY",
                        "price":  price,
                        "btc":    btc_held,
                        "sma_s":  s_now,
                        "sma_l":  l_now,
                    })
                    print(
                        f"GOLDEN CROSS  BUY  {btc_held:.4f} BTC @ {price:>10,.2f}"
                        f"  SMA{self.short_window}={s_now:,.0f}  SMA{self.long_window}={l_now:,.0f}"
                    )

            elif death_cross and btc_held > 0:
                # Sell all BTC
                proceeds  = btc_held * price * (1 - self.fee)
                cash     += proceeds
                trades.append({
                    "date":  idx[i].date(),
                    "side":  "SELL",
                    "price": price,
                    "btc":   btc_held,
                    "sma_s": s_now,
                    "sma_l": l_now,
                })
                print(
                    f"DEATH  CROSS  SELL {btc_held:.4f} BTC @ {price:>10,.2f}"
                    f"  SMA{self.short_window}={s_now:,.0f}  SMA{self.long_window}={l_now:,.0f}"
                )
                btc_held = 0.0

            portfolio_val = cash + btc_held * price
            equity.append((idx[i], portfolio_val))

        self.trades = trades
        equity_s = pd.Series(
            {ts: v for ts, v in equity},
            name=f"BTC/USD SMA {self.short_window}/{self.long_window} Crossover",
        )
        equity_s.index = pd.DatetimeIndex(equity_s.index).tz_localize(None)
        returns = equity_s.pct_change().dropna()
        returns.name = equity_s.name
        return returns


# ── reporting ─────────────────────────────────────────────────────────────────
def _btc_benchmark(df: pd.DataFrame) -> pd.Series:
    """BTC/USD buy-and-hold daily returns from the same OHLCV data."""
    ret = df["close"].pct_change().dropna()
    ret.index = ret.index.tz_localize(None)
    ret.name  = "BTC/USD buy & hold"
    return ret


def _print_metrics(returns: pd.Series, benchmark: pd.Series) -> None:
    qs.extend_pandas()
    width = 60
    print("\n" + "=" * width)
    print(f" {returns.name}")
    print(f" Benchmark : {benchmark.name}")
    print("=" * width)
    qs.reports.metrics(returns, benchmark=benchmark, mode="full")


def _html_tearsheet(returns: pd.Series, benchmark: pd.Series) -> None:
    qs.reports.html(
        returns,
        benchmark=benchmark,
        output=REPORT_FILE,
        title=f"BTC/USD SMA {SHORT_WINDOW}/{LONG_WINDOW} Crossover (Coinbase)",
        download_filename=REPORT_FILE,
    )
    print(f"\nHTML tearsheet  →  {REPORT_FILE}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"BTC/USD SMA {SHORT_WINDOW}/{LONG_WINDOW} Crossover — Coinbase (ccxt)")
    print(f"Period  : {START.date()} → {END.date()}")
    print(f"Capital : ${INITIAL_CAPITAL:,.0f}  |  Fee: {TAKER_FEE*100:.1f}%\n")

    exchange = ccxt.coinbase({"enableRateLimit": True})
    df = fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, START, END)
    print(f"Loaded {len(df)} daily candles  "
          f"({df.index[0].date()} → {df.index[-1].date()})\n")

    backtest = MACrossoverBacktest(df)
    returns  = backtest.run()

    benchmark = _btc_benchmark(df)
    returns, benchmark = returns.align(benchmark, join="inner")

    _print_metrics(returns, benchmark)
    _html_tearsheet(returns, benchmark)


if __name__ == "__main__":
    main()
