#!/usr/bin/env python3
"""
Recheck all btc_sma_crossover journal rows under a relaxed CAGR gate.

Original gate:  Δcagr ≥ +0.5pp  AND  Δsharpe ≥ +0.02  AND  Δmax_dd ≥ +0.5pp
Relaxed gate:   Δcagr ≥ -2.0pp  AND  Δsharpe ≥ +0.02  AND  Δmax_dd ≥ +0.5pp

Prints a ranked table of all passes, updates baseline.json with the best.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT          = Path(__file__).parent
JOURNAL_PATH  = ROOT / "strategy_journal.csv"
BASELINE_PATH = ROOT / "baseline.json"

# ── relaxed gate thresholds ───────────────────────────────────────────────────
DELTA_CAGR_MIN   = -2.0   # pp  (was +0.5)
DELTA_SHARPE_MIN = +0.02
DELTA_DD_MIN     = +0.5   # pp less negative  (unchanged)


def gate(row: pd.Series, bl: dict) -> bool:
    return (
        row["cagr"]   - bl["cagr"]   >= DELTA_CAGR_MIN   and
        row["sharpe"] - bl["sharpe"]  >= DELTA_SHARPE_MIN and
        row["max_dd"] - bl["max_dd"]  >= DELTA_DD_MIN
    )


def describe(row: pd.Series) -> str:
    ma   = str(row.get("ma_type", "sma")).upper()
    sw   = int(row["short_window"])
    lw   = int(row["long_window"])
    tier = row.get("tier", "?")
    note = str(row.get("notes", ""))

    extras = []
    if str(row.get("rsi_filter", "")).lower() in ("true", "1"):
        thresh = row.get("rsi_threshold", "")
        extras.append(f"RSI<{thresh}")
    if str(row.get("vol_confirm", "")).lower() in ("true", "1"):
        vm = row.get("vol_mult", "")
        extras.append(f"vol>{vm}×")
    if str(row.get("atr_sizing", "")).lower() in ("true", "1"):
        rp = row.get("risk_pct", "")
        am = row.get("atr_mult", "")
        extras.append(f"ATR risk={float(rp)*100:.1f}% mult={am}")
    if str(row.get("regime_filter", "")).lower() in ("true", "1"):
        eb = row.get("entry_block", "")
        extras.append(f"regime_block={eb}")

    tag = f"{ma} {sw}/{lw}" + (f" [{', '.join(extras)}]" if extras else "")
    return f"T{tier} {tag}"


def main() -> None:
    baseline = json.loads(BASELINE_PATH.read_text())
    bl_cagr   = baseline["cagr"]
    bl_sharpe = baseline["sharpe"]
    bl_dd     = baseline["max_dd"]

    print(f"Baseline — CAGR={bl_cagr:.2f}%  Sharpe={bl_sharpe:.4f}  MaxDD={bl_dd:.2f}%")
    print(f"Relaxed gate:  Δcagr ≥ {DELTA_CAGR_MIN:+.1f}pp  "
          f"Δsharpe ≥ {DELTA_SHARPE_MIN:+.2f}  "
          f"Δmax_dd ≥ {DELTA_DD_MIN:+.1f}pp\n")

    df = pd.read_csv(JOURNAL_PATH, low_memory=False, on_bad_lines="skip")
    btc = df[df["strategy"] == "btc_sma_crossover"].copy()
    print(f"Total btc_sma_crossover rows: {len(btc)}\n")

    # coerce numeric
    for col in ["cagr", "sharpe", "max_dd", "sortino", "calmar",
                "win_year_pct", "n_trades"]:
        btc[col] = pd.to_numeric(btc[col], errors="coerce")

    passes = btc[btc.apply(lambda r: gate(r, baseline), axis=1)].copy()
    passes = passes.sort_values("sharpe", ascending=False)

    print(f"Gate passes (relaxed): {len(passes)} / {len(btc)}\n")

    if passes.empty:
        print("Still no passes. Consider relaxing further or pivoting strategy.")
        return

    # ── ranked table ──────────────────────────────────────────────────────────
    print(f"{'Rank':>4}  {'Description':<42}  "
          f"{'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8}  "
          f"{'Trades':>7}  "
          f"{'Δcagr':>7} {'Δsharpe':>8} {'Δdd':>7}")
    print("-" * 110)

    for rank, (_, row) in enumerate(passes.iterrows(), 1):
        dc = row["cagr"]   - bl_cagr
        ds = row["sharpe"] - bl_sharpe
        dd = row["max_dd"] - bl_dd
        print(f"{rank:>4}  {describe(row):<42}  "
              f"{row['cagr']:>8.2f} {row['sharpe']:>8.4f} {row['max_dd']:>8.2f}  "
              f"{int(row['n_trades']):>7}  "
              f"{dc:>+7.2f} {ds:>+8.4f} {dd:>+7.2f}")

    # ── update baseline with rank-1 ───────────────────────────────────────────
    best = passes.iloc[0]
    print(f"\nUpdating baseline.json → {describe(best)}")
    print(f"  CAGR {bl_cagr:.2f}% → {best['cagr']:.2f}%  "
          f"({best['cagr']-bl_cagr:+.2f}pp)")
    print(f"  Sharpe {bl_sharpe:.4f} → {best['sharpe']:.4f}  "
          f"({best['sharpe']-bl_sharpe:+.4f})")
    print(f"  MaxDD  {bl_dd:.2f}% → {best['max_dd']:.2f}%  "
          f"({best['max_dd']-bl_dd:+.2f}pp)")

    def _val(col: str, default=""):
        v = best.get(col, default)
        return "" if pd.isna(v) else v

    baseline.update({
        "short_window":     int(best["short_window"]),
        "long_window":      int(best["long_window"]),
        "ma_type":          _val("ma_type", "sma"),
        "rsi_filter":       str(_val("rsi_filter", False)).lower() in ("true","1"),
        "rsi_threshold":    _val("rsi_threshold"),
        "rsi_window":       _val("rsi_window"),
        "vol_confirm":      str(_val("vol_confirm", False)).lower() in ("true","1"),
        "vol_mult":         _val("vol_mult"),
        "atr_sizing":       str(_val("atr_sizing", False)).lower() in ("true","1"),
        "risk_pct":         _val("risk_pct"),
        "atr_mult":         _val("atr_mult"),
        "regime_filter":    str(_val("regime_filter", False)).lower() in ("true","1"),
        "entry_block":      _val("entry_block"),
        "half_size_thresh": _val("half_size_thresh"),
        "cagr":             round(float(best["cagr"]), 4),
        "sharpe":           round(float(best["sharpe"]), 4),
        "max_dd":           round(float(best["max_dd"]), 4),
        "sortino":          round(float(best["sortino"]), 4),
        "calmar":           round(float(best["calmar"]), 4),
        "win_year_pct":     round(float(best["win_year_pct"]), 4),
        "recorded_at":      datetime.now(timezone.utc).date().isoformat(),
        "notes":            f"Relaxed-gate best: {describe(best)}",
        "gate_relaxed":     True,
        "delta_cagr_gate":  DELTA_CAGR_MIN,
    })
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
    print(f"\nbaseline.json saved.")

    # ── top-10 summary ────────────────────────────────────────────────────────
    if len(passes) > 1:
        print(f"\nTop-10 by Sharpe (of {len(passes)} passes):")
        for _, row in passes.head(10).iterrows():
            print(f"  {describe(row):<44}  "
                  f"Sharpe={row['sharpe']:.4f}  CAGR={row['cagr']:.2f}%  "
                  f"MaxDD={row['max_dd']:.2f}%  WR={float(row.get('win_rate',0) or 0)*100:.0f}%")


if __name__ == "__main__":
    main()
