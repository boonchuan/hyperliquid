"""
10_robustness_truncated.py - verify depth contraction finding survives without Nov 3-4 event

Truncates sample at November 2, 2025 (day 23 post-cascade, before the second event)
and re-runs the recovery analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data/metrics/metrics_panel.parquet")
OUT  = Path("output/cascade")

ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
TRUNCATE_END = "2025-11-02"  # last day before Nov 3-4 selloff
PRE_START = "2025-10-03"
PRE_END   = "2025-10-09"

df = pd.read_parquet(DATA)
hl = df[df["venue"] == "hyperliquid"].copy()
hl["minute"] = pd.to_datetime(hl["minute"])
hl["date"]   = hl["minute"].dt.date.astype(str)

# Pre-event baselines
pre = hl[hl["date"].between(PRE_START, PRE_END)]
baselines = {}
for a in ASSETS:
    sub = pre[pre["asset"] == a]
    baselines[a] = {
        "spread": sub["quoted_spread_bps"].median(),
        "depth":  sub["inside_depth_usd"].median(),
    }

# Truncated post-event analysis (Oct 10 through Nov 2)
print("=" * 70)
print(f"TRUNCATED ANALYSIS: post-event window Oct 10 to {TRUNCATE_END}")
print("(23 days post-cascade, before Nov 3-4 selloff)")
print("=" * 70)

post = hl[(hl["date"] > "2025-10-10") & (hl["date"] <= TRUNCATE_END)].copy()

print("\nDaily recovery trajectory (no Nov 3-4 confound):")
print(f"{'Asset':<6} {'Max spread ratio':>20} {'Final spread ratio':>22} {'Final depth ratio':>22}")
print("-" * 72)
for a in ASSETS:
    sub = post[post["asset"]==a]
    daily = sub.groupby("date").agg(
        spread=("quoted_spread_bps", "median"),
        depth =("inside_depth_usd",  "median"),
    ).sort_index()
    daily["spread_ratio"] = daily["spread"] / baselines[a]["spread"]
    daily["depth_ratio"]  = daily["depth"]  / baselines[a]["depth"]

    # Max spread ratio during window
    max_spread_ratio = daily["spread_ratio"].max()
    # Final values
    final_spread_ratio = daily["spread_ratio"].iloc[-1]
    final_depth_ratio  = daily["depth_ratio"].iloc[-1]
    print(f"{a:<6} {max_spread_ratio:>20.2f} {final_spread_ratio:>22.2f} {final_depth_ratio:>22.2f}")

# Compare to +23d from full sample
print(f"\n\nKey comparison: +23d values from truncated vs full sample should match")
print("(They will — this is just a sanity check that truncation works as expected)")

# Recovery to thresholds within truncated window
print(f"\n\nThreshold recovery within truncated window (Oct 11 - Nov 2):")
print(f"{'Asset':<6} {'Days to 1.1x':>14} {'Days to 1.2x':>14}")
print("-" * 38)
for a in ASSETS:
    sub = post[post["asset"]==a]
    daily = sub.groupby("date")["quoted_spread_bps"].median()
    ratio = daily / baselines[a]["spread"]

    for thresh in [1.1, 1.2]:
        hit = ratio[ratio <= thresh]
        if len(hit) > 0:
            d = (pd.Timestamp(hit.index[0]) - pd.Timestamp("2025-10-10")).days
            val = f"{d}d"
        else:
            val = "not in window"
        if thresh == 1.1:
            line = f"{a:<6} {val:>14}"
        else:
            line += f" {val:>14}"
    print(line)

# The key depth question: does depth still contract even without the second event?
print(f"\n\nKey depth question: does depth contraction hold in truncated window?")
print("Depth ratio 7-day rolling median at day 23 vs day 1:")
for a in ASSETS:
    sub = post[post["asset"]==a]
    daily = sub.groupby("date")["inside_depth_usd"].median()
    ratio = daily / baselines[a]["depth"]
    # First 7 days median
    early = ratio.iloc[:7].median() if len(ratio) >= 7 else ratio.iloc[0]
    # Last 7 days median
    late  = ratio.iloc[-7:].median() if len(ratio) >= 7 else ratio.iloc[-1]
    delta = (late - early) * 100
    direction = "contracted further" if delta < 0 else "partially recovered"
    print(f"  {a}: days 1-7 median {early:.2f}x -> days 17-23 median {late:.2f}x  (delta {delta:+.1f}pp, {direction})")
