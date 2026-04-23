"""
08_cascade_analysis.py — venue-internal stress study built on existing HL panel.

Inputs: data/metrics/metrics_panel.parquet (existing)
Outputs:
  output/cascade/recovery_panel.csv  — per-asset, per-day spread/depth ratios vs pre-event
  output/cascade/intraday_oct10.csv  — minute-level cascade evolution
  output/cascade/cascade_stats.txt    — formatted text summary for paper
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data/metrics/metrics_panel.parquet")
OUT  = Path("output/cascade")
OUT.mkdir(parents=True, exist_ok=True)

PRE_START = "2025-10-03"
PRE_END   = "2025-10-09"
POST_START = "2025-10-11"
POST_END   = "2025-12-15"

ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]


def load_hl():
    df = pd.read_parquet(DATA)
    hl = df[df["venue"] == "hyperliquid"].copy()
    hl["minute"] = pd.to_datetime(hl["minute"])
    hl["date"]   = hl["minute"].dt.date.astype(str)
    return hl


def baseline_stats(hl):
    """Pre-event baselines per asset."""
    pre = hl[hl["date"].between(PRE_START, PRE_END)]
    rows = []
    for a in ASSETS:
        sub = pre[pre["asset"] == a]
        rows.append({
            "asset": a,
            "n_minutes_pre": len(sub),
            "spread_pre_med": sub["quoted_spread_bps"].median(),
            "spread_pre_mean": sub["quoted_spread_bps"].mean(),
            "depth_pre_med": sub["inside_depth_usd"].median(),
            "rv_pre_med": sub["realized_vol_bps"].median(),
        })
    return pd.DataFrame(rows)


def recovery_curves(hl, baselines):
    """Per asset, per day, ratio of spread/depth vs pre-event median."""
    rows = []
    base = baselines.set_index("asset")
    for a in ASSETS:
        sub = hl[(hl["asset"] == a) & (hl["date"] >= PRE_START)].copy()
        daily = sub.groupby("date").agg(
            spread_med=("quoted_spread_bps", "median"),
            spread_mean=("quoted_spread_bps", "mean"),
            depth_med=("inside_depth_usd", "median"),
            rv_med=("realized_vol_bps", "median"),
        )
        daily["asset"] = a
        daily["spread_ratio"] = daily["spread_med"] / base.loc[a, "spread_pre_med"]
        daily["depth_ratio"]  = daily["depth_med"]  / base.loc[a, "depth_pre_med"]
        rows.append(daily.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "recovery_panel.csv", index=False)
    return out


def intraday_oct10(hl):
    """Minute-by-minute spread on Oct 10 itself, pooled across all 5 assets."""
    sub = hl[hl["date"] == "2025-10-10"].copy()
    sub["minute_idx"] = sub["minute"].dt.hour * 60 + sub["minute"].dt.minute
    minute_panel = sub.groupby(["minute_idx", "asset"]).agg(
        spread=("quoted_spread_bps", "first"),
        depth=("inside_depth_usd", "first"),
    ).reset_index()
    # Wide format
    spread_wide = minute_panel.pivot(index="minute_idx", columns="asset", values="spread")
    depth_wide  = minute_panel.pivot(index="minute_idx", columns="asset", values="depth")
    spread_wide["all_assets_med"] = spread_wide.median(axis=1)
    spread_wide.to_csv(OUT / "intraday_oct10_spread.csv")
    depth_wide.to_csv(OUT / "intraday_oct10_depth.csv")
    return spread_wide, depth_wide


def hourly_cascade_table(hl):
    """The 24h table for Oct 10 - pooled median, pooled p95, pooled depth."""
    sub = hl[hl["date"] == "2025-10-10"].copy()
    sub["hour"] = sub["minute"].dt.hour
    hourly = sub.groupby("hour").agg(
        median_spread=("quoted_spread_bps", "median"),
        p95_spread=("quoted_spread_bps", lambda x: x.quantile(0.95)),
        max_spread=("quoted_spread_bps", "max"),
        median_depth=("inside_depth_usd", "median"),
        median_rv=("realized_vol_bps", "median"),
    )
    return hourly


def report(hl, baselines, rec, hourly):
    """Write a structured text summary for paper drafting."""
    out = []
    out.append("=" * 80)
    out.append("CASCADE EVENT STUDY - STRUCTURED RESULTS FOR PAPER")
    out.append("=" * 80)

    out.append("\n--- TABLE A: Pre-event baselines (Oct 3-9, 2025) ---")
    out.append(baselines.to_string(index=False))

    out.append("\n--- TABLE B: Hourly cascade evolution (Oct 10 UTC) ---")
    out.append(hourly.round(3).to_string())

    out.append("\n--- TABLE C: Recovery snapshots, all 5 assets ---")
    base = baselines.set_index("asset")
    for offset in [1, 7, 14, 30, 51]:
        target_date = (pd.Timestamp("2025-10-10") + pd.Timedelta(days=offset)).date().isoformat()
        out.append(f"\n  +{offset}d ({target_date}):")
        for a in ASSETS:
            row = rec[(rec["asset"]==a) & (rec["date"]==target_date)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            out.append(f"    {a}: spread {r['spread_med']:.4f} bps "
                       f"({r['spread_ratio']:.2f}x pre), "
                       f"depth ${r['depth_med']:>12,.0f} ({r['depth_ratio']:.2f}x pre)")

    out.append("\n--- TABLE D: Time to recovery thresholds ---")
    out.append("Days to first reach spread within 1.1x, 1.2x, 1.5x of pre-event")
    for a in ASSETS:
        sub = rec[(rec["asset"]==a) & (rec["date"] > "2025-10-10")].sort_values("date")
        for thresh in [1.1, 1.2, 1.5]:
            hit = sub[sub["spread_ratio"] <= thresh]
            if len(hit) > 0:
                first = hit.iloc[0]
                d = (pd.Timestamp(first["date"]) - pd.Timestamp("2025-10-10")).days
                out.append(f"  {a} <= {thresh}x: by {first['date']} ({d}d)")
            else:
                last = sub.iloc[-1]
                out.append(f"  {a} <= {thresh}x: NEVER (last ratio {last['spread_ratio']:.2f}x at {last['date']})")
        out.append("")

    out.append("\n--- TABLE E: Asset-rank vs recovery (key cross-section) ---")
    out.append("Approx. order of liquidity (BTC most, DOGE least)")
    out.append("Persistent spread widening at +30d:")
    base = baselines.set_index("asset")
    for a in ASSETS:
        target = (pd.Timestamp("2025-10-10") + pd.Timedelta(days=30)).date().isoformat()
        row = rec[(rec["asset"]==a) & (rec["date"]==target)]
        if len(row) > 0:
            r = row.iloc[0]
            pct_wider = (r["spread_ratio"] - 1) * 100
            depth_pct = (1 - r["depth_ratio"]) * 100
            out.append(f"  {a:5s}: +{pct_wider:5.1f}% spread vs pre, depth -{depth_pct:5.1f}% vs pre")

    text = "\n".join(out)
    print(text)
    (OUT / "cascade_stats.txt").write_text(text, encoding="utf-8")


def main():
    print("Loading HL panel...")
    hl = load_hl()
    print(f"  {len(hl):,} HL minute observations across {hl['date'].nunique()} dates")

    baselines = baseline_stats(hl)
    rec = recovery_curves(hl, baselines)
    spread_wide, depth_wide = intraday_oct10(hl)
    hourly = hourly_cascade_table(hl)

    report(hl, baselines, rec, hourly)

    print(f"\nAll outputs written to: {OUT}")


if __name__ == "__main__":
    main()
