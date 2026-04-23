"""
09_cascade_figures.py - generate cascade-specific figures for v7 paper.

Inputs:
  data/metrics/metrics_panel.parquet
  output/cascade/recovery_panel.csv (from script 08)

Outputs (in output/cascade/):
  fig_cascade_1_intraday.png      - minute-level Oct 10 by asset
  fig_cascade_2_recovery.png      - daily spread ratio vs pre-event, all assets, 51 days
  fig_cascade_3_depth_recovery.png - same but for depth
  fig_cascade_4_scatter.png       - pre-event spread vs +30d persistent widening
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data/metrics/metrics_panel.parquet")
CASCADE_DIR = Path("output/cascade")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
COLORS = {"BTC":"#F7931A", "ETH":"#627EEA", "SOL":"#14F195",
          "XRP":"#23292F", "DOGE":"#C2A633"}

# ============================================================
# FIGURE 1: Intraday Oct 10 by asset
# ============================================================
print("Building Figure 1: intraday Oct 10...")
df = pd.read_parquet(DATA)
hl = df[df["venue"]=="hyperliquid"].copy()
hl["minute"] = pd.to_datetime(hl["minute"])
oct10 = hl[hl["minute"].dt.date == pd.Timestamp("2025-10-10").date()].copy()
oct10["hour_frac"] = oct10["minute"].dt.hour + oct10["minute"].dt.minute/60.0

fig, ax = plt.subplots(figsize=(10, 5))
for a in ASSETS:
    sub = oct10[oct10["asset"]==a].sort_values("minute")
    # 5-min rolling median to denoise
    sub["roll"] = sub["quoted_spread_bps"].rolling(5, min_periods=1).median()
    ax.plot(sub["hour_frac"], sub["roll"], label=a, color=COLORS[a], linewidth=1.2, alpha=0.85)

ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))
ax.set_xlabel("Hour of day (UTC)")
ax.set_ylabel("Quoted spread (bps), 5-min rolling median")
ax.set_yscale("log")
ax.set_ylim(0.05, 200)
ax.axvline(21, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
ax.text(21.1, 100, "Cascade peak\n(hour 21 UTC)", color="red", fontsize=9, va="top")
ax.set_title("October 10, 2025: intraday spread evolution by asset")
ax.legend(loc="upper left", frameon=False, ncol=5)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(CASCADE_DIR / "fig_cascade_1_intraday.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig_cascade_1_intraday.png")

# ============================================================
# FIGURE 2: Recovery curves - spread ratio vs pre-event
# ============================================================
print("Building Figure 2: recovery curves...")
rec = pd.read_csv(CASCADE_DIR / "recovery_panel.csv")
rec["date"] = pd.to_datetime(rec["date"])
post = rec[rec["date"] >= pd.Timestamp("2025-10-10")].copy()

fig, ax = plt.subplots(figsize=(10, 5))
for a in ASSETS:
    sub = post[post["asset"]==a].sort_values("date")
    days_after = (sub["date"] - pd.Timestamp("2025-10-10")).dt.days
    # 7-day rolling median for smoothing
    ratio = sub["spread_ratio"].rolling(7, min_periods=2).median()
    ax.plot(days_after, ratio, label=a, color=COLORS[a], linewidth=1.6, alpha=0.9)

ax.axhline(1.0, color="black", linestyle="-",  alpha=0.3, linewidth=0.8)
ax.axhline(1.1, color="green", linestyle=":",  alpha=0.4, linewidth=0.8, label="1.1x baseline")
ax.axhline(1.5, color="orange",linestyle=":",  alpha=0.4, linewidth=0.8, label="1.5x baseline")
ax.set_xlabel("Days after October 10 cascade")
ax.set_ylabel("Spread ratio vs pre-event median (7-day rolling)")
ax.set_title("Spread recovery dynamics, all 5 assets, 51 days post-cascade")
ax.set_xlim(0, 51)
ax.set_ylim(0.9, 2.5)
ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(CASCADE_DIR / "fig_cascade_2_recovery.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig_cascade_2_recovery.png")

# ============================================================
# FIGURE 3: Depth recovery curves
# ============================================================
print("Building Figure 3: depth recovery...")
fig, ax = plt.subplots(figsize=(10, 5))
for a in ASSETS:
    sub = post[post["asset"]==a].sort_values("date")
    days_after = (sub["date"] - pd.Timestamp("2025-10-10")).dt.days
    ratio = sub["depth_ratio"].rolling(7, min_periods=2).median()
    ax.plot(days_after, ratio, label=a, color=COLORS[a], linewidth=1.6, alpha=0.9)

ax.axhline(1.0, color="black", linestyle="-",  alpha=0.3, linewidth=0.8)
ax.axhline(0.9, color="green", linestyle=":",  alpha=0.4, linewidth=0.8, label="0.9x baseline")
ax.axhline(0.5, color="red",   linestyle=":",  alpha=0.4, linewidth=0.8, label="0.5x baseline")
ax.set_xlabel("Days after October 10 cascade")
ax.set_ylabel("Depth ratio vs pre-event median (7-day rolling)")
ax.set_title("Inside-depth recovery dynamics, all 5 assets, 51 days post-cascade")
ax.set_xlim(0, 51)
ax.set_ylim(0.3, 1.2)
ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(CASCADE_DIR / "fig_cascade_3_depth_recovery.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig_cascade_3_depth_recovery.png")

# ============================================================
# FIGURE 4: Cross-sectional scatter
# ============================================================
print("Building Figure 4: cross-sectional scatter...")
# Compute the +30d persistent widening per asset
target_date = pd.Timestamp("2025-11-09")
pre_dates = (rec["date"] >= pd.Timestamp("2025-10-03")) & (rec["date"] <= pd.Timestamp("2025-10-09"))
pre_spread = rec[pre_dates].groupby("asset")["spread_med"].median()
plus30 = rec[rec["date"]==target_date].set_index("asset")["spread_ratio"]
plus30_depth = rec[rec["date"]==target_date].set_index("asset")["depth_ratio"]

scatter_df = pd.DataFrame({
    "pre_spread": pre_spread,
    "spread_widening_pct": (plus30 - 1) * 100,
    "depth_contraction_pct": (1 - plus30_depth) * 100,
}).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: spread
ax = axes[0]
for _, row in scatter_df.iterrows():
    ax.scatter(row["pre_spread"], row["spread_widening_pct"],
               s=200, color=COLORS[row["asset"]], alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.annotate(row["asset"], (row["pre_spread"], row["spread_widening_pct"]),
                xytext=(7, 7), textcoords="offset points", fontsize=11, fontweight="bold")
ax.set_xscale("log")
ax.set_xlabel("Pre-event median spread (bps, log scale)")
ax.set_ylabel("Spread widening at +30d (% vs pre-event)")
ax.set_title("Persistent spread widening vs baseline liquidity")
ax.grid(True, alpha=0.2)
ax.set_ylim(15, 50)

# Right: depth
ax = axes[1]
for _, row in scatter_df.iterrows():
    ax.scatter(row["pre_spread"], row["depth_contraction_pct"],
               s=200, color=COLORS[row["asset"]], alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.annotate(row["asset"], (row["pre_spread"], row["depth_contraction_pct"]),
                xytext=(7, 7), textcoords="offset points", fontsize=11, fontweight="bold")
ax.set_xscale("log")
ax.set_xlabel("Pre-event median spread (bps, log scale)")
ax.set_ylabel("Depth contraction at +30d (% vs pre-event)")
ax.set_title("Persistent depth contraction vs baseline liquidity")
ax.grid(True, alpha=0.2)
ax.set_ylim(15, 50)

plt.tight_layout()
plt.savefig(CASCADE_DIR / "fig_cascade_4_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig_cascade_4_scatter.png")

print("\nAll cascade figures written to:", CASCADE_DIR)
