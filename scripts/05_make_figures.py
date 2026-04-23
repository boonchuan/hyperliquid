"""
05_make_figures.py  v1 — produces the figures for the paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "metrics" / "metrics_panel.parquet"
OUT  = Path(__file__).resolve().parent.parent / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.frameon": False,
    "legend.fontsize": 9,
})

VENUE_COLORS = {"hyperliquid": "#1f77b4", "binance":  "#d62728"}
VENUE_LABELS = {"hyperliquid": "Hyperliquid", "binance":  "Binance"}


def load():
    df = pd.read_parquet(DATA).copy()
    df["minute"] = pd.to_datetime(df["minute"])
    df["date_dt"] = df["minute"].dt.date
    if "quoted_spread_bps" in df.columns:
        lo, hi = df["quoted_spread_bps"].quantile([0.005, 0.995])
        df["quoted_spread_bps_w"] = df["quoted_spread_bps"].clip(lo, hi)
    return df


def fig1_spread_density(df):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for venue, sub in df.groupby("venue"):
        s = sub["quoted_spread_bps_w"].dropna()
        if len(s) == 0:
            continue
        s.plot.kde(ax=ax, label=VENUE_LABELS[venue],
                    color=VENUE_COLORS[venue], linewidth=1.5)
    ax.set_xlim(0, max(2, df["quoted_spread_bps_w"].quantile(0.95)))
    ax.set_xlabel("Quoted spread (basis points)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of quoted spreads, March 2025 - February 2026")
    ax.legend()
    fig.savefig(OUT / "fig1_spread_by_venue.png")
    plt.close(fig)


def fig2_spread_by_asset(df):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    summary = (df.groupby(["asset", "venue"])["quoted_spread_bps_w"]
                  .median().unstack("venue"))
    summary = summary[["binance", "hyperliquid"]]
    summary.plot.bar(ax=ax,
                      color=[VENUE_COLORS["binance"], VENUE_COLORS["hyperliquid"]])
    ax.set_xlabel("Asset")
    ax.set_ylabel("Median quoted spread (bps)")
    ax.set_title("Median quoted spread by asset and venue")
    ax.legend(["Binance", "Hyperliquid"])
    plt.xticks(rotation=0)
    fig.savefig(OUT / "fig2_spread_by_asset.png")
    plt.close(fig)


def fig3_cost_by_size(df):
    hl = df[df["venue"] == "hyperliquid"].copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rows = []
    for size_label, col in [("$50K", "cost_50k_bps"),
                              ("$100K", "cost_100k_bps"),
                              ("$500K", "cost_500k_bps")]:
        if col not in hl.columns:
            continue
        for asset in sorted(hl["asset"].unique()):
            sub = hl[hl["asset"] == asset][col].dropna()
            if len(sub) > 0:
                rows.append({"size": size_label, "asset": asset,
                             "median_cost": sub.median()})
    plot_df = pd.DataFrame(rows)
    if len(plot_df) > 0:
        wide = plot_df.pivot(index="asset", columns="size", values="median_cost")
        wide = wide[["$50K", "$100K", "$500K"]]
        wide.plot.bar(ax=ax, colormap="viridis")
        ax.set_xlabel("Asset")
        ax.set_ylabel("Median execution cost (bps)")
        ax.set_title("Hyperliquid simulated execution cost by order size")
        ax.legend(title="Order size")
        plt.xticks(rotation=0)
    fig.savefig(OUT / "fig3_cost_by_size.png")
    plt.close(fig)


def fig4_spread_vs_vol(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, venue in zip(axes, ["binance", "hyperliquid"]):
        sub = df[(df["venue"] == venue)
                 & df["realized_vol_bps"].notna()
                 & df["quoted_spread_bps_w"].notna()]
        if len(sub) > 0:
            x = np.log(sub["realized_vol_bps"].clip(lower=0.1))
            y = np.log(sub["quoted_spread_bps_w"].clip(lower=0.01))
            hb = ax.hexbin(x, y, gridsize=40, cmap="viridis", mincnt=1)
            ax.set_title(VENUE_LABELS[venue])
            ax.set_xlabel("log(realized vol, bps)")
            fig.colorbar(hb, ax=ax, label="Count")
    axes[0].set_ylabel("log(quoted spread, bps)")
    fig.suptitle("Quoted spread versus realized volatility, by venue")
    fig.savefig(OUT / "fig4_spread_vs_vol.png")
    plt.close(fig)


def fig5_spread_timeseries(df):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for venue, sub in df.groupby("venue"):
        daily = sub.groupby("date_dt")["quoted_spread_bps_w"].median()
        daily.plot(ax=ax, label=VENUE_LABELS[venue],
                    color=VENUE_COLORS[venue], linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily median quoted spread (bps)")
    ax.set_title("Quoted spread over the sample period")
    ax.legend()
    fig.savefig(OUT / "fig5_spread_timeseries.png")
    plt.close(fig)


def fig6_binance_adv_selection(df):
    bn = df[df["venue"] == "binance"].copy()
    if "price_impact_bps" not in bn.columns:
        return
    summary = (bn.groupby("asset")[["realized_spread_bps",
                                      "price_impact_bps",
                                      "effective_spread_bps"]]
                  .median())
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    summary[["realized_spread_bps", "price_impact_bps"]].plot.bar(
        stacked=True, ax=ax,
        color=["#1f77b4", "#d62728"]
    )
    ax.set_xlabel("Asset")
    ax.set_ylabel("Median half-spread (bps)")
    ax.set_title("Binance: Huang-Stoll decomposition")
    ax.legend(["Realized half-spread (MM revenue)",
                "Price impact (adverse selection)"])
    plt.xticks(rotation=0)
    fig.savefig(OUT / "fig6_binance_adv_selection.png")
    plt.close(fig)


def main():
    print(f"Loading {DATA}")
    df = load()
    print(f"Sample: {len(df):,} obs")
    print()

    print("Generating figures:")
    for name, fn in [
        ("fig1 - spread density by venue", fig1_spread_density),
        ("fig2 - median spread by asset", fig2_spread_by_asset),
        ("fig3 - HL cost by order size", fig3_cost_by_size),
        ("fig4 - spread vs. realized vol", fig4_spread_vs_vol),
        ("fig5 - spread time series", fig5_spread_timeseries),
        ("fig6 - Binance adverse selection", fig6_binance_adv_selection),
    ]:
        try:
            fn(df)
            print(f"  OK  {name}")
        except Exception as e:
            print(f"  ERR {name}: {e}")

    print()
    print(f"Figures written to: {OUT}")


if __name__ == "__main__":
    main()
