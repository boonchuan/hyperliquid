"""
03_compute_metrics.py  v4 — corrected metrics with HL trade inference + proper A-R
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

ASSETS_HL = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
ASSETS_BN = {a: f"{a}USDT" for a in ASSETS_HL}
ORDER_SIZES = [50_000, 100_000, 500_000]


def walk_book_cost(levels_one_side, mid, target_notional, is_ask):
    filled_size = 0.0
    filled_cost = 0.0
    target_filled = False
    for level in levels_one_side:
        px = float(level["px"])
        sz = float(level["sz"])
        level_notional = px * sz
        remaining_target = target_notional - filled_cost
        if remaining_target <= 0:
            target_filled = True
            break
        take_notional = min(level_notional, remaining_target)
        take_size = take_notional / px
        filled_size += take_size
        filled_cost += take_notional
        if filled_cost >= target_notional:
            target_filled = True
            break
    if not target_filled or filled_size == 0:
        return np.nan
    vwap = filled_cost / filled_size
    if is_ask:
        return (vwap - mid) / mid * 1e4
    else:
        return (mid - vwap) / mid * 1e4


def parse_hl_snapshot(row):
    levels = row["raw"]["data"]["levels"]
    bids = levels[0]
    asks = levels[1]
    if len(bids) == 0 or len(asks) == 0:
        return None
    best_bid = float(bids[0]["px"])
    best_ask = float(asks[0]["px"])
    mid = (best_bid + best_ask) / 2
    spread_bps = (best_ask - best_bid) / mid * 1e4
    best_bid_sz = float(bids[0]["sz"])
    best_ask_sz = float(asks[0]["sz"])
    inside_depth = best_bid * best_bid_sz + best_ask * best_ask_sz
    costs = {}
    for size in ORDER_SIZES:
        buy_cost = walk_book_cost(asks, mid, size, is_ask=True)
        sell_cost = walk_book_cost(bids, mid, size, is_ask=False)
        if np.isnan(buy_cost) and np.isnan(sell_cost):
            costs[size] = np.nan
        elif np.isnan(buy_cost):
            costs[size] = sell_cost
        elif np.isnan(sell_cost):
            costs[size] = buy_cost
        else:
            costs[size] = (buy_cost + sell_cost) / 2
    return {
        "mid": mid, "best_bid": best_bid, "best_ask": best_ask,
        "spread_bps": spread_bps, "inside_depth_usd": inside_depth,
        "best_bid_sz": best_bid_sz, "best_ask_sz": best_ask_sz,
        "cost_50k_bps":  costs[50_000],
        "cost_100k_bps": costs[100_000],
        "cost_500k_bps": costs[500_000],
    }


def infer_hl_trades(snap_df):
    snap_df = snap_df.sort_index()
    pa = snap_df["best_ask"].values
    pb = snap_df["best_bid"].values
    sa = snap_df["best_ask_sz"].values
    sb = snap_df["best_bid_sz"].values
    ts = snap_df.index.values
    trades = []
    for i in range(1, len(snap_df)):
        if pa[i] > pa[i-1]:
            trades.append({"timestamp": ts[i], "price": pa[i-1], "direction": 1})
        elif pa[i] == pa[i-1] and sa[i] < sa[i-1] * 0.5:
            trades.append({"timestamp": ts[i], "price": pa[i], "direction": 1})
        if pb[i] < pb[i-1]:
            trades.append({"timestamp": ts[i], "price": pb[i-1], "direction": -1})
        elif pb[i] == pb[i-1] and sb[i] < sb[i-1] * 0.5:
            trades.append({"timestamp": ts[i], "price": pb[i], "direction": -1})
    if not trades:
        return None
    return pd.DataFrame(trades).set_index("timestamp")


def roll_spread_bps(prices):
    if len(prices) < 3:
        return np.nan
    dp = np.diff(np.log(prices))
    if len(dp) < 2:
        return np.nan
    cov = np.cov(dp[:-1], dp[1:])[0, 1]
    if cov >= 0:
        return np.nan
    return 2 * np.sqrt(-cov) * 1e4


def abdi_ranaldo_two_day(close_t, high_t, low_t, close_t1, high_t1, low_t1):
    if any(pd.isna([close_t, high_t, low_t, close_t1, high_t1, low_t1])):
        return np.nan
    if min(close_t, high_t, low_t, close_t1, high_t1, low_t1) <= 0:
        return np.nan
    log_c_t  = np.log(close_t)
    log_h_t  = np.log(high_t); log_l_t  = np.log(low_t)
    log_h_t1 = np.log(high_t1); log_l_t1 = np.log(low_t1)
    eta_t  = (log_h_t  + log_l_t)  / 2
    eta_t1 = (log_h_t1 + log_l_t1) / 2
    s_sq = 4 * (log_c_t - eta_t) * (log_c_t - eta_t1)
    if s_sq < 0:
        return np.nan
    return np.sqrt(s_sq) * 1e4


def compute_hl_day(asset, date_str):
    book_path = DATA / "hyperliquid" / f"{asset}_{date_str}_l2Book.parquet"
    if not book_path.exists():
        return None, None
    df = pd.read_parquet(book_path)
    parsed = df.apply(parse_hl_snapshot, axis=1)
    parsed = parsed.dropna()
    if len(parsed) == 0:
        return None, None
    snap = pd.DataFrame(list(parsed.values))
    snap["timestamp"] = pd.to_datetime(df.loc[parsed.index, "time"].values, format="ISO8601")
    snap = snap.set_index("timestamp").sort_index()

    snap["minute"] = snap.index.floor("1min")
    minute = snap.groupby("minute").agg(
        quoted_spread_bps=("spread_bps", "mean"),
        inside_depth_usd=("inside_depth_usd", "mean"),
        cost_50k_bps=("cost_50k_bps", "mean"),
        cost_100k_bps=("cost_100k_bps", "mean"),
        cost_500k_bps=("cost_500k_bps", "mean"),
        mid_price=("mid", "last"),
        n_snapshots=("mid", "count"),
    )
    mid_1s = snap["mid"].resample("1s").last().ffill()
    rv = (np.log(mid_1s).diff() ** 2).resample("1min").sum().pow(0.5) * 1e4
    rv.name = "realized_vol_bps"
    minute = minute.join(rv, how="left")

    trades = infer_hl_trades(snap)
    if trades is not None and len(trades) > 0:
        trades["minute"] = trades.index.floor("1min")
        roll_per_min = trades.groupby("minute")["price"].apply(
            lambda p: roll_spread_bps(p.values)
        )
        roll_per_min.name = "hl_roll_inferred_bps"
        minute = minute.join(roll_per_min, how="left")
    else:
        minute["hl_roll_inferred_bps"] = np.nan

    snap["hour"] = snap.index.floor("1h")
    hourly_ohlc = snap.groupby("hour").agg(
        open=("mid", "first"),
        high=("mid", "max"),
        low=("mid", "min"),
        close=("mid", "last"),
    )
    hourly_ohlc["asset"] = asset
    hourly_ohlc["venue"] = "hyperliquid"

    minute["asset"] = asset
    minute["venue"] = "hyperliquid"
    minute["date"]  = date_str
    return minute.reset_index(), hourly_ohlc.reset_index()


def compute_binance_trades_minute(symbol, date_str):
    path = DATA / "binance" / f"{symbol}_{date_str}_aggTrades.parquet"
    if not path.exists():
        return None, None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    df["px"] = df["price"].astype(float)
    df["sz"] = df["quantity"].astype(float)
    df["notional"] = df["px"] * df["sz"]
    df["direction"] = np.where(df["is_buyer_maker"], -1, 1)
    df = df.set_index("timestamp").sort_index()
    df["minute"] = df.index.floor("1min")
    spread_roll = df.groupby("minute", group_keys=False).apply(
        lambda g: roll_spread_bps(g["px"].values)
    )
    spread_roll.name = "spread_roll_bps"
    vwap_minute = df.groupby("minute").apply(
        lambda g: (g["px"] * g["sz"]).sum() / g["sz"].sum()
    )
    df["vwap_minute"] = df["minute"].map(vwap_minute)
    df["eff_halfspread_bps"] = np.abs(df["px"] - df["vwap_minute"]) / df["vwap_minute"] * 1e4
    px_1s = df["px"].resample("1s").mean().ffill()
    df["px_5m_later"] = df.index.to_series().apply(
        lambda t: px_1s.asof(t + pd.Timedelta(minutes=5))
    ).values
    df["realized_halfspread_bps"] = (df["direction"] *
                                      (df["px"] - df["px_5m_later"]) /
                                      df["px"] * 1e4)
    df["price_impact_bps"] = df["eff_halfspread_bps"] - df["realized_halfspread_bps"]
    minute = df.groupby("minute").agg(
        mid_price=("vwap_minute", "mean"),
        effective_spread_bps=("eff_halfspread_bps", "mean"),
        realized_spread_bps=("realized_halfspread_bps", "mean"),
        price_impact_bps=("price_impact_bps", "mean"),
        volume_usd=("notional", "sum"),
        n_trades=("px", "count"),
    )
    minute = minute.join(spread_roll.to_frame(), how="left")
    rv = (np.log(px_1s).diff() ** 2).resample("1min").sum().pow(0.5) * 1e4
    rv.name = "realized_vol_bps"
    minute = minute.join(rv, how="left")
    minute["quoted_spread_bps"] = minute["spread_roll_bps"]

    df["hour"] = df.index.floor("1h")
    hourly_ohlc = df.groupby("hour").agg(
        open=("px", "first"),
        high=("px", "max"),
        low=("px", "min"),
        close=("px", "last"),
    )
    hourly_ohlc["asset"] = symbol.replace("USDT", "")
    hourly_ohlc["venue"] = "binance"
    return minute, hourly_ohlc.reset_index()


def compute_binance_depth_cost_minute(symbol, date_str):
    path = DATA / "binance" / f"{symbol}_{date_str}_bookDepth.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str))
    df["abs_pct"] = df["percentage"].abs()
    cum = df.groupby(["timestamp", "abs_pct"])["notional"].sum().unstack()
    cum.columns = [f"cum_{int(c)}pct" for c in cum.columns]
    cum = cum.cumsum(axis=1)
    def cost_for_size(row, target):
        for pct in [1, 2, 3, 4, 5]:
            col = f"cum_{pct}pct"
            if col in row and row[col] >= target:
                return (pct - 0.5) * 100
        return 500.0
    for size in ORDER_SIZES:
        cum[f"cost_{size//1000}k_bps"] = cum.apply(
            lambda r: cost_for_size(r, size), axis=1)
    cum["minute"] = cum.index.floor("1min")
    cost_cols = [f"cost_{s//1000}k_bps" for s in ORDER_SIZES]
    return cum.groupby("minute")[cost_cols].mean()


def compute_binance_day(asset, date_str):
    sym = ASSETS_BN[asset]
    trades_out = compute_binance_trades_minute(sym, date_str)
    if trades_out is None:
        trades, hourly_ohlc = None, None
    else:
        trades, hourly_ohlc = trades_out
    depth_cost = compute_binance_depth_cost_minute(sym, date_str)
    if trades is None and depth_cost is None:
        return None, None
    if trades is None:
        out = depth_cost
    elif depth_cost is None:
        out = trades
    else:
        out = trades.join(depth_cost, how="outer")
    out["asset"] = asset
    out["venue"] = "binance"
    out["date"]  = date_str
    return out.reset_index(), hourly_ohlc


def main():
    panels = []
    hourly_panels = []
    hl_files = sorted((DATA / "hyperliquid").glob("*_l2Book.parquet"))
    combos = set()
    for f in hl_files:
        parts = f.stem.split("_")
        combos.add((parts[0], parts[1]))
    # SUBSET MODE: restrict to Sep 15 - Dec 15 2025 (Oct 10 cascade + buffer)
    SUBSET_START = "20250915"
    SUBSET_END   = "20251215"
    combos = {(a, d) for (a, d) in combos if SUBSET_START <= d <= SUBSET_END}
    print(f"SUBSET MODE: {len(combos)} (asset, date) combos in window {SUBSET_START}..{SUBSET_END}")
    for asset, date_str in sorted(combos):
        try:
            print(f"  HL  {asset} {date_str} ...", end="", flush=True)
            hl_min, hl_hourly = compute_hl_day(asset, date_str)
            if hl_min is not None:
                panels.append(hl_min)
                hourly_panels.append(hl_hourly)
                print(f" {len(hl_min)} min")
            else:
                print(" no data")
        except Exception as e:
            print(f" ERROR {e}")
        bn_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        try:
            print(f"  BN  {asset} {bn_date} ...", end="", flush=True)
            bn_min, bn_hourly = compute_binance_day(asset, bn_date)
            if bn_min is not None:
                panels.append(bn_min)
                if bn_hourly is not None:
                    hourly_panels.append(bn_hourly)
                print(f" {len(bn_min)} min")
            else:
                print(" no data")
        except Exception as e:
            print(f" ERROR {e}")
        gc.collect()
    if not panels:
        print("No data computed.")
        return
    full = pd.concat(panels, ignore_index=True)
    full["date_dt"] = pd.to_datetime(full["date"], format="mixed")
    full["is_oct10_window"] = full["date_dt"].between(
        pd.Timestamp("2025-10-09"), pd.Timestamp("2025-10-11")
    )
    full = full.drop(columns=["date_dt"])
    out_path = OUT / "metrics_panel.parquet"
    full.to_parquet(out_path, compression="snappy")
    print()
    print(f"Wrote {len(full):,} minute rows to {out_path}")

    hourly = pd.concat(hourly_panels, ignore_index=True)
    hourly = hourly.rename(columns={"hour": "ts"})
    ar_rows = []
    for (asset, venue), grp in hourly.groupby(["asset", "venue"]):
        grp = grp.set_index("ts").sort_index()
        for i in range(len(grp) - 1):
            r0 = grp.iloc[i]
            r1 = grp.iloc[i + 1]
            ar = abdi_ranaldo_two_day(r0["close"], r0["high"], r0["low"],
                                       r1["close"], r1["high"], r1["low"])
            ar_rows.append({"asset": asset, "venue": venue,
                            "ts": grp.index[i], "ar_spread_bps": ar})
    ar = pd.DataFrame(ar_rows)
    ar.to_parquet(OUT / "ar_hourly_panel.parquet", compression="snappy")
    print(f"Wrote {len(ar):,} hourly A-R rows to ar_hourly_panel.parquet")

    print()
    print("=" * 70)
    print("SUMMARY - median by venue")
    print("=" * 70)
    cols = ["quoted_spread_bps", "hl_roll_inferred_bps",
            "cost_50k_bps", "cost_100k_bps", "cost_500k_bps",
            "realized_vol_bps"]
    avail = [c for c in cols if c in full.columns]
    print(full.groupby("venue")[avail].median().round(3))
    print()
    print("A-R median by venue (hourly):")
    print(ar.groupby("venue")["ar_spread_bps"].median().round(3))


if __name__ == "__main__":
    main()
