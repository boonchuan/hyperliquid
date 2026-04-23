"""
04_regressions.py  v2 — adds HL effective spread, A-R hourly, Oct 10 + vol-tail robustness
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

DATA_MIN = Path(__file__).resolve().parent.parent / "data" / "metrics" / "metrics_panel.parquet"
DATA_AR  = Path(__file__).resolve().parent.parent / "data" / "metrics" / "ar_hourly_panel.parquet"
OUT      = Path(__file__).resolve().parent.parent / "output" / "regressions"
OUT.mkdir(parents=True, exist_ok=True)


def prepare(df):
    df = df.copy()
    df["date"] = df["date"].astype(str)
    df["minute"] = pd.to_datetime(df["minute"])
    df["hour"] = df["minute"].dt.hour
    df["hl"] = (df["venue"] == "hyperliquid").astype(int)
    df["spread_primary"] = df["quoted_spread_bps"]
    if "hl_roll_inferred_bps" not in df.columns:
        df["hl_roll_inferred_bps"] = np.nan
    df["spread_effective"] = np.where(
        df["hl"] == 1,
        df["hl_roll_inferred_bps"],
        df["quoted_spread_bps"],
    )
    for col in ["spread_primary", "spread_effective", "realized_vol_bps"]:
        df[f"log_{col}"] = np.log(df[col].clip(lower=0.01))
    df["cluster_id"] = df["asset"] + "_" + df["date"]
    for col in ["spread_primary", "spread_effective"]:
        if col in df.columns:
            lo, hi = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lo, hi)
    return df


def drop_na_for_model(df, cols):
    return df.dropna(subset=cols).copy()


def h1_main(df, y, label):
    needed = [y, "hl", "log_realized_vol_bps", "asset", "hour", "cluster_id"]
    sub = drop_na_for_model(df, needed)
    if len(sub) < 1000:
        return None, f"{label}: insufficient ({len(sub)})"
    formula = f"{y} ~ hl + log_realized_vol_bps + C(asset) + C(hour)"
    m = smf.ols(formula, data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["cluster_id"]}
    )
    return m, f"{label}: {len(sub):,} obs"


def h2_mechanism(df, y, label):
    needed = [y, "hl", "log_realized_vol_bps", "asset", "hour", "cluster_id"]
    sub = drop_na_for_model(df, needed)
    if len(sub) < 1000:
        return None, f"{label}: insufficient"
    formula = f"{y} ~ hl * log_realized_vol_bps + C(asset) + C(hour)"
    m = smf.ols(formula, data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["cluster_id"]}
    )
    return m, f"{label}: {len(sub):,} obs"


def h3_cross_section(df, y):
    rows = []
    for asset in sorted(df["asset"].unique()):
        sub = df[df["asset"] == asset].copy()
        sub = drop_na_for_model(sub, [y, "hl", "log_realized_vol_bps",
                                       "hour", "cluster_id"])
        if len(sub) < 500:
            rows.append({"asset": asset, "n": len(sub), "coef": np.nan,
                         "se": np.nan, "tstat": np.nan})
            continue
        formula = f"{y} ~ hl + log_realized_vol_bps + C(hour)"
        m = smf.ols(formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["cluster_id"]}
        )
        rows.append({
            "asset": asset, "n": int(m.nobs),
            "coef": m.params.get("hl", np.nan),
            "se":   m.bse.get("hl", np.nan),
            "tstat":m.tvalues.get("hl", np.nan),
        })
    return pd.DataFrame(rows)


def report(model, label):
    if model is None:
        print(f"  {label}: model is None")
        return None
    coef = model.params.get("hl", np.nan)
    se   = model.bse.get("hl", np.nan)
    t    = model.tvalues.get("hl", np.nan)
    p    = model.pvalues.get("hl", np.nan)
    print(f"  {label}")
    print(f"    HL coef: {coef:+.4f}  (SE {se:.4f}, t={t:+.2f}, p={p:.3g})")
    print(f"    HL/BN multiplier: exp({coef:.3f}) = {np.exp(coef):.2f}x")
    return {"label": label, "coef": coef, "se": se, "t": t, "p": p,
            "multiplier": np.exp(coef)}


def write_summary(model, path, header):
    with open(path, "w") as f:
        f.write(header + "\n" + "=" * 80 + "\n")
        f.write(str(model.summary()) + "\n")


def main():
    print(f"Loading {DATA_MIN}")
    df = pd.read_parquet(DATA_MIN)
    df = prepare(df)
    print(f"Sample: {len(df):,} obs, {df['date'].nunique()} dates, {df['asset'].nunique()} assets")
    print()

    print("=" * 80)
    print("TABLE 1 - Descriptives by venue")
    print("=" * 80)
    cols = ["spread_primary", "spread_effective",
            "cost_50k_bps", "cost_100k_bps", "cost_500k_bps",
            "realized_vol_bps"]
    avail = [c for c in cols if c in df.columns]
    desc = df.groupby("venue")[avail].agg(["median", "mean", "std"]).round(3)
    print(desc)
    desc.to_csv(OUT / "table1_descriptives.csv")
    print()

    all_h1 = []

    print("=" * 80)
    print("TABLE 2 - H1 main results")
    print("=" * 80)
    print("\n  PANEL A: HL quoted vs BN Roll (original specification)")
    m, _ = h1_main(df, "log_spread_primary", "h1_primary")
    all_h1.append(report(m, "Primary (HL quoted vs BN Roll)"))
    if m is not None:
        write_summary(m, OUT / "table2a_h1_primary.txt", "H1 Primary")

    print("\n  PANEL B: HL inferred-Roll vs BN Roll (apples-to-apples effective)")
    m, _ = h1_main(df, "log_spread_effective", "h1_effective")
    all_h1.append(report(m, "Effective (HL inferred-Roll vs BN Roll)"))
    if m is not None:
        write_summary(m, OUT / "table2b_h1_effective.txt", "H1 Effective")

    print("\n  PANEL C: A-R hourly comparison (corrected estimator)")
    if DATA_AR.exists():
        ar = pd.read_parquet(DATA_AR).copy()
        ar["hl"] = (ar["venue"] == "hyperliquid").astype(int)
        ar["log_ar"] = np.log(ar["ar_spread_bps"].clip(lower=0.01))
        ar["date"]   = ar["ts"].dt.date.astype(str)
        ar["cluster_id"] = ar["asset"] + "_" + ar["date"]
        ar_clean = ar.dropna(subset=["log_ar", "hl", "asset", "cluster_id"])
        if len(ar_clean) > 100:
            m_ar = smf.ols("log_ar ~ hl + C(asset)", data=ar_clean).fit(
                cov_type="cluster", cov_kwds={"groups": ar_clean["cluster_id"]}
            )
            all_h1.append(report(m_ar, "A-R hourly (corrected)"))
            write_summary(m_ar, OUT / "table2c_h1_ar.txt", "H1 A-R")
        else:
            print("    Insufficient A-R observations.")
    else:
        print("    A-R panel not found.")

    print()
    print("=" * 80)
    print("TABLE 3 - H2 mechanism (HL x log realized vol)")
    print("=" * 80)
    for spec_y, spec_label in [("log_spread_primary", "Primary"),
                                ("log_spread_effective", "Effective")]:
        print(f"\n  {spec_label}:")
        m, _ = h2_mechanism(df, spec_y, f"h2_{spec_label}")
        if m is not None:
            for term in ["hl", "log_realized_vol_bps", "hl:log_realized_vol_bps"]:
                if term in m.params.index:
                    c, s, t = m.params[term], m.bse[term], m.tvalues[term]
                    print(f"    {term:<35s} {c:+.4f}  (SE {s:.4f}, t={t:+.2f})")
            write_summary(m, OUT / f"table3_h2_{spec_label}.txt", f"H2 {spec_label}")

    print()
    print("=" * 80)
    print("TABLE 4 - H3 cross-section by asset")
    print("=" * 80)
    h3_p = h3_cross_section(df, "log_spread_primary")
    h3_e = h3_cross_section(df, "log_spread_effective")
    print("\n  Primary:")
    print(h3_p.to_string(index=False))
    print("\n  Effective:")
    print(h3_e.to_string(index=False))
    h3_p.to_csv(OUT / "table4a_h3_primary.csv", index=False)
    h3_e.to_csv(OUT / "table4b_h3_effective.csv", index=False)

    print()
    print("=" * 80)
    print("TABLE 5 - Robustness: exclude October 10 cascade window")
    print("=" * 80)
    if "is_oct10_window" in df.columns:
        excl = df[~df["is_oct10_window"]].copy()
        print(f"\n  Sample after exclusion: {len(excl):,} obs ({len(df)-len(excl):,} dropped)")
        m, _ = h1_main(excl, "log_spread_primary", "h1_no_oct10")
        report(m, "H1 primary, excluding Oct 10")
        m, _ = h2_mechanism(excl, "log_spread_primary", "h2_no_oct10")
        if m is not None:
            term = "hl:log_realized_vol_bps"
            if term in m.params.index:
                c, s, t = m.params[term], m.bse[term], m.tvalues[term]
                print(f"    H2 interaction: {c:+.4f} (SE {s:.4f}, t={t:+.2f})")
    else:
        print("  Oct 10 flag not in data.")

    print()
    print("=" * 80)
    print("TABLE 6 - Robustness: exclude top 1% realized-vol minutes")
    print("=" * 80)
    rv_99 = df["realized_vol_bps"].quantile(0.99)
    excl_vol = df[df["realized_vol_bps"] <= rv_99].copy()
    print(f"\n  RV99 cutoff: {rv_99:.2f} bps; sample: {len(excl_vol):,} obs")
    m, _ = h1_main(excl_vol, "log_spread_primary", "h1_no_top_vol")
    report(m, "H1 primary, excluding top 1% vol")
    m, _ = h2_mechanism(excl_vol, "log_spread_primary", "h2_no_top_vol")
    if m is not None:
        term = "hl:log_realized_vol_bps"
        if term in m.params.index:
            c, s, t = m.params[term], m.bse[term], m.tvalues[term]
            print(f"    H2 interaction: {c:+.4f} (SE {s:.4f}, t={t:+.2f})")

    print()
    print("=" * 80)
    print("TABLE 7 - Binance adverse-selection decomposition")
    print("=" * 80)
    bn = df[df["venue"] == "binance"]
    if "price_impact_bps" in bn.columns:
        rows = []
        for asset in sorted(bn["asset"].unique()):
            s = bn[bn["asset"] == asset]
            rows.append({
                "asset": asset, "n": len(s),
                "median_eff_spread": s["effective_spread_bps"].median(),
                "median_realized_spread": s["realized_spread_bps"].median(),
                "median_price_impact": s["price_impact_bps"].median(),
            })
        adv = pd.DataFrame(rows).round(3)
        print(adv.to_string(index=False))
        adv.to_csv(OUT / "table7_adv_selection.csv", index=False)

    pd.DataFrame([h for h in all_h1 if h is not None]).to_csv(
        OUT / "h1_all_specifications.csv", index=False
    )
    print()
    print(f"All tables written to: {OUT}")


if __name__ == "__main__":
    main()
