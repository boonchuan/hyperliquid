"""
02_pull_binance.py  v6 — pyarrow-thread-safe, full year, resumable
"""

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

# Pre-touch pyarrow + pandas extension registration on the main thread BEFORE
# any worker threads start. Avoids the "type extension already defined" race.
import pyarrow as pa
_warmup = pd.DataFrame({"x": pd.period_range("2025-01-01", periods=2, freq="D")})
_warmup.to_parquet("/dev/null" if False else str(Path.cwd() / "_warmup.parquet"))
Path("_warmup.parquet").unlink(missing_ok=True)

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
START_DATE = datetime(2025, 3, 1)
END_DATE   = datetime(2026, 2, 28)
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "binance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://data.binance.vision/data/futures/um/daily"

# Lock around the parquet write to fully eliminate any remaining race
WRITE_LOCK = Lock()


def download_zip_to_parquet(url, out_file):
    try:
        r = requests.get(url, timeout=180)
    except Exception as e:
        return None, f"CONN-ERR {e}"
    if r.status_code == 404:
        return None, "404"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)
        with WRITE_LOCK:
            df.to_parquet(out_file, compression="snappy")
    except Exception as e:
        return None, f"PARSE {e}"
    return out_file, "pulled"


def pull_one(symbol, day, kind):
    date_str = day.strftime("%Y-%m-%d")
    out_file = OUT_DIR / f"{symbol}_{date_str}_{kind}.parquet"
    if out_file.exists():
        return kind, symbol, day, out_file, "cached"
    url = f"{BASE}/{kind}/{symbol}/{symbol}-{kind}-{date_str}.zip"
    out, status = download_zip_to_parquet(url, out_file)
    return kind, symbol, day, out, status


def main():
    print(f"Pulling Binance: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Assets: {ASSETS}")
    print("-" * 70)

    jobs = []
    day = START_DATE
    while day <= END_DATE:
        for sym in ASSETS:
            for kind in ["bookDepth", "aggTrades"]:
                jobs.append((sym, day, kind))
        day += timedelta(days=1)

    total = len(jobs)
    done = 0
    pulled = 0
    cached = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(pull_one, *j) for j in jobs]
        for f in as_completed(futures):
            kind, sym, d, out, status = f.result()
            done += 1
            pct = done / total * 100
            if status == "pulled":
                pulled += 1
            elif status == "cached":
                cached += 1
            else:
                failed += 1
            if out and hasattr(out, "stat"):
                size_kb = out.stat().st_size // 1024
                print(f"  [{done:>5}/{total}] ({pct:5.1f}%)  {d.date()} {sym:9s} {kind:10s}: {status:7s}  {size_kb:>8,} KB")
            else:
                print(f"  [{done:>5}/{total}] ({pct:5.1f}%)  {d.date()} {sym:9s} {kind:10s}: {status}")

    print("-" * 70)
    print(f"Done. pulled={pulled} cached={cached} failed={failed}")


if __name__ == "__main__":
    main()
