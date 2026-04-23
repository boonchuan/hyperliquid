"""
01_pull_hyperliquid.py  v3 — parallel downloads, resumable
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import lz4.frame
import pandas as pd

ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
START_DATE = datetime(2025, 3, 1)
END_DATE   = datetime(2026, 2, 28)
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "hyperliquid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKET = "hyperliquid-archive"
REGION = "ap-northeast-1"
EXTRA  = {"RequestPayer": "requester"}

CFG = Config(region_name=REGION, max_pool_connections=50,
             retries={"max_attempts": 3, "mode": "standard"})
s3 = boto3.client("s3", config=CFG)


def fetch_one_hour(asset, date_str, hour, data_type):
    key = f"market_data/{date_str}/{hour}/{data_type}/{asset}.lz4"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key, **EXTRA)
        raw = obj["Body"].read()
    except ClientError as e:
        return []
    try:
        decompressed = lz4.frame.decompress(raw)
    except Exception:
        return []
    rows = []
    for line in decompressed.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def pull_day(asset, day, data_type):
    date_str = day.strftime("%Y%m%d")
    out_file = OUT_DIR / f"{asset}_{date_str}_{data_type}.parquet"
    if out_file.exists():
        return out_file, "cached"
    all_rows = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(fetch_one_hour, asset, date_str, h, data_type)
                   for h in range(24)]
        for f in as_completed(futures):
            all_rows.extend(f.result())
    if not all_rows:
        return None, "empty"
    df = pd.DataFrame(all_rows)
    df.to_parquet(out_file, compression="snappy")
    return out_file, "pulled"


def main():
    print(f"Pulling Hyperliquid: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Assets: {ASSETS}")
    print(f"Output: {OUT_DIR}")
    print("-" * 70)
    try:
        s3.list_objects_v2(Bucket=BUCKET, Prefix="market_data/20250301/0/",
                           MaxKeys=1, **EXTRA)
        print("AWS credentials verified.")
    except ClientError as e:
        print(f"AWS ERROR: {e}")
        sys.exit(1)

    total_days = (END_DATE - START_DATE).days + 1
    total_jobs = total_days * len(ASSETS)
    done_jobs = 0

    day = START_DATE
    while day <= END_DATE:
        for asset in ASSETS:
            t0 = time.time()
            try:
                out, status = pull_day(asset, day, "l2Book")
            except Exception as e:
                status = f"ERROR {e}"
                out = None
            dt = time.time() - t0
            done_jobs += 1
            pct = done_jobs / total_jobs * 100
            if out:
                size_kb = out.stat().st_size // 1024
                print(f"  [{done_jobs:>4}/{total_jobs}] ({pct:5.1f}%)  {day.date()} {asset:5s}: {status:7s}  {size_kb:>8,} KB  ({dt:.1f}s)")
            else:
                print(f"  [{done_jobs:>4}/{total_jobs}] ({pct:5.1f}%)  {day.date()} {asset:5s}: {status}")
        day += timedelta(days=1)
    print("-" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
