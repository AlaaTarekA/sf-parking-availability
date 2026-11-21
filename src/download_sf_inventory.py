"""Download the SFMTA Parking Meters inventory from DataSF.

Run this on your own computer (the DataSF API must be reachable):

    python src/download_sf_inventory.py

It saves data/raw/sf_parking_meters.csv (about 38,000 rows, ~37 MB).
Source: San Francisco Municipal Transportation Agency, Parking Meters,
https://data.sfgov.org/d/8vzz-qzz9
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
API = "https://data.sfgov.org/resource/8vzz-qzz9.csv"
PAGE = 50_000


def fetch_page(offset: int) -> pd.DataFrame:
    query = urllib.parse.urlencode({"$limit": PAGE, "$offset": offset, "$order": "objectid"})
    with urllib.request.urlopen(f"{API}?{query}", timeout=120) as response:
        return pd.read_csv(io.BytesIO(response.read()), dtype=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(RAW / "sf_parking_meters.csv"))
    args = parser.parse_args()

    pages, offset = [], 0
    while True:
        page = fetch_page(offset)
        pages.append(page)
        print(f"downloaded {len(page):,} rows (offset {offset:,})")
        if len(page) < PAGE:
            break
        offset += PAGE
        time.sleep(1)
    meters = pd.concat(pages, ignore_index=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meters.to_csv(out, index=False)
    manifest = {
        "source": "https://data.sfgov.org/d/8vzz-qzz9",
        "rows": len(meters),
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_as_of": sorted(meters.get("data_as_of", pd.Series(dtype=str)).dropna().unique())[-1:],
    }
    (out.parent / "sf_inventory_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved {len(meters):,} meters to {out}")


if __name__ == "__main__":
    main()
