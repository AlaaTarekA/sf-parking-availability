"""Pilot step 1: turn Melbourne 15-second sensor snapshots into block-minute occupancy.

Data (public, not redistributed here):
  * Sensor snapshots, February 2019 (Parsonage, 2019):
      git clone https://github.com/HughParsonage/MelbourneParking2019Feb
  * Bay locations and road segments, City of Melbourne open data, via
      https://github.com/bgav114/SpatioTemporalAnalysis-ParkingSensorData
      (file "Parking Sensor Data/bay information.csv")

A "block" is a City of Melbourne road segment (rd_seg_id): one side of a street
between two cross streets. For each block and minute we record how many sensor
bays reported and how many were occupied.

    python src/pilot_prepare.py --archive <MelbourneParking2019Feb>/data-raw \
                                --bays "<path>/bay information.csv"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAYS = (1, 2, 4, 5, 7, 8)          # complete days in the archive (3 Feb is partial)
HOURS = range(7, 20)               # 07:00-19:59 local time
MIN_BAYS = 4                       # ignore blocks with too few sensors


def bay_table(path: Path) -> pd.DataFrame:
    bays = pd.read_csv(path, encoding="utf-8-sig", usecols=["the_geom", "bay_id", "rd_seg_id", "rd_seg_dsc"])
    bays = bays.dropna(subset=["bay_id", "rd_seg_id"]).drop_duplicates("bay_id")

    def centroid(geom: str) -> tuple[float, float]:
        nums = np.array(re.findall(r"-?\d+\.\d+", str(geom)), dtype=float)
        return nums[1::2].mean(), nums[0::2].mean()  # lat, lon

    lat_lon = bays["the_geom"].map(centroid)
    bays["lat"] = [p[0] for p in lat_lon]
    bays["lon"] = [p[1] for p in lat_lon]
    bays["bay_id"] = bays["bay_id"].astype(int)
    bays["rd_seg_id"] = bays["rd_seg_id"].astype(int)
    return bays[["bay_id", "rd_seg_id", "rd_seg_dsc", "lat", "lon"]]


def minute_files(archive: Path):
    for day in DAYS:
        for hour in HOURS:
            folder = archive / "2019" / "2" / str(day) / f"H{hour:02d}"
            for minute in range(60):
                f = folder / f"{hour:02d}{minute:02d}00.tsv"
                if f.exists():
                    yield pd.Timestamp(2019, 2, day, hour, minute), f


def main() -> None:
    parser = argparse.ArgumentParser(description="Build block-minute occupancy for the pilot.")
    parser.add_argument("--archive", required=True, help="MelbourneParking2019Feb/data-raw folder")
    parser.add_argument("--bays", required=True, help="'bay information.csv' file")
    parser.add_argument("--out", default=str(ROOT / "data" / "processed"))
    args = parser.parse_args()

    bays = bay_table(Path(args.bays))
    seg_of = bays.set_index("bay_id")["rd_seg_id"]
    frames, seen_bays, matched_bays = [], set(), set()
    for ts, path in minute_files(Path(args.archive)):
        snap = pd.read_csv(path, sep="\t", usecols=["Occupied", "bay_id"])
        seen_bays.update(snap["bay_id"].unique())
        snap["block"] = snap["bay_id"].map(seg_of)
        snap = snap.dropna(subset=["block"])
        matched_bays.update(snap["bay_id"].unique())
        agg = snap.groupby("block")["Occupied"].agg(bays="size", occupied="sum").reset_index()
        agg["time"] = ts
        frames.append(agg)
    panel = pd.concat(frames, ignore_index=True)
    panel["block"] = panel["block"].astype(int)

    # Keep blocks that are consistently instrumented.
    typical = panel.groupby("block")["bays"].median()
    keep = typical[typical >= MIN_BAYS].index
    panel = panel[panel["block"].isin(keep)].sort_values(["block", "time"])
    panel["free"] = panel["bays"] - panel["occupied"]

    blocks = bays[bays["rd_seg_id"].isin(keep)].groupby("rd_seg_id").agg(
        description=("rd_seg_dsc", "first"), lat=("lat", "mean"), lon=("lon", "mean"),
        mapped_bays=("bay_id", "size")).reset_index().rename(columns={"rd_seg_id": "block"})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "pilot_block_minutes.csv.gz", index=False)
    blocks.to_csv(out / "pilot_blocks.csv", index=False)
    print(f"snapshots: {panel['time'].nunique():,}  blocks: {len(keep):,}  rows: {len(panel):,}")
    print(f"sensor bays seen: {len(seen_bays):,}  matched to a block: {len(matched_bays):,}")


if __name__ == "__main__":
    main()
