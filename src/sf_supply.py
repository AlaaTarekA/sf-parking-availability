"""Build the Downtown San Francisco parking supply layer.

Steps (Intermediate Report, Section 2):
1. Keep meters in the Financial District / South Beach neighborhood.
2. Keep active (flag "M") on-street meters.
3. Classify each meter by cap color: who may park there.
4. Count general spaces per block (blockface) and locate each block.
5. Compute distances between blocks and build 200 m / 400 m choice sets.

Input:  data/raw/sf_parking_meters.csv  (from src/download_sf_inventory.py)
Output: results/sf_supply_summary.json, results/sf_block_supply.csv
        (one row per block with its 200 m / 400 m choice set), results/figures/sf_*.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEIGHBORHOOD = "Financial District/South Beach"
CAP_USE = {
    "Grey": "general",
    "Green": "short-term",
    "Yellow": "commercial loading",
    "Red": "truck loading",
    "Black": "motorcycle",
}
DRIVER_USABLE = {"general", "short-term"}
RADII_M = (200, 400)


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Straight-line distance in metres between every pair of points."""
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def clean(meters: pd.DataFrame, neighborhood: str = NEIGHBORHOOD) -> tuple[pd.DataFrame, dict]:
    area = meters[meters["analysis_neighborhood"] == neighborhood].copy()
    on_street = area[area["on_offstreet_type"] == "ON"]
    active = on_street[on_street["active_meter_flag"] == "M"].copy()
    active["use"] = active["cap_color"].map(CAP_USE).fillna("other")
    active["latitude"] = pd.to_numeric(active["latitude"], errors="coerce")
    active["longitude"] = pd.to_numeric(active["longitude"], errors="coerce")
    active = active.dropna(subset=["latitude", "longitude", "blockface_id"])
    counts = {
        "neighborhood_meters": int(len(area)),
        "on_street_meters": int(len(on_street)),
        "excluded_on_street_other_flags": int(len(on_street) - (on_street["active_meter_flag"] == "M").sum()),
        "active_on_street_meters": int(len(active)),
        "blocks": int(active["blockface_id"].nunique()),
        "streets": int(active["street_name"].nunique()),
    }
    return active, counts


def block_supply(active: pd.DataFrame) -> pd.DataFrame:
    usable = active[active["use"].isin(DRIVER_USABLE)]
    blocks = active.groupby("blockface_id").agg(
        street=("street_name", "first"), meters=("post_id", "nunique"))
    blocks["general_spaces"] = usable.groupby("blockface_id")["post_id"].nunique()
    blocks["general_spaces"] = blocks["general_spaces"].fillna(0).astype(int)
    # Block location = average of its usable meters (all meters if it has none).
    loc_all = active.groupby("blockface_id")[["latitude", "longitude"]].mean()
    loc_use = usable.groupby("blockface_id")[["latitude", "longitude"]].mean()
    blocks[["latitude", "longitude"]] = loc_use.reindex(blocks.index).fillna(loc_all)
    return blocks.reset_index()


def choice_sets(blocks: pd.DataFrame, radii=RADII_M) -> pd.DataFrame:
    general = blocks[blocks["general_spaces"] > 0].reset_index(drop=True)
    dist = haversine_matrix(blocks["latitude"].to_numpy(), blocks["longitude"].to_numpy())
    gidx = blocks.index[blocks["general_spaces"] > 0].to_numpy()
    rows = []
    for i, block in blocks.iterrows():
        row = {"blockface_id": block["blockface_id"]}
        d = dist[i, gidx]
        others = gidx != i
        for r in radii:
            near = (d <= r) & others
            row[f"blocks_within_{r}m"] = int(near.sum())
            row[f"spaces_within_{r}m"] = int(general["general_spaces"].to_numpy()[near].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(active: pd.DataFrame, counts: dict, blocks: pd.DataFrame, sets: pd.DataFrame) -> dict:
    by_use = active["use"].value_counts().to_dict()
    usable = int(sum(by_use.get(u, 0) for u in DRIVER_USABLE))
    with_general = blocks[blocks["general_spaces"] > 0]
    s = sets[sets["blockface_id"].isin(with_general["blockface_id"])]
    out = dict(counts)
    out.update({
        "meters_by_use": {k: int(v) for k, v in by_use.items()},
        "driver_usable_meters": usable,
        "driver_usable_share": round(usable / len(active), 3),
        "blocks_without_general": int((blocks["general_spaces"] == 0).sum()),
        "median_general_spaces_per_block_with_any": float(with_general["general_spaces"].median()),
        "port_managed_meters": int(active["jurisdiction"].str.upper().str.contains("PORT", na=False).sum()) if "jurisdiction" in active else None,
    })
    for r in RADII_M:
        out[f"median_blocks_within_{r}m"] = float(s[f"blocks_within_{r}m"].median())
        out[f"median_spaces_within_{r}m"] = float(s[f"spaces_within_{r}m"].median())
        out[f"share_blocks_under_20_spaces_within_{r}m"] = round(float((s[f"spaces_within_{r}m"] < 20).mean()), 3)
    return out


def figures(active: pd.DataFrame, sets: pd.DataFrame, blocks: pd.DataFrame, figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"general": "#8C8C8C", "short-term": "#3C9D4E", "commercial loading": "#E2B100",
              "truck loading": "#C8382E", "motorcycle": "#222222", "other": "#B0A8D9"}
    figdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    for use, group in active.groupby("use"):
        ax.scatter(group["longitude"], group["latitude"], s=6, c=colors.get(use, "#999"),
                   label=f"{use} ({len(group):,})")
    ax.set_title("Active on-street meters by type of use\nFinancial District / South Beach")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend(markerscale=3, fontsize=8)
    ax.set_aspect(1 / np.cos(np.radians(37.79)))
    fig.tight_layout(); fig.savefig(figdir / "sf_meters_by_use_map.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    order = active["use"].value_counts()
    axes[0].barh(order.index[::-1], order.values[::-1], color=[colors.get(u, "#999") for u in order.index[::-1]])
    axes[0].set_title("Meters by type of use"); axes[0].set_xlabel("Active on-street meters")
    general = sets[sets["blockface_id"].isin(blocks.loc[blocks["general_spaces"] > 0, "blockface_id"])]
    axes[1].hist(general["spaces_within_400m"], bins=30, color="#1F8A80")
    axes[1].set_title("General spaces within 400 m of each block")
    axes[1].set_xlabel("General metered spaces"); axes[1].set_ylabel("Blocks")
    fig.tight_layout(); fig.savefig(figdir / "sf_supply_summary.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Downtown SF supply layer.")
    parser.add_argument("--inventory", default=str(ROOT / "data" / "raw" / "sf_parking_meters.csv"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()

    meters = pd.read_csv(args.inventory, dtype=str)
    active, counts = clean(meters)
    blocks = block_supply(active)
    sets = choice_sets(blocks)
    summary = summarize(active, counts, blocks, sets)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blocks.merge(sets, on="blockface_id").to_csv(out / "sf_block_supply.csv", index=False)
    (out / "sf_supply_summary.json").write_text(json.dumps(summary, indent=2))
    figures(active, sets, blocks, out / "figures")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
