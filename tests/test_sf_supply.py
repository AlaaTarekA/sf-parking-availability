"""Checks for the SF supply layer on a small synthetic inventory."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sf_supply  # noqa: E402

AREA = sf_supply.NEIGHBORHOOD


def meter(pid, block, color, lat, lon, flag="M", on="ON", area=AREA, jur="SFMTA"):
    return {"post_id": pid, "blockface_id": block, "cap_color": color, "latitude": lat,
            "longitude": lon, "active_meter_flag": flag, "on_offstreet_type": on,
            "analysis_neighborhood": area, "street_name": f"ST{block}", "jurisdiction": jur}


class SupplyTests(unittest.TestCase):
    def setUp(self):
        rows = [
            meter("a1", "1", "Grey", 37.7900, -122.4000),
            meter("a2", "1", "Grey", 37.7900, -122.4001),
            meter("a3", "1", "Yellow", 37.7900, -122.4002),
            meter("b1", "2", "Green", 37.7910, -122.4000),   # ~111 m north of block 1
            meter("c1", "3", "Red", 37.7930, -122.4000, jur="PORT"),  # no general spaces
            meter("d1", "4", "Grey", 37.7950, -122.4000),   # ~556 m from block 1
            meter("x1", "1", "Grey", 37.79, -122.40, flag="U"),       # inactive
            meter("x2", "1", "Grey", 37.79, -122.40, on="OFF"),       # off-street
            meter("x3", "9", "Grey", 37.77, -122.42, area="Mission"),  # other area
        ]
        self.meters = pd.DataFrame(rows).astype(str)

    def test_cleaning_and_classification(self):
        active, counts = sf_supply.clean(self.meters)
        self.assertEqual(counts["neighborhood_meters"], 8)
        self.assertEqual(counts["active_on_street_meters"], 6)
        self.assertEqual(counts["excluded_on_street_other_flags"], 1)
        self.assertEqual(active["use"].value_counts()["general"], 3)

    def test_block_supply_counts_only_usable_spaces(self):
        active, _ = sf_supply.clean(self.meters)
        blocks = sf_supply.block_supply(active).set_index("blockface_id")
        self.assertEqual(blocks.loc["1", "general_spaces"], 2)
        self.assertEqual(blocks.loc["2", "general_spaces"], 1)
        self.assertEqual(blocks.loc["3", "general_spaces"], 0)

    def test_choice_sets_use_distance_and_exclude_self(self):
        active, _ = sf_supply.clean(self.meters)
        blocks = sf_supply.block_supply(active)
        sets = sf_supply.choice_sets(blocks).set_index("blockface_id")
        self.assertEqual(sets.loc["1", "blocks_within_200m"], 1)   # block 2 only
        self.assertEqual(sets.loc["1", "spaces_within_200m"], 1)
        self.assertEqual(sets.loc["1", "blocks_within_400m"], 1)   # block 3 has no general spaces
        self.assertEqual(sets.loc["4", "blocks_within_400m"], 0)

    def test_summary_shares(self):
        active, counts = sf_supply.clean(self.meters)
        blocks = sf_supply.block_supply(active)
        summary = sf_supply.summarize(active, counts, blocks, sf_supply.choice_sets(blocks))
        self.assertEqual(summary["driver_usable_meters"], 4)
        self.assertEqual(summary["blocks_without_general"], 1)
        self.assertEqual(summary["port_managed_meters"], 1)


if __name__ == "__main__":
    unittest.main()
