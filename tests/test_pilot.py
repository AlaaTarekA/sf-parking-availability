"""Checks that pilot examples use only the past and are split by day."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pilot_train as pt  # noqa: E402


def synthetic(days=(1, 7, 8), blocks=3):
    idx = pd.DatetimeIndex([t for d in days for t in pd.date_range(f"2019-02-{d:02d} 07:00",
                                                                   f"2019-02-{d:02d} 08:59", freq="min")])
    rng = np.random.default_rng(0)
    free = pd.DataFrame(rng.integers(0, 3, size=(len(idx), blocks)), index=idx, columns=range(blocks))
    bays = pd.DataFrame(5, index=idx, columns=range(blocks))
    return free.astype(float), bays.astype(float)


class PilotTests(unittest.TestCase):
    def test_target_is_future_and_features_are_past(self):
        free, bays = synthetic()
        h = 10
        data = pt.build_examples(free, bays, h)
        for _, row in data.sample(50, random_state=1).iterrows():
            t, b = row["time"], row["block"]
            self.assertEqual(row["free_now"], free.loc[t, b])
            self.assertEqual(row["free_future"], free.loc[t + pd.Timedelta(minutes=h), b])
            self.assertEqual(row["y"], int(free.loc[t + pd.Timedelta(minutes=h), b] >= 1))
            past = free.loc[t - pd.Timedelta(minutes=15):t, b]
            self.assertEqual(row["min_free_15"], past.min())

    def test_examples_never_cross_days(self):
        free, bays = synthetic()
        data = pt.build_examples(free, bays, 15)
        arrival = data["time"] + pd.Timedelta(minutes=15)
        self.assertTrue((arrival.dt.day == data["day"]).all())

    def test_history_uses_training_days_only(self):
        free, bays = synthetic(days=(1, 8))
        data = pt.build_examples(free, bays, 5)
        before = pt.add_history(data.copy())["hist_rate"]
        changed = data.copy()
        changed.loc[changed["day"] == 8, "y"] = 1 - changed.loc[changed["day"] == 8, "y"]
        after = pt.add_history(changed)["hist_rate"]
        self.assertTrue(np.allclose(before, after))

    def test_split_days_are_disjoint(self):
        self.assertFalse(set(pt.TRAIN_DAYS) & set(pt.CAL_DAYS))
        self.assertFalse(set(pt.TRAIN_DAYS) & set(pt.TEST_DAYS))
        self.assertFalse(set(pt.CAL_DAYS) & set(pt.TEST_DAYS))


if __name__ == "__main__":
    unittest.main()
