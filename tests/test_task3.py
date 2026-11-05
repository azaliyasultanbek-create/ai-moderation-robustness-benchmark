"""Small tests for Task 3: normalisation, warm-up filtering, the paired test."""

import unittest

import pandas as pd

from src import process_data as pdx
from src.normalize import normalize


class TestTask3(unittest.TestCase):
    def test_normalize_fixes_common_tricks(self):
        self.assertEqual(normalize("1d10t"), "idiot")
        self.assertEqual(normalize("St.u*pid"), "stupid")
        self.assertEqual(normalize("іdіot"), "idiot")        # Cyrillic і
        self.assertEqual(normalize("traaaash"), "trash")

    def test_normalize_keeps_numbers_mentions_links(self):
        self.assertEqual(normalize("@user 2024 #news http://t.co/x"), "@user 2024 #news http://t.co/x")

    def test_warmup_rows_are_dropped(self):
        lat = pd.DataFrame({"phase": ["warmup", "eval", "eval"], "ms": [50.0, 1.0, 1.2], "n_items": [1, 1, 1]})
        ev, wu = pdx.drop_warmup(lat)
        self.assertEqual((len(ev), len(wu)), (2, 1))

    def test_paired_test_direction(self):
        r = pdx.paired_test("H1", "m", "x", "1111111111", "1111100000")   # 100 % -> 50 % recall
        self.assertAlmostEqual(r["delta_pp"], 50.0)
        self.assertEqual((r["b_lost"], r["c_gained"]), (5, 0))


if __name__ == "__main__":
    unittest.main()
