import math
import unittest

import numpy as np
import pandas as pd

from asmr.analysis.release import DICTIONARY
from asmr.analysis.supplementary.robustness import _subsets, _top_channels
from asmr.analysis.supplementary.statistics import gini, hhi, shannon, share_top
from asmr.collection.discover import _is_quota_error, _safe_exc_text
from asmr.processing.preprocessing import THEME_REGEX_PATTERNS


class ConcentrationMeasureTests(unittest.TestCase):
    def test_gini_is_zero_for_equal_values_and_near_one_for_extreme_inequality(self):
        self.assertAlmostEqual(gini(np.ones(100)), 0.0)
        self.assertGreater(gini(np.array([0.0] * 999 + [1000.0])), 0.99)

    def test_gini_of_empty_or_all_zero_input_is_nan(self):
        self.assertTrue(math.isnan(gini(np.array([]))))
        self.assertTrue(math.isnan(gini(np.zeros(5))))

    def test_share_top_returns_share_held_by_largest_units(self):
        values = np.array([70.0, 10.0, 10.0, 5.0, 5.0])
        self.assertAlmostEqual(share_top(values, 0.2), 0.70)

    def test_shannon_entropy_is_normalised(self):
        self.assertAlmostEqual(shannon(np.array([5, 5, 5, 5])), 1.0)
        self.assertAlmostEqual(shannon(np.array([10, 0, 0, 0])), 0.0)

    def test_hhi_is_one_for_a_single_category(self):
        self.assertAlmostEqual(hhi(np.array([42, 0, 0])), 1.0)
        self.assertAlmostEqual(hhi(np.array([1, 1, 1, 1])), 0.25)


class ApiErrorHandlingTests(unittest.TestCase):
    def test_quota_errors_are_recognised_from_status_or_text(self):
        class FakeResp:
            status = 429

        class FakeError(Exception):
            resp = FakeResp()

        self.assertTrue(_is_quota_error(FakeError("boom")))
        self.assertTrue(_is_quota_error(Exception("Quota exceeded for quota metric 'Search Queries'")))
        self.assertFalse(_is_quota_error(Exception("connection reset")))

    def test_api_keys_are_removed_from_logged_error_text(self):
        text = _safe_exc_text(Exception("GET https://x/y?q=ASMR&key=AIzaSECRET123&alt=json failed"))
        self.assertNotIn("AIzaSECRET123", text)
        self.assertIn("key=REDACTED", text)


if __name__ == "__main__":
    unittest.main()


class RobustnessSubsetTests(unittest.TestCase):
    def _frame(self):
        n = 200
        return pd.DataFrame({
            "video_id": [f"v{i}" for i in range(n)],
            "channel_id": [f"c{i % 100}" for i in range(n)],
            "views": np.arange(n, dtype=float),
            "upload_year": [2010 + (i % 15) for i in range(n)],
            "language": ["English" if i % 3 else "Korean" for i in range(n)],
        })

    def test_top_channels_by_views_returns_the_top_one_percent(self):
        df = self._frame()
        top = _top_channels(df, "views")
        self.assertEqual(len(top), 1)
        self.assertEqual(top, {"c99"})

    def test_subsets_partition_periods_and_languages(self):
        subsets = dict(_subsets(self._frame()))
        periods = ["Uploaded 2008-2015", "Uploaded 2016-2019", "Uploaded 2020-2026"]
        self.assertEqual(sum(len(subsets[p]) for p in periods), len(subsets["All videos"]))
        self.assertEqual(len(subsets["English only"]) + len(subsets["Non-English only"]), len(subsets["All videos"]))
        self.assertLess(len(subsets["Without top 1% of channels by views"]), len(subsets["All videos"]))


class ReleaseTests(unittest.TestCase):
    def test_dictionary_variables_are_unique_and_cover_all_themes(self):
        names = [row[0] for row in DICTIONARY]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(set(THEME_REGEX_PATTERNS) <= set(names))

    def test_theme_rules_match_expected_phrases(self):
        import re
        hit = lambda col, text: bool(re.search(THEME_REGEX_PATTERNS[col], text, re.IGNORECASE))  # noqa: E731
        self.assertTrue(hit("has_whisper", "Whispering for sleep"))
        self.assertTrue(hit("has_sleep", "ASMR for sleep"))
        self.assertFalse(hit("has_whisper", "A loud video"))
