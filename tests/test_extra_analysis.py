import math
import unittest

import numpy as np

from asmr.analysis.supplementary.statistics import gini, hhi, shannon, share_top
from asmr.collection.discover import _is_quota_error, _safe_exc_text


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
