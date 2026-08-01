import math
import unittest

import common
from main import ASMRFetcher
from utils.preprocessing import Preprocessing
from utils.tool import Tools
from utils.viz_summaries import theme_display_name


class ConfigurationTests(unittest.TestCase):
    def test_publication_reference_date_is_fixed(self):
        self.assertEqual(
            common.get_configs("analysis_reference_date"),
            "2026-08-01T00:00:00Z",
        )

    def test_placeholder_collection_dates_are_unset(self):
        fetcher = ASMRFetcher.__new__(ASMRFetcher)
        self.assertIsNone(fetcher._normalize_published_bound("YYYY-MM-DD"))
        self.assertIsNone(fetcher._normalize_published_bound("none"))
        self.assertEqual(
            fetcher._normalize_published_bound("2026-08-01"),
            "2026-08-01T00:00:00Z",
        )


class DerivedMeasureTests(unittest.TestCase):
    def setUp(self):
        self.preprocessing = Preprocessing()

    def test_daily_rates_use_fixed_reference_date(self):
        data = {
            "video_one": {
                "title": "ASMR whisper for sleep",
                "description": "No talking binaural roleplay",
                "language": "en",
                "views": 300,
                "likes": 30,
                "duration": 600,
                "uploadDate": "2026-07-02T00:00:00Z",
            }
        }

        frame = self.preprocessing.json_to_dataframe(data)
        row = frame.iloc[0]

        self.assertAlmostEqual(row["days_since_upload"], 30.0)
        self.assertAlmostEqual(row["views_per_day"], 10.0)
        self.assertAlmostEqual(row["likes_per_day"], 1.0)
        self.assertAlmostEqual(row["engagement_rate"], 0.1)
        self.assertEqual(
            row["analysis_reference_date"],
            "2026-08-01T00:00:00+00:00",
        )

    def test_future_upload_has_no_daily_rate(self):
        data = {
            "video_two": {
                "title": "ASMR example",
                "description": "",
                "language": "en",
                "views": 100,
                "likes": 5,
                "duration": 600,
                "uploadDate": "2026-08-02T00:00:00Z",
            }
        }

        frame = self.preprocessing.json_to_dataframe(data)
        self.assertTrue(math.isnan(frame.iloc[0]["days_since_upload"]))
        self.assertTrue(math.isnan(frame.iloc[0]["views_per_day"]))

    def test_zero_views_produce_missing_engagement(self):
        data = {
            "video_three": {
                "title": "ASMR example",
                "description": "",
                "language": "en",
                "views": 0,
                "likes": 0,
                "duration": 600,
                "uploadDate": "2026-04-01T00:00:00Z",
            }
        }

        frame = self.preprocessing.json_to_dataframe(data)
        self.assertTrue(math.isnan(frame.iloc[0]["engagement_rate"]))

    def test_unrecognised_language_code_is_not_collapsed_to_missing(self):
        self.assertEqual(
            self.preprocessing.normalize_language_code("x-custom"),
            "x-custom",
        )
        self.assertEqual(
            self.preprocessing.normalize_language_code(None),
            "Unknown",
        )


class ThemeAndDurationTests(unittest.TestCase):
    def test_theme_display_names_hide_internal_column_names(self):
        self.assertEqual(theme_display_name("has_roleplay"), "role play")
        self.assertEqual(theme_display_name("has_no_talking"), "no talking")
        self.assertEqual(theme_display_name("drive"), "driving")
        self.assertEqual(theme_display_name("has_custom_theme"), "custom theme")

    def test_publication_theme_rules_are_deterministic(self):
        preprocessing = Preprocessing()
        data = {
            "video_four": {
                "title": "ASMR whisper no talking sleep binaural roleplay",
                "description": (
                    "ear cleaning mukbang keyboard visual triggers driving"
                ),
                "language": "en",
                "views": 10,
                "likes": 1,
                "duration": 600,
                "uploadDate": "2026-04-01T00:00:00Z",
            }
        }

        frame = preprocessing.json_to_dataframe(data)
        theme_columns = [
            "has_whisper",
            "has_no_talking",
            "has_sleep",
            "has_binaural",
            "has_roleplay",
            "has_ear_cleaning",
            "has_mukbang",
            "has_keyboard",
            "has_visual",
            "has_drive",
        ]

        self.assertTrue(frame.loc[0, theme_columns].all())
        self.assertEqual(
            frame.loc[0, "theme_detection_method"],
            "english_lexical_rules",
        )
        self.assertEqual(frame.loc[0, "theme_rule_version"], "1.0.0")

    def test_duration_boundaries_match_documentation(self):
        tools = Tools()
        self.assertEqual(tools._duration_bucket(9.99), "under_10min")
        self.assertEqual(tools._duration_bucket(10), "10_to_30min")
        self.assertEqual(tools._duration_bucket(30), "30_to_60min")
        self.assertEqual(tools._duration_bucket(60), "60_to_180min")
        self.assertEqual(tools._duration_bucket(180), "over_180min")

    def test_collection_threshold_retains_59_seconds(self):
        fetcher = ASMRFetcher.__new__(ASMRFetcher)
        self.assertTrue(fetcher._is_short_video(58))
        self.assertFalse(fetcher._is_short_video(59))


if __name__ == "__main__":
    unittest.main()
