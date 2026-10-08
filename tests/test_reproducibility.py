import math
import unittest

import numpy as np
import pandas as pd

import common
from asmr.analysis.clustering import Clustering
from asmr.collection.discover import ASMRFetcher
from asmr.processing.preprocessing import Preprocessing
from asmr.processing.text_tools import Tools
from asmr.visualization.figures import Plots
from asmr.visualization.summary_figures import theme_display_name


class ConfigurationTests(unittest.TestCase):
    def test_publication_reference_date_is_fixed(self):
        self.assertEqual(
            common.get_configs("analysis_reference_date"),
            "2026-09-01T00:00:00Z",
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
                "uploadDate": "2026-08-02T00:00:00Z",
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
            "2026-09-01T00:00:00+00:00",
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
                "uploadDate": "2026-09-02T00:00:00Z",
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


class ClusteringPreprocessingTests(unittest.TestCase):
    def test_numeric_preprocessing_uses_median_imputation_without_zero_fill(self):
        clustering = Clustering()
        frame = pd.DataFrame(
            {
                "duration_minutes": [10.0, None, 30.0],
                "engagement_rate": [0.10, 0.20, None],
                "views_per_day": [100.0, None, 300.0],
            }
        )

        numeric_cols = clustering._prepare_numeric_features(frame)
        pipeline = clustering._numeric_pipeline()
        transformed = pipeline.fit_transform(frame[numeric_cols])

        np.testing.assert_allclose(
            pipeline.named_steps["imputer"].statistics_,
            [np.log10([10.0, 30.0]).mean(), np.log10([0.10, 0.20]).mean(), np.log10([100.0, 300.0]).mean()],
        )
        self.assertFalse(np.isnan(np.asarray(transformed, dtype=float)).any())
        self.assertEqual(transformed.shape[1], len(numeric_cols))
        self.assertTrue(math.isnan(frame.loc[1, numeric_cols[0]]))

    def test_zero_duration_is_missing_not_a_measured_value(self):
        clustering = Clustering()
        frame = pd.DataFrame(
            {
                "duration_minutes": [0.0, 10.0],
                "engagement_rate": [0.1, 0.1],
                "views_per_day": [1.0, 1.0],
            }
        )

        numeric_cols = clustering._prepare_numeric_features(frame)
        self.assertTrue(math.isnan(frame.loc[0, numeric_cols[0]]))
        self.assertAlmostEqual(frame.loc[1, numeric_cols[0]], 1.0)
        self.assertEqual(frame.loc[0, "duration_minutes"], 0.0)


class VisualReproducibilityTests(unittest.TestCase):
    def test_wordcloud_layout_is_deterministic(self):
        plots = Plots()
        text = "asmr sleep whisper tapping relaxation " * 20

        first = plots.generate_wordcloud_image(text, set())
        second = plots.generate_wordcloud_image(text, set())

        self.assertTrue(np.array_equal(first, second))


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
