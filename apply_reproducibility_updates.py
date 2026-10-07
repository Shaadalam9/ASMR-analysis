from pathlib import Path


ROOT = Path.cwd()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def replace_exact(path: Path, old: str, new: str, expected_count: int = 1) -> None:
    text = read_text(path)
    count = text.count(old)

    if count == expected_count:
        write_text(path, text.replace(old, new))
        print(f"Updated {path}")
        return

    if count == 0 and text.count(new) >= 1:
        print(f"Already updated: {path}")
        return

    raise RuntimeError(
        f"Could not safely update {path}: expected {expected_count} occurrence(s), "
        f"found {count}. The repository may have changed."
    )


def update_readme() -> None:
    path = ROOT / "README.md"

    replace_exact(
        path,
        """git clone https://github.com/Shaadalam9/ASMR-analysis.git
cd ASMR-analysis
uv sync --frozen""",
        """git clone https://github.com/Shaadalam9/ASMR-analysis.git
cd ASMR-analysis
uv sync --frozen
cp default.config config""",
    )

    replace_exact(
        path,
        """* Randomised algorithms use the configured seed of 42.
* Every analysis run writes `_output/analysis/reproducibility_manifest.json`, containing the input data SHA256 digest, record count, analysis settings, Python version, and package versions.""",
        """* Randomised algorithms and word cloud layouts use the configured seed of 42.
* Missing numeric clustering features are median imputed inside the fitted preprocessing pipeline, and missingness indicators are added before scaling. Missing values are therefore not treated as observed zeros.
* Normality diagnostics are written to `_output/analysis/normality_tests.csv`. The D'Agostino Pearson test uses all positive view counts, while the Shapiro Wilk test uses at most 5,000 observations sampled deterministically with the configured seed.
* Every analysis run writes `_output/analysis/reproducibility_manifest.json`, containing the input data SHA256 digest, record count, analysis settings, Python version, and package versions.""",
    )

    replace_exact(
        path,
        """The versioned defaults in `default.config` reproduce the publication settings. A local `config` file is optional. When present, it only needs to contain values that override the defaults.""",
        """The versioned defaults in `default.config` reproduce the publication settings. The current configuration loader expects a local `config` file containing the complete set of configuration keys. For a reproducible local run, copy `default.config` to `config` and change only values that must differ locally, such as the dataset directory.""",
    )

    replace_exact(
        path,
        """Example local override:

```json
{
  "data": "/path/to/deposited/data"
}
```""",
        """Create the local configuration from the versioned defaults:

```bash
cp default.config config
```

Then edit only the values that need to differ locally. For example, change the `data` entry in `config` to the directory containing the deposited dataset while leaving the publication analysis settings unchanged.""",
    )

    replace_exact(
        path,
        """* `theme_flag_counts.csv`
* theme trend tables""",
        """* `theme_flag_counts.csv`
* `normality_tests.csv`
* theme trend tables""",
    )

    replace_exact(
        path,
        """* standardised duration, engagement rate, and views per day;
* one hot encoded language labels.

K means uses \\(k=11\\), seed 42, and ten initialisations.""",
        """* standardised duration, engagement rate, and views per day;
* one hot encoded language labels.

Before scaling, missing values in the three numeric clustering features are median imputed using statistics fitted on the clustering input, and missingness indicators are appended for numeric features that contain missing values. This prevents missing engagement, growth, or duration values from being treated as genuine zeros.

K means uses \\(k=11\\), seed 42, and ten initialisations.""",
    )


def update_pyproject() -> None:
    path = ROOT / "pyproject.toml"

    replace_exact(
        path,
        'description = "Add your description here"',
        'description = "Reproducible collection and analysis pipeline for a multilingual longitudinal study of ASMR on YouTube"',
    )

    replace_exact(
        path,
        '    "scikit-learn>=1.7.2",\n    "spacy>=3.8.11",',
        '    "scikit-learn>=1.7.2",\n    "scipy>=1.16.3",\n    "spacy>=3.8.11",',
    )


def update_uv_lock() -> None:
    path = ROOT / "uv.lock"

    replace_exact(
        path,
        '    { name = "scikit-learn" },\n    { name = "spacy" },',
        '    { name = "scikit-learn" },\n    { name = "scipy" },\n    { name = "spacy" },',
        expected_count=1,
    )

    replace_exact(
        path,
        '    { name = "scikit-learn", specifier = ">=1.7.2" },\n    { name = "spacy", specifier = ">=3.8.11" },',
        '    { name = "scikit-learn", specifier = ">=1.7.2" },\n    { name = "scipy", specifier = ">=1.16.3" },\n    { name = "spacy", specifier = ">=3.8.11" },',
        expected_count=1,
    )


def update_analysis_manifest() -> None:
    path = ROOT / "analysis.py"

    replace_exact(
        path,
        '        "scikit-learn",\n        "spacy",',
        '        "scikit-learn",\n        "scipy",\n        "spacy",',
    )


def update_clustering() -> None:
    path = ROOT / "utils" / "clustering_utils.py"

    replace_exact(
        path,
        "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.pipeline import Pipeline",
        "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.pipeline import Pipeline",
    )

    replace_exact(
        path,
        """class Clustering_utils():
    def __init__(self) -> None:
        pass

    def cluster_videos""",
        """class Clustering_utils():
    def __init__(self) -> None:
        pass

    @staticmethod
    def _prepare_numeric_features(df: pd.DataFrame) -> list[str]:
        \"\"\"Coerce clustering numerics while preserving missing values for imputation.\"\"\"
        numeric_cols = [\"duration_minutes\", \"engagement_rate\", \"views_per_day\"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors=\"coerce\")

        missing_counts = df[numeric_cols].isna().sum()
        if int(missing_counts.sum()) > 0:
            logger.info(
                \"Clustering numeric missing values before median imputation:\\n\"
                + missing_counts.to_string()
            )
        return numeric_cols

    @staticmethod
    def _numeric_pipeline() -> Pipeline:
        \"\"\"Median impute numeric features, retain missingness, then standardise.\"\"\"
        return Pipeline(
            steps=[
                (
                    \"imputer\",
                    SimpleImputer(
                        strategy=\"median\",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                (\"scaler\", StandardScaler(with_mean=False)),
            ]
        )

    def cluster_videos""",
    )

    replace_exact(
        path,
        """        for col in ["duration_minutes", "engagement_rate", "views_per_day"]:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0.0)""",
        """        numeric_cols = self._prepare_numeric_features(df_copy)""",
        expected_count=4,
    )

    replace_exact(
        path,
        """                    StandardScaler(with_mean=False),
                    ["duration_minutes", "engagement_rate", "views_per_day"],""",
        """                    self._numeric_pipeline(),
                    numeric_cols,""",
        expected_count=4,
    )

    replace_exact(
        path,
        """        # Ensure numeric columns are numeric
        numeric_cols = self._prepare_numeric_features(df_copy)

        # ColumnTransformer identical to cluster_videos""",
        """        # Preserve missing numeric values for fitted median imputation.
        numeric_cols = self._prepare_numeric_features(df_copy)

        # ColumnTransformer identical to cluster_videos""",
    )


def update_viz_core() -> None:
    path = ROOT / "utils" / "viz_core.py"

    replace_exact(
        path,
        """            stopwords=stopwords,
            collocations=False,
        ).generate(text)""",
        """            stopwords=stopwords,
            collocations=False,
            random_state=int(common.get_configs("random_seed")),
        ).generate(text)""",
    )

    replace_exact(
        path,
        """            stopwords=stopwords,
            collocations=False,
        ).generate_from_frequencies(frequencies)""",
        """            stopwords=stopwords,
            collocations=False,
            random_state=int(common.get_configs("random_seed")),
        ).generate_from_frequencies(frequencies)""",
    )

    replace_exact(
        path,
        """        log_views = np.log10(views)

        logger.info("===== LOG10(VIEWS) DISTRIBUTION ANALYSIS =====")""",
        """        log_views = np.log10(views)
        random_seed = int(common.get_configs("random_seed"))

        logger.info("===== LOG10(VIEWS) DISTRIBUTION ANALYSIS =====")""",
    )

    replace_exact(
        path,
        """        sample = log_views
        max_n_shapiro = 5000
        if len(sample) > max_n_shapiro:
            sample = sample.sample(max_n_shapiro, random_state=42)  # type: ignore

        w_stat, p_shapiro = stats.shapiro(sample)
        logger.info(
            f"Shapiro–Wilk: W = {w_stat:.3f}, p-value = {p_shapiro:.3g} "
            "(H0: data come from a normal distribution)"
        )

        logger.info(
            "Interpretation: if p-values are << 0.05, log10(views) deviates from a "
            "perfect Gaussian; larger p-values mean you cannot reject normality."
        )""",
        """        max_n_shapiro = 5000
        shapiro_sampled = len(log_views) > max_n_shapiro
        sample = (
            log_views.sample(max_n_shapiro, random_state=random_seed)  # type: ignore
            if shapiro_sampled
            else log_views
        )

        w_stat, p_shapiro = stats.shapiro(sample)
        shapiro_sampling = (
            f"simple random sample without replacement from N={len(log_views)}"
            if shapiro_sampled
            else "full positive-view sample"
        )
        logger.info(
            f"Shapiro–Wilk (n={len(sample)}; {shapiro_sampling}): "
            f"W = {w_stat:.3f}, p-value = {p_shapiro:.3g} "
            "(H0: data come from a normal distribution)"
        )

        analysis_dir = os.path.join(common.output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        normality_results = pd.DataFrame(
            [
                {
                    "test": "D'Agostino-Pearson",
                    "statistic": float(k2),
                    "p_value": float(p_normaltest),
                    "n": int(len(log_views)),
                    "sampling": "all positive view counts",
                    "random_seed": None,
                },
                {
                    "test": "Shapiro-Wilk",
                    "statistic": float(w_stat),
                    "p_value": float(p_shapiro),
                    "n": int(len(sample)),
                    "sampling": shapiro_sampling,
                    "random_seed": random_seed if shapiro_sampled else None,
                },
            ]
        )
        normality_path = os.path.join(analysis_dir, "normality_tests.csv")
        normality_results.to_csv(normality_path, index=False)
        logger.info(f"Normality diagnostics saved to {normality_path}")

        logger.info(
            "Interpretation: if p-values are << 0.05, log10(views) deviates from a "
            "perfect Gaussian; larger p-values mean you cannot reject normality."
        )""",
    )


def update_tests() -> None:
    path = ROOT / "tests" / "test_reproducibility.py"

    replace_exact(
        path,
        """import math
import unittest

import common""",
        """import math
import unittest

import numpy as np
import pandas as pd

import common""",
    )

    replace_exact(
        path,
        """from main import ASMRFetcher
from utils.preprocessing import Preprocessing
from utils.tool import Tools
from utils.viz_summaries import theme_display_name""",
        """from main import ASMRFetcher
from utils.clustering_utils import Clustering_utils
from utils.preprocessing import Preprocessing
from utils.tool import Tools
from utils.viz_core import Plots
from utils.viz_summaries import theme_display_name""",
    )

    replace_exact(
        path,
        """class ThemeAndDurationTests(unittest.TestCase):""",
        """class ClusteringPreprocessingTests(unittest.TestCase):
    def test_numeric_preprocessing_uses_median_imputation_and_indicators(self):
        clustering = Clustering_utils()
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
            [20.0, 0.15, 200.0],
        )
        self.assertFalse(np.isnan(np.asarray(transformed, dtype=float)).any())
        self.assertGreater(transformed.shape[1], len(numeric_cols))
        self.assertTrue(math.isnan(frame.loc[1, "duration_minutes"]))


class VisualReproducibilityTests(unittest.TestCase):
    def test_wordcloud_layout_is_deterministic(self):
        plots = Plots()
        text = "asmr sleep whisper tapping relaxation " * 20

        first = plots.generate_wordcloud_image(text, set())
        second = plots.generate_wordcloud_image(text, set())

        self.assertTrue(np.array_equal(first, second))


class ThemeAndDurationTests(unittest.TestCase):""",
    )


def write_ci() -> None:
    path = ROOT / ".github" / "workflows" / "ci.yml"
    content = """name: CI

on:
  push:
  pull_request:

jobs:
  tests:
    runs-on: ubuntu-latest

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Set up uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true

      - name: Create local configuration
        run: cp default.config config

      - name: Install locked dependencies
        run: uv sync --frozen

      - name: Run deterministic tests
        run: uv run python -m unittest discover -s tests -v
"""
    write_text(path, content)
    print(f"Updated {path}")


def main() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "analysis.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "utils" / "clustering_utils.py",
        ROOT / "utils" / "viz_core.py",
        ROOT / "tests" / "test_reproducibility.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Run this script from the ASMR-analysis repository root. Missing: "
            + ", ".join(missing)
        )

    update_readme()
    update_pyproject()
    update_uv_lock()
    update_analysis_manifest()
    update_clustering()
    update_viz_core()
    update_tests()
    write_ci()

    print("\nUpdates applied successfully.")
    print("Next run:")
    print("  uv sync --frozen")
    print("  uv run python -m unittest discover -s tests -v")
    print("  uv run python analysis.py")
    print("\nBecause clustering now uses median imputation plus missingness indicators,")
    print("rerun the clustering outputs and update manuscript cluster values if they change.")


if __name__ == "__main__":
    main()
