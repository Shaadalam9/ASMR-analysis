from pathlib import Path

ROOT = Path.cwd()


def replace_exact(path: Path, old: str, new: str, expected_count: int = 1) -> None:
    text = path.read_text(encoding='utf-8')
    count = text.count(old)
    if count == expected_count:
        path.write_text(text.replace(old, new), encoding='utf-8')
        print(f'Updated {path}')
        return
    if count == 0 and text.count(new) >= 1:
        print(f'Already updated: {path}')
        return
    raise RuntimeError(f'Could not safely update {path}: expected {expected_count} occurrence(s), found {count}.')


def main() -> None:
    clustering = ROOT / 'utils' / 'clustering_utils.py'
    readme = ROOT / 'README.md'
    tests = ROOT / 'tests' / 'test_reproducibility.py'
    summaries = ROOT / 'utils' / 'summaries.py'
    analysis = ROOT / 'analysis.py'

    for path in [clustering, readme, tests, summaries, analysis]:
        if not path.exists():
            raise RuntimeError(f'Missing {path}. Run this script from the ASMR-analysis repository root.')

    replace_exact(
        clustering,
        '                    SimpleImputer(\n                        strategy="median",\n                        add_indicator=True,\n                        keep_empty_features=True,\n                    ),',
        '                    SimpleImputer(\n                        strategy="median",\n                        keep_empty_features=True,\n                    ),',
    )

    replace_exact(
        clustering,
        '"""Median impute numeric features, retain missingness, then standardise."""',
        '"""Median impute numeric features, then standardise."""',
    )

    replace_exact(
        readme,
        '* Missing numeric clustering features are median imputed inside the fitted preprocessing pipeline, and missingness indicators are added before scaling. Missing values are therefore not treated as observed zeros.',
        '* Missing numeric clustering features are median imputed inside the fitted preprocessing pipeline before scaling. Missing values are therefore not treated as observed zeros and missingness itself is not used as a clustering feature.',
    )

    replace_exact(
        readme,
        'Before scaling, missing values in the three numeric clustering features are median imputed using statistics fitted on the clustering input, and missingness indicators are appended for numeric features that contain missing values. This prevents missing engagement, growth, or duration values from being treated as genuine zeros.',
        'Before scaling, missing values in the three numeric clustering features are median imputed using statistics fitted on the clustering input. Missingness indicators are not included in the feature space, so clusters are not formed merely because a metric is unavailable. This prevents missing engagement, growth, or duration values from being treated as genuine zeros.',
    )

    replace_exact(
        tests,
        'def test_numeric_preprocessing_uses_median_imputation_and_indicators(self):',
        'def test_numeric_preprocessing_uses_median_imputation_without_zero_fill(self):',
    )

    replace_exact(
        tests,
        '        self.assertGreater(transformed.shape[1], len(numeric_cols))\n        self.assertTrue(math.isnan(frame.loc[1, "duration_minutes"]))',
        '        self.assertEqual(transformed.shape[1], len(numeric_cols))\n        self.assertTrue(math.isnan(frame.loc[1, "duration_minutes"]))',
    )

    replace_exact(
        summaries,
        'df_copy.groupby("duration_bucket")',
        'df_copy.groupby("duration_bucket", observed=False)',
    )

    replace_exact(
        summaries,
        'df_copy.groupby("title_length_bucket")',
        'df_copy.groupby("title_length_bucket", observed=False)',
    )

    replace_exact(
        analysis,
        'df_copy.groupby([theme_col, "duration_bucket"])',
        'df_copy.groupby([theme_col, "duration_bucket"], observed=False)',
    )

    print('\nFollow-up fix applied successfully.')
    print('Run:')
    print('  uv run python -m unittest discover -s tests -v')
    print('  rm -f _output/analysis/asmr_videos_with_clusters*.pkl')
    print('  uv run python analysis.py')
    print('\nDo not use the cluster numbers from the previous run in the manuscript.')
    print('The rerun should no longer create clusters whose defining feature is missingness.')


if __name__ == '__main__':
    main()
