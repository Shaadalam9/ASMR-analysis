# ASMR on YouTube: a multilingual, longitudinal analysis

Code for collecting, enriching and analysing metadata of explicitly labelled ASMR videos on YouTube, and for reproducing the figures, tables and supplementary analyses of:

> Alam, M. S., and Bazilinskyy, P. (2026). *Nineteen years of ASMR on YouTube: A multilingual, theme-level analysis of 90,189 videos*. Manuscript in preparation.

This is a descriptive platform-metadata study. It describes what is produced and how attention is distributed. It does not measure viewers, sleep quality, wellbeing, subjective ASMR experiences, or causal effects.

[![Word cloud of the 200 most frequent words in ASMR video titles](figures/wordcloud_title.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/wordcloud_title.html)

## What we did

1. **Collected** 90,189 videos whose titles contain "ASMR", uploaded between 1 January 2008 and 31 August 2026 (18,126 channels, 81 language categories), using the YouTube Data API in three-month windows plus `pytubefix` search as a supplementary route.
2. **Enriched** the records with duration, views, likes, upload date, language labels, title-style features and ten rule-based textual theme labels.
3. **Described** the corpus: language, duration, engagement, themes and monthly growth.
4. **Measured** how supply diversified over time and how concentrated attention is (Gini coefficients, top shares, Herfindahl-Hirschman index, entropy).
5. **Compared** how duration, themes, language and title style relate to views per day and engagement, with pooled models, models with upload-year fixed effects, and models with channel fixed effects, and checked those results in subsets of the corpus.
6. **Audited** the text labels and the clustering for sensitivity to language, text field, scaling and the number of clusters.

```mermaid
flowchart LR
    A[asmr.collection.discover<br/>discover videos] --> B[asmr.collection.enrich_metadata<br/>refresh metadata]
    B --> C[(data/asmr_results.json)]
    A --> C
    R[asmr.collection.recover_ids<br/>recover missing IDs] --> C
    C --> D[asmr.analysis.publication<br/>derived measures, themes,<br/>tables, figures, clusters, UMAP]
    D --> E[(_output/analysis/<br/>enriched dataset)]
    E --> F[asmr.analysis.supplementary<br/>label audit, diversification, models,<br/>duration reversal, robustness, cluster diagnostics]
    D --> G[figures/]
    F --> G
```

## Main findings

All numbers come from the deposited snapshot and the fixed reference date of 1 September 2026. The figures behind each finding are in the [figure gallery](#figure-gallery).

**The genre broadened and diversified.** Active channels grew from 285 (2008 to 2011) to 8,248 (2020 to 2023). The share of uploads from the most prolific 1% of channels fell from 83.9% to 18.9%. The non-English share rose from 6.3% to about one third, and the normalised entropy of theme labels rose from 0.65 to 0.93, also when computed for English videos only (`ecosystem_diversification`).

**Attention is extremely concentrated.** The Gini coefficient of views across videos is 0.92. The top 1% of videos hold 49.4% of all views, and the top 1% of channels hold 67.3%. The median video gets 21.9 views per day against a mean of 1,038 (`ecosystem_diversification`, `duration_vs_views`).

**Means mislead.** Videos under 10 minutes have the highest mean views per day (1,853) but the lowest median (1.2), a mean-to-median ratio of about 1,600 (`duration_reversal`, panel a). The ranking of themes also changes between means and medians.

**Pooled comparisons confound format with creator.** In a pooled model, videos under 10 minutes have about 76% fewer views per day than 10 to 30 minute videos. Within the same channel the difference is small (about 5% in the full sample), and videos over 180 minutes have about 130% more views per day. The same pattern holds on the videos and channels used by the within-channel model (`duration_reversal`, panel b).

**The within-channel advantage of longer videos is stable.** It is present for English and non-English videos, in all three upload periods, and after excluding the channels with the most views or uploads. The within-channel result for short videos is less stable and varies between about -29% and +12% across subsets (`duration_robustness`).

**Text labels are language dependent.** The English-lexical rules find at least one theme in 54.5% of English videos but only 35.7% of non-English videos. The most common theme depends on whether descriptions are searched (sleep for full text, role play for titles). Extending two rules to ten other languages more than doubles their prevalence among non-English videos without changing their growth profile.

**Cluster structure is weak.** After log-scaling the numeric inputs, k-means is best at k = 4 (silhouette 0.29, Davies-Bouldin 1.14, resampling ARI 0.995). The clusters separate mainly by attention regime and duration, not by subgenre (`kmeans_k_diagnostics`, `cluster_umap_full_static_umap_full`).

## Figure gallery

> **Note:** The interactive figures are displayed using `htmlpreview.github.io`, which may be blocked on some institutional or education networks. If an interactive figure does not open, try another network or download the corresponding `.html` file from the [`figures/`](figures/) folder and open it in a browser. Plotly.js is loaded from a CDN, so an internet connection is needed.

Every figure is made with Plotly and written to [`figures/`](figures/) as interactive HTML, PNG and EPS. Click a picture to open the interactive version.

### Corpus overview

[![Word cloud of video titles](figures/wordcloud_title.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/wordcloud_title.html)
Word cloud of the 200 most frequent words in the titles of all videos, after stopword removal and text cleaning. Font size is proportional to corpus frequency.

[![Word cloud of video descriptions](figures/wordcloud_description.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/wordcloud_description.html)
Word cloud of the 200 most frequent words in the video descriptions, after stopword removal and text cleaning.

[![Word cloud of titles and descriptions](figures/wordcloud_both.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/wordcloud_both.html)
Word cloud of the concatenation of title and description, which is the text used for the theme labels and the clustering.

[![Monthly video counts](figures/monthly_video_counts.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/monthly_video_counts.html)
Number of videos uploaded per month. *Note:* the most recent months are thinly sampled because newly uploaded videos are retrieved less completely.

[![Duration against views](figures/duration_vs_views.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/duration_vs_views.html)
Log-log density heatmap (70 by 70 bins) of video duration against cumulative views. Colour is the logarithm of the number of videos in a cell. Videos with missing or zero duration or views cannot be shown on logarithmic axes.

[Duration against views, interactive scatter](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/duration_vs_views_interactive.html)
Interactive version of the duration-against-views plot for browsing individual videos. *Note:* HTML only; the image links to the interactive file.

[![Q-Q plot of log10 views](figures/log_views_qq_plot.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/log_views_qq_plot.html)
Quantile-quantile plot of log10 views against the normal distribution. Departures in both tails show that even the logarithm of views is not normally distributed.


### Supply and concentration of attention

[![Structure of supply and attention over time](figures/ecosystem_diversification.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/ecosystem_diversification.html)
Four panels: (a) uploads and active channels per year (log scale); (b) non-English share of uploads and number of language categories; (c) normalised theme entropy and mean number of theme labels per video; (d) Lorenz curves of cumulative views across videos and channels with Gini coefficients. *Note:* 2026 covers January to August only.


### Language and title style

[![Language growth over the years](figures/language_growth_over_years.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/language_growth_over_years.html)
Uploads per year for the larger language categories.

[![Mean views per day by language](figures/language_mean_views_per_day.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/language_mean_views_per_day.html)
Mean views per day for language categories with at least 20 videos. *Note:* very small categories are noisy.

[![Mean engagement rate by language](figures/language_mean_engagement_rate.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/language_mean_engagement_rate.html)
Mean engagement rate (likes divided by views) for language categories with at least 20 videos.

[![Title length and views](figures/title_length_mean_views.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/title_length_mean_views.html)
Mean views by title-length bucket (up to 5, 6 to 10, 11 to 20, and over 20 words).

[![Title length and engagement](figures/title_length_mean_engagement_rate.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/title_length_mean_engagement_rate.html)
Mean engagement rate by the same title-length buckets.


### Theme labels over time

[![Whisper videos per year](figures/has_whisper_trend_overall_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_whisper_trend_overall_fig.html)
Number of videos per upload year whose title or description matches the whisper rule.

[![No talking videos per year](figures/has_no_talking_trend_overall_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_no_talking_trend_overall_fig.html)
Number of videos per upload year whose title or description matches the no talking rule.

[![Sleep related videos per year](figures/has_sleep_trend_overall_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_sleep_trend_overall_fig.html)
Number of videos per upload year whose title or description matches the sleep related rule.

[![Binaural / 3d audio videos per year](figures/has_binaural_trend_overall_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_binaural_trend_overall_fig.html)
Number of videos per upload year whose title or description matches the binaural / 3D audio rule.

[![Role play videos per year](figures/has_roleplay_trend_overall_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_roleplay_trend_overall_fig.html)
Number of videos per upload year whose title or description matches the role play rule.

[![Ear cleaning / ear focus videos per year](figures/has_ear_cleaning_trend_overall_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_ear_cleaning_trend_overall_fig.html)
Number of videos per upload year whose title or description matches the ear cleaning / ear focus rule.

[![Mukbang / eating videos per year](figures/has_mukbang_trend_overall_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_mukbang_trend_overall_fig.html)
Number of videos per upload year whose title or description matches the mukbang / eating rule.

[![Keyboard / typing videos per year](figures/has_keyboard_trend_overall_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_keyboard_trend_overall_fig.html)
Number of videos per upload year whose title or description matches the keyboard / typing rule.

[![Visual triggers videos per year](figures/has_visual_trend_overall_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_visual_trend_overall_fig.html)
Number of videos per upload year whose title or description matches the visual triggers rule.

[![Driving videos per year](figures/has_drive_trend_overall_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_drive_trend_overall_fig.html)
Number of videos per upload year that match the driving rule.

[![Driving videos per year (legacy name)](figures/drive_trend_overall_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/drive_trend_overall_fig.html)
Copy of the driving trend under the legacy file name used by earlier runs, kept for compatibility.


### Theme labels and attention

[![Views per day with and without whisper](figures/has_whisper_views_per_day_boxplot.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_whisper_views_per_day_boxplot.html)
Views per day for videos with and without the whisper label. Compare medians rather than means, because views per day is heavily skewed.

[![Views per day with and without no talking](figures/has_no_talking_views_per_day_boxplot.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_no_talking_views_per_day_boxplot.html)
Views per day for videos with and without the no talking label. Compare medians rather than means, because views per day is heavily skewed.

[![Views per day with and without sleep related](figures/has_sleep_views_per_day_boxplot.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_sleep_views_per_day_boxplot.html)
Views per day for videos with and without the sleep related label. Compare medians rather than means, because views per day is heavily skewed.

[![Views per day with and without binaural / 3D audio](figures/has_binaural_views_per_day_boxplot.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_binaural_views_per_day_boxplot.html)
Views per day for videos with and without the binaural / 3D audio label. Compare medians rather than means, because views per day is heavily skewed.

[![Views per day with and without driving](figures/has_drive_views_per_day_boxplot.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_drive_views_per_day_boxplot.html)
Views per day for videos with and without the driving label. Compare medians rather than means, because views per day is heavily skewed.


### Format, creators and attention (duration)

[![Means versus medians and pooled versus within-channel duration associations](figures/duration_reversal.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/duration_reversal.html)
(a) Mean and median views per day by duration bucket (log scale): short videos have the highest mean and the lowest median. (b) Ratio of views per day relative to videos of 10 to 30 minutes from three models: pooled on all videos, pooled on the videos used by the within-channel model, and within channel with channel and year fixed effects. Error bars are 95% confidence intervals from channel-clustered standard errors.

[![Stability of the duration associations](figures/duration_robustness.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/duration_robustness.html)
The same pooled and within-channel ratios, one panel per duration bucket, in eight subsets: all videos, three upload periods, English and non-English videos, and the corpus without the 1% of channels with the most views or the most uploads.


### Clusters (exploratory)

[![K-means elbow curve](figures/kmeans_elbow_both.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/kmeans_elbow_both.html)
Inertia against the number of clusters k. Shown for completeness; k was not chosen from the elbow.

[![Cluster-number diagnostics](figures/kmeans_k_diagnostics.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/kmeans_k_diagnostics.html)
(a) Mean silhouette coefficient, (b) Davies-Bouldin index and (c) stability under 80% subsampling (mean adjusted Rand index) for k = 2 to 20. The dotted line marks the reported k = 4.

[![Cluster sizes, UMAP run](figures/cluster_sizes_umap.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/cluster_sizes_umap.html)
Number of videos in each of the four clusters (UMAP run).

[![Mean views per day by cluster, UMAP run](figures/cluster_mean_views_per_day_umap.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/cluster_mean_views_per_day_umap.html)
Mean views per day in each cluster (UMAP run).

[![UMAP scatter coloured by cluster](figures/cluster_scatter_embedding_umap.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/cluster_scatter_embedding_umap.html)
Sampled two-dimensional UMAP projection of the feature space (text, duration, engagement, views per day and language), coloured by k-means cluster, with circles around the clusters.

[![Research-style UMAP](figures/cluster_umap_research_umap_research.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/cluster_umap_research_umap_research.html)
UMAP projection with labelled clusters and ellipses, downsampled for the browser.

[![UMAP of all videos](figures/cluster_umap_full_static_umap_full.png)](figures/cluster_umap_full_static_umap_full.png)
Static UMAP of all 90,189 videos coloured by cluster. *Note:* PNG and EPS only, because it is too large for an interactive version.

[![Cluster sizes, PCA run](figures/cluster_sizes_pca.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/cluster_sizes_pca.html)
Number of videos per cluster in the run whose two-dimensional embedding is PCA.

[![Mean views per day by cluster, PCA run](figures/cluster_mean_views_per_day_pca.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/cluster_mean_views_per_day_pca.html)
Mean views per day per cluster in the PCA-embedding run.

[![PCA scatter coloured by cluster](figures/cluster_scatter_embedding_pca.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/cluster_scatter_embedding_pca.html)
Sampled two-dimensional PCA projection coloured by k-means cluster.

## Repository structure

```text
main.py                         Entry point for video discovery (used by the cron job)
analysis.py                     Entry point for the publication analysis
common.py                       Paths, configuration (`config`, `default.config`), secrets and shared helpers
custom_logger.py, logmod.py     Logging helpers (`str.format`-style logger and log set-up)
asmr/                           Python package (run modules with `python -m ...`)
  collection/
    discover.py                 Windowed video discovery and initial metadata collection
    enrich_metadata.py          Optional refresh of existing metadata and video statistics
    recover_ids.py              Recovery of identifiers missing from the main JSON file
  processing/
    preprocessing.py            Derived measures, title features, language normalisation, theme labels
    keywords.py                 Keyword and lemma analysis, word-cloud pipelines
    text_tools.py               Stopwords and text normalisation helpers
  analysis/
    publication.py              Publication pipeline: measures, tables, figures, clustering, UMAP
    summaries.py                Descriptive summary tables and temporal counts
    clustering.py               Shared TF-IDF preprocessor, k-means, PCA and UMAP
    supplementary/              Supplementary analyses (one module per stage)
      measurement.py            Theme-label audit
      ecosystem.py              Diversification and concentration of attention
      models.py                 Multivariable models with year and channel fixed effects
      reversal.py               Means versus medians and pooled versus within-channel duration figure
      robustness.py             Stability of the duration associations across subsets
      cluster_diagnostics.py    Cluster-number diagnostics and stability
      statistics.py             Gini, shares, entropy and HHI
      shared.py                 Paths, data loading and figure/table export
  visualization/
    figures.py                  Plotly figure generation and export
    summary_figures.py          Summary and trend figures
tests/                          Deterministic unit tests that do not require YouTube access
figures/                        Generated figures (HTML, PNG, EPS); all figures are made with Plotly
default.config                  Versioned publication defaults
pyproject.toml, uv.lock         Dependencies and lint configuration
```

## Reproducibility safeguards

* `analysis_reference_date` is fixed to `2026-09-01T00:00:00Z`, so views per day and likes per day do not change simply because the analysis is rerun later.
* Theme detection defaults to the deterministic rule-based method. It does not change according to whether a local spaCy model is installed.
* `force_recompute` defaults to `true`, so stale pickle files cannot silently determine results.
* Randomised algorithms and word cloud layouts use the configured seed of 42.
* Clustering uses log10-scaled duration, engagement rate and views per day. Missing numeric values are median imputed inside the fitted pipeline, and missingness is not used as a feature.
* Normality diagnostics are written to `_output/analysis/normality_tests.csv`.
* Every analysis run writes `_output/analysis/reproducibility_manifest.json` with the input data SHA256 digest, record count, settings, Python version, and package versions.
* Local configuration, credentials, raw data, caches, and generated output are excluded from version control.
* The collection code redacts API keys from logged search errors, rotates keys on quota errors (HTTP 429), and stops early when both the Data API and `pytubefix` are unavailable instead of looping through empty windows.

## Requirements

* Python 3.12
* [`uv`](https://docs.astral.sh/uv/)

Install `uv`, clone the repository, and create the locked environment:

```bash
git clone https://github.com/Shaadalam9/ASMR-analysis.git
cd ASMR-analysis
uv sync --frozen
cp default.config config
```

The repository contains `.python-version`, `pyproject.toml`, and `uv.lock`. The frozen installation should therefore use the same dependency resolution as the publication release.

## Data

The code archive does not contain API credentials or the research dataset. Place the deposited dataset at:

```text
data/asmr_results.json
```

The expected structure is a JSON object keyed by YouTube video identifier:

```json
{
  "VIDEO_ID": {
    "title": "Example ASMR title",
    "description": "Example description",
    "duration": 1200,
    "channelId": "CHANNEL_ID",
    "author": "Channel name",
    "views": 100000,
    "likes": 4000,
    "uploadDate": "2020-01-01T12:00:00Z",
    "language": "en",
    "languageSource": null,
    "channel_average_views": 85000.0,
    "metadataCollectedAt": null
  }
}
```

For exact reproduction of the reported results, use the deposited 90,189 video snapshot rather than recollecting current YouTube values. The snapshot used for the reported results has SHA256 digest `b1a25ec911d05fbcea923e74bf66b33cda03be3bf8088ae79b3a08516395ca41` (also written to `_output/analysis/reproducibility_manifest.json` on every run). YouTube statistics and availability change over time.

## Configuration of project

Configuration of the project needs to be defined in `config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. Create a local copy with:

```bash
cp default.config config
```

The `config` file is ignored by git. The loader stops with an error if `config` has fewer parameters than `default.config`. The config file has the following parameters:
- **`data`**: Directory containing the dataset (`asmr_results.json` and the collection state files).
- **`query`**: Search term used for collection. The lower-cased term must also appear in the video title for a video to be kept.
- **`analysis_text_source`**: Text used for the text-based analyses (theme labels, word clouds, clustering). Can be: `title`, `description` or `both`.
- **`date_before`**: Optional upper bound (YYYY-MM-DD) for the upload dates collected. `null` means up to today.
- **`date_after`**: Optional lower bound (YYYY-MM-DD) for the upload dates collected. `null` means from the start of the collection window.
- **`date`**: Older single-date setting used as the upper bound when `date_before` is `null`.
- **`date_window_months`**: Size in months of the upload-date windows that the YouTube API search is split into, to get around the cap of 5,000 results per search.
- **`analysis_reference_date`**: Fixed date used as the denominator for daily rates (views per day, likes per day), so that rates are reproducible.
- **`theme_detection_mode`**: Method for the textual theme labels. `rule_based` uses deterministic regular expressions. `spacy` is optional, exploratory, and fails explicitly if the model is unavailable.
- **`theme_rule_version`**: Version of the theme rules, recorded in the outputs and caches.
- **`force_recompute`**: Rebuild outputs instead of trusting cached files (good for reproducing results; set to `false` to reuse caches).
- **`refresh_existing_statistics`**: Whether the enrichment step may overwrite existing view and like counts. Keep `false` to preserve the deposited values.
- **`random_seed`**: Seed for clustering, UMAP and sampling.
- **`clustering_n_clusters`**: Number of k-means clusters used in the reported solution (4). Chosen with `python -m asmr.analysis.supplementary cluster`.
- **`auto_open_plots`**: Set to `true` to open every saved HTML figure in the browser. Keep `false` for batch runs.
- **`font_family`**: Font family used in the figures.
- **`font_size`**: Font size used in the figures.
- **`plotly_template`**: Template for Plotly figures.
- **`logger_level`**: Level of console output. Can be: debug, info, warning, error.

### The `secret` file

API keys live in a file named `secret` (no extension) in the root of the project. It is ignored by git, so it is never committed. To create it, copy the template `default.secret` to `secret` and fill in the values you need:

```bash
cp default.secret secret
```

The file is a single JSON object:

```json
{
    "google-api-keys": ["key_one", "key_two"]
}
```

| Key | Used by | What for | Where to get it |
|---|---|---|---|
| `google-api-keys` | `asmr.collection.discover`, `asmr.collection.enrich_metadata`, `asmr.collection.recover_ids` | One or more YouTube Data API v3 keys. Several keys are rotated automatically when the daily quota of one key is used up. | Google Cloud Console; see below |

Which keys you need depends on what you run:
- **reproducing the analysis from the deposited dataset** (`analysis.py`, `asmr.analysis.supplementary`): none.
- **collecting videos** (`main.py`, i.e. `asmr.collection.discover`): `google-api-keys`.
- **refreshing metadata** (`asmr.collection.enrich_metadata`) and **recovering missing identifiers** (`asmr.collection.recover_ids`): `google-api-keys`. As an alternative, `enrich_metadata` reads the environment variable `YOUTUBE_API_KEY` and `recover_ids` reads `YOUTUBE_API_KEYS` (comma separated) or `YOUTUBE_API_KEY`.

#### Getting a YouTube Data API key

The key is free. In the [Google Cloud Console](https://console.cloud.google.com/), create a project, enable the *YouTube Data API v3*, and create an API key under *Credentials*. The default quota is 10,000 units a day per project, and one search request costs 100 units, so a single key covers roughly 100 search pages a day. Collecting the whole history takes many requests, which is why several keys are supported. When the daily quota is used up, the collector logs a quota error, moves to the next key and, when all keys are exhausted, stops early instead of retrying. Never commit the `secret` file or paste keys into issues or logs.

## Reproduce the publication analysis

With the deposited JSON snapshot in `data/asmr_results.json`, run:

```bash
uv run python -m asmr.analysis.publication
```

The main machine readable outputs are written to `_output/analysis/`. Publication copies of figures are also written to `figures/`.

The supplementary analyses (theme-label audit, diversification and concentration measures, multivariable models, duration robustness and cluster-number diagnostics) are produced from the enriched dataset written by `asmr.analysis.publication`, so run that first:

```bash
uv run python -m asmr.analysis.supplementary            # all stages
uv run python -m asmr.analysis.supplementary models     # or: measurement, ecosystem, models, models_title_only, cluster, reversal, robustness
```

Outputs are written to `_output/analysis/extra/` and `figures/`. The `cluster` stage fits k means for k = 2 to 20 and takes about ten to twenty minutes.

Important outputs include:

* `reproducibility_manifest.json`
* `asmr_videos_enriched.csv`
* `duration_stats.csv`
* `language_stats.csv`
* `title_style_stats.csv`
* `theme_flag_counts.csv`
* `normality_tests.csv`
* theme trend tables
* cluster summaries and full UMAP coordinates

Run the deterministic tests with:

```bash
uv run python -m unittest discover -s tests -v
```

## Collect a new dataset

Set valid date bounds in a local `config` file and provide a YouTube API key. Then run:

```bash
uv run python -m asmr.collection.discover
```

The API branch partitions the configured period into consecutive three month windows. The `pytubefix` branch uses YouTube search relevance ordering and is therefore a supplementary discovery route rather than a guarantee of complete platform coverage.

A record is retained when:

* its title contains the configured query as a case insensitive substring;
* it is a standard video result;
* its known duration is at least 59 seconds;
* it passes the configured upload date bounds; and
* its video identifier is not already present.

Search results are not a census of all YouTube content. API limits, ranking, removed videos, private videos, missing fields, and platform changes affect recall.

## Create or refresh a metadata snapshot

Do not run `asmr.collection.enrich_metadata`, `asmr.collection.discover`, or `asmr.collection.recover_ids` when reproducing the publication analysis from the deposited snapshot. These scripts can change the corpus or its metadata. For exact reproduction, run only the tests and `asmr.analysis.publication`.

To preserve deposited values, `refresh_existing_statistics` is `false` by default. To deliberately refresh views and likes for all existing records, create a local override:

```json
{
  "refresh_existing_statistics": true
}
```

Then run:

```bash
uv run python -m asmr.collection.enrich_metadata
```

Archive the resulting JSON once the refresh finishes and report the actual retrieval period. A refresh performed after publication will not reproduce the original numerical results because YouTube metrics change continuously.

## Implemented measures

For video \(v\), with the fixed reference date used to calculate age:

```text
views_per_day(v) = views(v) / days_since_upload(v)
likes_per_day(v) = likes(v) / days_since_upload(v)
engagement_rate(v) = likes(v) / views(v)
```

Engagement is missing when views are zero or missing. Group means are calculated over the videos with a valid value for the relevant measure.

Duration groups are:

* under 10 minutes
* 10 to under 30 minutes
* 30 to under 60 minutes
* 60 to under 180 minutes
* 180 minutes or longer
* unknown

## Language labels

The collection code uses the following order:

1. YouTube `defaultAudioLanguage`
2. YouTube `defaultLanguage`
3. deterministic `langdetect` prediction from the concatenated title and description

The selected source is stored in `languageSource` when available. The deposited 90,189 record publication snapshot retains the language labels but not their record level source field; the reproducibility manifest therefore reports the source as `unknown`. Language detection from short, mixed language, or creator supplied text can be inaccurate and should be interpreted as metadata level classification.
Known code aliases are normalised to shared display names. Unrecognised nonempty codes are preserved rather than being merged with genuinely missing language values.

## Theme labels

The reported themes are Boolean textual indicators derived from lowercased titles and descriptions. The default rules match English lexical forms for:

* whisper
* no talking
* sleep
* binaural or spatial audio
* role play
* ear focused content
* mukbang or eating
* keyboard or typing
* visual triggers
* driving

These are not manually verified audiovisual content categories. The dataset is multilingual, but the reported theme rules are primarily English lexical rules. Consequently, prevalence can be underestimated for creators who use equivalent labels only in other languages. The exact regular expressions are versioned in `utils/preprocessing.py`.

The optional `spacy` mode is retained for exploratory work. It is not the publication default and fails explicitly if the requested model is unavailable.

## Exploratory clustering

The clustering input combines:

* up to 5,000 TF IDF title and description unigram and bigram features, with minimum document frequency 5;
* log10-transformed and standardised duration, engagement rate, and views per day (rates are floored at 1e-4, and non-positive durations are treated as missing);
* one hot encoded language labels.

Before scaling, missing values in the three numeric clustering features are median imputed using statistics fitted on the clustering input. Missingness indicators are not included in the feature space, so clusters are not formed merely because a metric is unavailable. This prevents missing engagement, growth, or duration values from being treated as genuine zeros.

K means uses \(k=4\) (selected using silhouette, Davies Bouldin, and resampling stability diagnostics; see `python -m asmr.analysis.supplementary cluster`), seed 42, and ten initialisations. The numeric inputs are log transformed because raw duration, engagement, and views per day are heavy tailed and otherwise let a handful of extreme videos define whole clusters. UMAP is a visual projection of the fitted feature space, not a definitive taxonomy of ASMR subgenres. The two dimensional UMAP uses 50 TruncatedSVD components, 30 neighbours, minimum distance 0.1, cosine distance, and seed 42.

## Supplementary analyses (`asmr.analysis.supplementary`)

Run after `asmr.analysis.publication`. Each stage writes CSV tables to `_output/analysis/extra/`.

| Stage | Question | Main outputs |
| --- | --- | --- |
| `measurement` | How sensitive are the theme labels to text field and language? | `theme_title_vs_full_text.csv`, `theme_detection_english_vs_non_english.csv`, `theme_multilingual_lexicon_sensitivity.csv`, `theme_validation_sample_to_annotate.csv` |
| `ecosystem` | How did supply diversify and how concentrated is attention? | `ecosystem_by_year.csv`, `ecosystem_by_period.csv`, `channel_entrants_by_year.csv`, figure `ecosystem_diversification` |
| `models` | How do duration, themes, language and title style relate to views per day and engagement, with year and channel fixed effects? | `model_coefficients.csv`, `model_block_r2.csv` |
| `models_title_only` | Do the model results hold when themes are defined from titles only? | `model_coefficients_title_only_themes.csv` |
| `cluster` | Which number of clusters is supported, and how stable is it? | `kmeans_k_diagnostics.csv`, `kmeans_per_cluster_silhouette.csv`, figure `kmeans_k_diagnostics` |
| `reversal` | Do means and medians, and pooled and within-channel models, disagree about duration? (run after `models`) | `duration_mean_vs_median.csv`, figure `duration_reversal` |
| `robustness` | Do the duration associations hold across periods, languages and after excluding prolific channels? | `duration_robustness.csv`, figure `duration_robustness` |

`theme_validation_sample_to_annotate.csv` holds 20 flagged and 20 unflagged videos per theme for manual annotation. No manual precision or recall figure is reported yet.

## Licence

The code is released under the MIT Licence. The YouTube data remain subject to the terms and policies applicable to their source and repository deposition.
