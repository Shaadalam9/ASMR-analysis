# ASMR YouTube analysis

This repository contains the collection, enrichment, analysis, and visualisation code for:

> Alam, M. S., and Bazilinskyy, P. (2026). *Nineteen years of ASMR on YouTube: A multilingual, theme-level analysis of 89,241 videos*. PLOS ONE manuscript PONE-D-26-27689, under revision.

The study describes 89,241 explicitly labelled ASMR videos uploaded between 1 January 2008 and 31 July 2026. Metadata were processed on 1 August 2026, which is also the fixed reference date for the publication analysis.

## Scope

The code supports four tasks:

1. Discover videos using the YouTube Data API and the public YouTube search interface through `pytubefix`.
2. Enrich and recover video metadata.
3. Derive descriptive measures, title features, and textual theme labels.
4. Produce summary tables, figures, K means clusters, and a full corpus UMAP projection.

This is a descriptive platform metadata study. The code does not infer viewer wellbeing, sleep quality, subjective ASMR experiences, or causal effects.

## Repository structure

| Path | Purpose |
| --- | --- |
| `main.py` | Windowed video discovery and initial metadata collection |
| `JSONMetadataEnricher.py` | Optional refresh of existing metadata and video statistics |
| `recover_from_seen_ids.py` | Recovery of identifiers missing from the main JSON file |
| `analysis.py` | Publication analysis entry point |
| `utils/preprocessing.py` | Derived measures, title features, language normalisation, and theme labels |
| `utils/summaries.py` | Descriptive summary tables and temporal counts |
| `utils/clustering_utils.py` | TF IDF features, K means, PCA, t SNE, and UMAP |
| `utils/viz_core.py` | Common figure generation and export |
| `default.config` | Versioned publication defaults |
| `tests/` | Deterministic unit tests that do not require YouTube access |

## Reproducibility safeguards

The revision package includes the following safeguards:

* `analysis_reference_date` is fixed to `2026-08-01T00:00:00Z`. Views per day and likes per day therefore do not change simply because the analysis is rerun later.
* Theme detection defaults to the deterministic rule based method used for the reported manuscript results. It no longer changes silently according to whether a local spaCy model is installed.
* `force_recompute` defaults to `true`, so stale pickle files cannot silently determine publication results.
* Randomised algorithms use the configured seed of 42.
* Every analysis run writes `_output/analysis/reproducibility_manifest.json`, containing the input data SHA256 digest, record count, analysis settings, Python version, and package versions.
* The reproducibility manifest reports coverage of the optional `metadataCollectedAt` and `languageSource` provenance fields. The deposited publication snapshot does not retain these record level fields, so the manifest reports zero timestamps and an unknown language source while preserving the existing language labels.
* Local configuration, credentials, raw data, caches, and generated output are excluded from version control.

## Requirements

* Python 3.12
* [`uv`](https://docs.astral.sh/uv/)

Install `uv`, clone the repository, and create the locked environment:

```bash
git clone https://github.com/Shaadalam9/ASMR-analysis.git
cd ASMR-analysis
uv sync --frozen
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

For exact reproduction of the paper, use the deposited 89,241 video snapshot rather than recollecting current YouTube values. The publication snapshot has SHA256 digest `e260339147d2af21c7fc796d73f26e2f382cc29385443bcee65d189c34eccb18`. YouTube statistics and availability change over time.

## Configuration

The versioned defaults in `default.config` reproduce the publication settings. A local `config` file is optional. When present, it only needs to contain values that override the defaults.

| Setting | Publication value | Meaning |
| --- | --- | --- |
| `data` | `data` | Directory containing `asmr_results.json` |
| `query` | `ASMR` | Search term and required title substring |
| `analysis_text_source` | `both` | Use concatenated titles and descriptions |
| `date_before` | `null` | Optional collection upper date bound |
| `date_after` | `null` | Optional collection lower date bound |
| `date_window_months` | `3` | YouTube API search window size |
| `analysis_reference_date` | `2026-08-01T00:00:00Z` | Fixed denominator date for daily rates |
| `theme_detection_mode` | `rule_based` | Deterministic textual theme method |
| `theme_rule_version` | `1.0.0` | Version recorded in output and caches |
| `force_recompute` | `true` | Rebuild publication outputs instead of trusting caches |
| `refresh_existing_statistics` | `false` | Preserve deposited values unless an explicit refresh is requested |
| `random_seed` | `42` | Seed for clustering and sampling |
| `clustering_n_clusters` | `11` | K used for the reported exploratory solution |
| `auto_open_plots` | `false` | Do not open browser windows during batch runs |

Example local override:

```json
{
  "data": "/path/to/deposited/data"
}
```

## Credentials

Credentials must never be committed or included in a shared ZIP file. For collection or metadata refresh, set one of these environment variables:

```bash
export YOUTUBE_API_KEY="your_key"
```

or, for key rotation:

```bash
export YOUTUBE_API_KEYS="key_one,key_two"
```

The scripts retain backwards compatibility with an ignored local `secret` JSON file, but environment variables are recommended.

## Reproduce the publication analysis

With the deposited JSON snapshot in `data/asmr_results.json`, run:

```bash
uv run python analysis.py
```

The main machine readable outputs are written to `_output/analysis/`. Publication copies of figures are also written to `figures/`.

Important outputs include:

* `reproducibility_manifest.json`
* `asmr_videos_enriched.csv`
* `duration_stats.csv`
* `language_stats.csv`
* `title_style_stats.csv`
* `theme_flag_counts.csv`
* theme trend tables
* cluster summaries and full UMAP coordinates

Run the deterministic tests with:

```bash
uv run python -m unittest discover -s tests -v
```

## Collect a new dataset

Set valid date bounds in a local `config` file and provide a YouTube API key. Then run:

```bash
uv run python main.py
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

Do not run `JSONMetadataEnricher.py`, `main.py`, or `recover_from_seen_ids.py` when reproducing the publication analysis from the deposited snapshot. These scripts can change the corpus or its metadata. For exact reproduction, run only the tests and `analysis.py`.

To preserve deposited values, `refresh_existing_statistics` is `false` by default. To deliberately refresh views and likes for all existing records, create a local override:

```json
{
  "refresh_existing_statistics": true
}
```

Then run:

```bash
uv run python JSONMetadataEnricher.py
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

The selected source is stored in `languageSource` when available. The deposited 89,241 record publication snapshot retains the language labels but not their record level source field; the reproducibility manifest therefore reports the source as `unknown`. Language detection from short, mixed language, or creator supplied text can be inaccurate and should be interpreted as metadata level classification.
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
* standardised duration, engagement rate, and views per day;
* one hot encoded language labels.

K means uses \(k=11\), seed 42, and ten initialisations. UMAP is a visual projection of the fitted feature space, not a definitive taxonomy of ASMR subgenres. The two dimensional UMAP uses 50 TruncatedSVD components, 30 neighbours, minimum distance 0.1, cosine distance, and seed 42.

## Licence

The code is released under the MIT Licence. The YouTube data remain subject to the terms and policies applicable to their source and repository deposition.
