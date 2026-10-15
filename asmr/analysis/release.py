"""Build the public dataset release (files for a repository deposit such as Zenodo).

Usage (after ``python -m asmr.analysis.publication`` has written the enriched dataset)::

    python -m asmr.analysis.release

Writes ``release/`` with the video table, a data dictionary, the theme rules, the reproducibility manifest and
SHA-256 checksums. Nothing is uploaded.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sys
import zipfile

import pandas as pd

import common
from asmr.processing.preprocessing import THEME_REGEX_PATTERNS
from custom_logger import CustomLogger

logger = CustomLogger(__name__)

RELEASE_DIR = os.path.join(common.root_dir, "release")
ENRICHED = os.path.join(common.output_dir, "analysis", "asmr_videos_enriched.csv")
MANIFEST = os.path.join(common.output_dir, "analysis", "reproducibility_manifest.json")
RAW_NAME = "asmr_results.json"
SEEN_NAME = "seen_video_ids.txt"

# (column, type, unit or values, description)
DICTIONARY = [
    (
        'video_id',
        'string',
        'YouTube video identifier',
        'Unique key of the table. The video is at https://www.youtube.com/watch?v=<video_id>.',
    ),
    ('title', 'string', 'text', 'Video title as returned by YouTube at collection.'),
    ('description', 'string', 'text', 'Video description as returned by YouTube at collection.'),
    (
        'language',
        'string',
        'language name or raw code',
        ('Language category from platform metadata or automatic detection (langdetect) on title and '
         'description. Aliases are normalised; unrecognised codes are kept as separate categories.'),
    ),
    ('views', 'number', 'count', 'Cumulative views at collection.'),
    (
        'likes',
        'number',
        'count',
        'Cumulative likes at collection. Missing when the channel hides likes or retrieval failed.',
    ),
    (
        'duration_seconds',
        'number',
        'seconds',
        'Video duration. At least 59 seconds where known; 559 records have 0 (not retrieved) and one is missing.',
    ),
    ('channel_id', 'string', 'YouTube channel identifier', 'Channel that uploaded the video.'),
    ('author', 'string', 'text', 'Channel display name.'),
    ('upload_datetime', 'string', 'ISO 8601, UTC', 'Upload timestamp.'),
    ('days_since_upload', 'number', 'days', 'Days between upload and the reference date 2026-09-01.'),
    (
        'channel_average_views',
        'number',
        'views per video',
        'Average views per video of the channel at collection, when available.',
    ),
    ('duration_minutes', 'number', 'minutes', 'duration_seconds divided by 60.'),
    (
        'duration_bucket',
        'category',
        'under_10min, 10_to_30min, 30_to_60min, 60_to_180min, over_180min, unknown',
        'Duration group. unknown marks records without a valid duration or upload date.',
    ),
    (
        'engagement_rate',
        'number',
        'likes per view',
        'likes divided by views. Missing when likes are missing or views are zero.',
    ),
    (
        'views_per_day',
        'number',
        'views per day',
        'views divided by days_since_upload (lifetime average; depends on video age).',
    ),
    ('likes_per_day', 'number', 'likes per day', 'likes divided by days_since_upload.'),
    ('rel_views_vs_channel_avg', 'number', 'ratio', 'views divided by channel_average_views.'),
    ('upload_year', 'number', 'year', 'Year of upload (UTC).'),
    ('upload_month', 'number', '1-12', 'Month of upload.'),
    ('upload_day', 'number', '1-31', 'Day of month of upload.'),
    ('upload_date', 'string', 'YYYY-MM-DD', 'Date of upload.'),
    (
        'upload_season',
        'category',
        'Winter, Spring, Summer, Autumn, unknown',
        'Meteorological season of the upload month in the northern hemisphere.',
    ),
    ('title_word_count', 'number', 'words', 'Number of words in the title.'),
    ('title_char_count', 'number', 'characters', 'Number of characters in the title.'),
    ('title_has_brackets', 'boolean', 'True/False', 'Title contains brackets.'),
    ('title_has_all_caps_word', 'boolean', 'True/False', 'Title contains an all-caps word of at least three letters.'),
    ('title_has_exclamation', 'boolean', 'True/False', 'Title contains an exclamation mark.'),
    ('title_has_question', 'boolean', 'True/False', 'Title contains a question mark.'),
    ('title_has_hashtag', 'boolean', 'True/False', 'Title contains a hashtag.'),
    ('title_has_no_talking_tag', 'boolean', 'True/False', 'Title carries an explicit no-talking tag.'),
    ('has_whisper', 'boolean', 'True/False', 'Whisper rule matches the title or description (see theme_rules.csv).'),
    ('has_no_talking', 'boolean', 'True/False', 'No-talking rule matches.'),
    ('has_sleep', 'boolean', 'True/False', 'Sleep rule matches.'),
    ('has_binaural', 'boolean', 'True/False', 'Binaural or 3D audio rule matches.'),
    ('has_roleplay', 'boolean', 'True/False', 'Role-play rule matches.'),
    ('has_ear_cleaning', 'boolean', 'True/False', 'Ear cleaning or ear focus rule matches.'),
    ('has_mukbang', 'boolean', 'True/False', 'Mukbang or eating rule matches.'),
    ('has_keyboard', 'boolean', 'True/False', 'Keyboard or typing rule matches.'),
    ('has_visual', 'boolean', 'True/False', 'Visual trigger rule matches.'),
    ('has_drive', 'boolean', 'True/False', 'Driving rule matches.'),
    (
        'growth_category',
        'category',
        'slow (<100 views/day), medium (100-5,000), fast (>=5,000), unknown',
        'Fixed thresholds on views_per_day.',
    ),
    (
        'language_code_raw',
        'string',
        'language code',
        'Language code stored in the source snapshot before normalisation (for example en, ko, es).',
    ),
    (
        'likes_raw_status',
        'category',
        'value, empty, not_retrieved',
        'How likes appear in the source snapshot: value = a number; empty = the source returned an empty value '
        '(typically likes hidden by the channel); not_retrieved = the value is null in the source snapshot.',
    ),
    (
        'recovery_status',
        'category',
        'recovered_from_api, recovered_from_pytubefix, created_before_api_merge, created_before_pytube_merge, empty',
        'Provenance of the record: recovered from a stored list of seen identifiers through the API or pytubefix, or '
        'created before one of the two merge steps. Empty for records from the main collection run.',
    ),
]

RAW_SCHEMA = [
    ("title", "string", "Video title."),
    ("duration", "integer or null", "Duration in seconds; null if not retrieved."),
    ("channelId", "string", "YouTube channel identifier."),
    ("author", "string", "Channel display name."),
    ("views", "integer or null", "Cumulative views at collection; null if not retrieved."),
    ("likes", "integer, empty string or null",
     "Cumulative likes; empty string when the source returned an empty value (typically hidden likes); "
     "null if not retrieved."),
    ("description", "string or null", "Video description."),
    ("uploadDate", "string", "Upload timestamp, ISO 8601, UTC."),
    ("language", "string", "Language code from platform metadata or automatic detection, before normalisation."),
    ("channel_average_views", "number or null", "Average views per video of the channel at collection."),
    ("recovery_status", "string, optional",
     "Provenance of records created or recovered after the main collection run; absent for the other records."),
]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gzip_copy(source: str, target: str) -> None:
    """Gzip a file deterministically (fixed modification time) without loading it all into memory."""
    with open(source, "rb") as src, gzip.GzipFile(target, "wb", mtime=0) as dst:
        shutil.copyfileobj(src, dst, 1 << 20)


def _add_raw_fields(df: pd.DataFrame, raw: dict) -> pd.DataFrame:
    """Add the source-snapshot fields that the enriched table does not carry."""
    df = df.copy()
    df["language_code_raw"] = df["video_id"].map(lambda v: raw.get(v, {}).get("language"))

    def likes_status(video_id: str) -> str:
        value = raw.get(video_id, {}).get("likes")
        if value is None:
            return "not_retrieved"
        return "empty" if value == "" else "value"

    df["likes_raw_status"] = df["video_id"].map(likes_status)
    df["recovery_status"] = df["video_id"].map(lambda v: raw.get(v, {}).get("recovery_status"))
    return df


def build_release() -> None:
    data_dir = common.get_configs("data")
    raw_path = os.path.join(data_dir, RAW_NAME)
    seen_path = os.path.join(data_dir, SEEN_NAME)
    with open(MANIFEST, encoding="utf-8") as handle:
        manifest = json.load(handle)
    raw_digest = _sha256(raw_path)
    if raw_digest != manifest.get("dataset_sha256"):
        raise SystemExit("The snapshot in the data folder differs from the one the analysis used "
                         f"({raw_digest} against {manifest.get('dataset_sha256')}). Rerun the analysis first.")
    with open(raw_path, encoding="utf-8") as handle:
        raw = json.load(handle)

    df = _add_raw_fields(pd.read_csv(ENRICHED, low_memory=False), raw)
    columns = [c for c, *_ in DICTIONARY]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns missing from the enriched dataset: {missing}")
    out = df[columns].sort_values("video_id")
    os.makedirs(RELEASE_DIR, exist_ok=True)

    table = os.path.join(RELEASE_DIR, "asmr_videos.csv.gz")
    with gzip.GzipFile(table, "wb", mtime=0) as handle:
        handle.write(out.to_csv(index=False).encode("utf-8"))
    _gzip_copy(raw_path, os.path.join(RELEASE_DIR, RAW_NAME + ".gz"))
    shutil.copyfile(seen_path, os.path.join(RELEASE_DIR, SEEN_NAME))

    dictionary = pd.DataFrame(DICTIONARY, columns=["variable", "type", "unit_or_values", "description"])
    dictionary["missing_count"] = [int(out[c].isna().sum()) for c in dictionary["variable"]]
    dictionary.to_csv(os.path.join(RELEASE_DIR, "data_dictionary.csv"), index=False)
    pd.DataFrame(RAW_SCHEMA, columns=["field", "type", "description"]).to_csv(
        os.path.join(RELEASE_DIR, "raw_snapshot_schema.csv"), index=False)

    rules = pd.DataFrame({"column": list(THEME_REGEX_PATTERNS),
                          "regular_expression": list(THEME_REGEX_PATTERNS.values())})
    rules["rule_version"] = "1.0.0"
    rules["flags"] = "case-insensitive, applied to the lower-cased title and description"
    rules.to_csv(os.path.join(RELEASE_DIR, "theme_rules.csv"), index=False)

    shutil.copyfile(MANIFEST, os.path.join(RELEASE_DIR, "reproducibility_manifest.json"))
    dates = out["upload_date"].dropna().astype(str)
    with open(os.path.join(RELEASE_DIR, "release_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"videos": len(out), "channels": int(out["channel_id"].nunique()),
                   "language_categories": int(out["language"].nunique()),
                   "upload_range": [dates.min(), dates.max()],
                   "table_bytes": os.path.getsize(table),
                   "raw_snapshot_sha256_uncompressed": raw_digest,
                   "raw_snapshot_bytes_uncompressed": os.path.getsize(raw_path),
                   "raw_snapshot_bytes_gzip": os.path.getsize(os.path.join(RELEASE_DIR, RAW_NAME + ".gz")),
                   "seen_video_ids": sum(1 for _ in open(seen_path, encoding="utf-8"))}, handle, indent=2)
    sums = {name: _sha256(os.path.join(RELEASE_DIR, name)) for name in sorted(os.listdir(RELEASE_DIR))
            if name != "SHA256SUMS.txt" and os.path.isfile(os.path.join(RELEASE_DIR, name))}
    with open(os.path.join(RELEASE_DIR, "SHA256SUMS.txt"), "w", encoding="utf-8") as handle:
        handle.write("".join(f"{digest}  {name}\n" for name, digest in sums.items()))
    logger.info("Release written to {} ({} videos)", RELEASE_DIR, len(out))


SUPPLEMENT_NAME = "Supplementary material"
CODE_FILES = [
    ".python-version", "LICENSE", "SECURITY.md", "analysis.py", "common.py", "custom_logger.py", "default.config",
    "default.secret", "logmod.py", "main.py", "pyproject.toml", "uv.lock",
]
CODE_DIRS = ["asmr", "tests"]
CITATION = (
    "Alam, M. S., & Bazilinskyy, P. (2026). A multilingual metadata dataset of 90,189 explicitly labelled ASMR videos "
    "uploaded to YouTube between 2008 and 2026. Manuscript submitted to Data in Brief."
)


def _readme_text(config_keys: dict) -> str:
    descriptions = {
        "data": "directory with the dataset; in this package \"../data\", relative to the folder \"code\".",
        "query": "search term used for collection; the lower-cased term must also appear in the video title.",
        "analysis_text_source": "text used for text-based analyses: \"title\", \"description\" or \"both\".",
        "date_before": "optional upper bound (YYYY-MM-DD) for the upload dates collected; null means up to today.",
        "date_after": "optional lower bound (YYYY-MM-DD) for the upload dates collected; null means no lower bound.",
        "date": "older single-date setting, used as the upper bound when \"date_before\" is null.",
        "date_window_months": "size in months of the upload-date windows that the YouTube API search is split into.",
        "analysis_reference_date": "fixed date used as the denominator of daily rates (views per day, likes per day).",
        "theme_detection_mode": "method for the theme indicators; \"rule_based\" uses the regular expressions in "
                                "data/theme_rules.csv.",
        "theme_rule_version": "version of the theme rules, recorded in the outputs.",
        "force_recompute": "rebuild outputs instead of reusing cached files.",
        "refresh_existing_statistics": "whether the enrichment step may overwrite existing view and like counts; "
                                       "keep false.",
        "random_seed": "seed for clustering, UMAP and sampling.",
        "clustering_n_clusters": "number of k-means clusters in the exploratory clustering.",
        "auto_open_plots": "if true, opens every saved HTML figure in the browser; keep false for batch runs.",
        "font_family": "font family used in the generated figures.",
        "font_size": "font size used in the generated figures.",
        "plotly_template": "template used to make the figures.",
        "logger_level": "logging level, for example \"info\".",
    }
    config_lines = [f'* "{k}": {descriptions.get(k, "see code/default.config.")}' for k in config_keys]
    template_path = os.path.join(os.path.dirname(__file__), "release_readme.txt")
    with open(template_path, encoding="utf-8") as handle:
        template = handle.read()
    return template.replace("{citation}", CITATION).replace("{config_lines}", "\n".join(config_lines))


def build_supplementary_zip(target: str) -> None:
    """Write ``target``: a zip with readme.txt, code/ and data/, laid out like a journal supplementary package."""
    build_release()
    root = common.root_dir
    with open(os.path.join(root, "default.config"), encoding="utf-8") as handle:
        config = json.load(handle)
    packaged_config = {**config, "data": "../data"}
    base = SUPPLEMENT_NAME + "/"
    skip = {"__pycache__", ".DS_Store", ".ruff_cache", ".pytest_cache"}
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(base + "readme.txt", _readme_text(config))
        for name in CODE_FILES:
            if name == "default.config":
                archive.writestr(base + "code/default.config", json.dumps(packaged_config, indent=4) + "\n")
            elif name == "pyproject.toml":
                with open(os.path.join(root, name), encoding="utf-8") as handle:
                    text = "".join(line for line in handle if not line.startswith("readme ="))
                archive.writestr(base + "code/pyproject.toml", text)
            else:
                archive.write(os.path.join(root, name), base + "code/" + name)
        for folder in CODE_DIRS:
            for dirpath, dirnames, filenames in os.walk(os.path.join(root, folder)):
                dirnames[:] = sorted(d for d in dirnames if d not in skip)
                for filename in sorted(filenames):
                    if filename in skip or filename.endswith(".pyc"):
                        continue
                    path = os.path.join(dirpath, filename)
                    archive.write(path, base + "code/" + os.path.relpath(path, root))
        for name in sorted(os.listdir(RELEASE_DIR)):
            archive.write(os.path.join(RELEASE_DIR, name), base + "data/" + name)
    logger.info("Supplementary package written to {} ({:.1f} MB)", target, os.path.getsize(target) / 1e6)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--zip":
        build_supplementary_zip(sys.argv[2])
    else:
        build_release()
