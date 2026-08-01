from custom_logger import CustomLogger
from logmod import logs
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px

import common

from utils.tool import Tools
from utils.viz_core import Plots
from utils.viz_summaries import Viz_summaries
from utils.preprocessing import Preprocessing
from utils.clustering_utils import Clustering_utils
from utils.summaries import Summaries
from utils.keyword_analysis import Keyword_analysis


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

# Choose which text source to use for text-based analyses:
#   "title"       -> titles only
#   "description" -> descriptions only
#   "both"        -> title + description
TEXT_SOURCE = common.get_configs("analysis_text_source")
REFERENCE_DATE = common.get_configs("analysis_reference_date")
FORCE_RECOMPUTE = bool(common.get_configs("force_recompute"))
RANDOM_SEED = int(common.get_configs("random_seed"))
N_CLUSTERS = int(common.get_configs("clustering_n_clusters"))
THEME_DETECTION_MODE = str(common.get_configs("theme_detection_mode"))
THEME_RULE_VERSION = str(common.get_configs("theme_rule_version"))

# Default scaling factor for saved PNG images.
SCALE = 3

font_family = common.get_configs("font_family")
font_size = common.get_configs("font_size")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"plotly\.io\._kaleido")

tool_class = Tools()
plot_class = Plots()
pre_process_class = Preprocessing()
clustering_class = Clustering_utils()
summary_class = Summaries()
viz_summary = Viz_summaries()
key_class = Keyword_analysis()


THEME_COLUMNS = [
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


def _sha256_file(path: str) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_matches_publication_settings(df: pd.DataFrame) -> bool:
    """Check whether an enriched cache was built with the active settings."""
    return (
        str(df.attrs.get("analysis_reference_date")) == str(REFERENCE_DATE)
        and str(df.attrs.get("theme_detection_mode")) == THEME_DETECTION_MODE
        and str(df.attrs.get("theme_rule_version")) == THEME_RULE_VERSION
    )


def _stamp_publication_settings(df: pd.DataFrame) -> pd.DataFrame:
    """Attach analysis settings that pandas preserves in pickle caches."""
    df.attrs["analysis_reference_date"] = str(REFERENCE_DATE)
    df.attrs["theme_detection_mode"] = THEME_DETECTION_MODE
    df.attrs["theme_rule_version"] = THEME_RULE_VERSION
    return df


def write_reproducibility_manifest(json_path: str, data: Dict[str, Any], text_source: str) -> None:
    """Save the exact data digest, analysis settings, and package versions."""
    analysis_dir = os.path.join(common.output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    package_names = [
        "pandas",
        "numpy",
        "scikit-learn",
        "spacy",
        "umap-learn",
        "plotly",
        "matplotlib",
        "langdetect",
        "pytubefix",
    ]
    package_versions = {}
    for package_name in package_names:
        try:
            package_versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            package_versions[package_name] = None

    metadata_timestamps = sorted(
        str(record.get("metadataCollectedAt"))
        for record in data.values()
        if isinstance(record, dict) and record.get("metadataCollectedAt")
    )
    language_source_counts: Dict[str, int] = {}
    for record in data.values():
        if not isinstance(record, dict):
            continue
        source = str(record.get("languageSource") or "unknown")
        language_source_counts[source] = language_source_counts.get(source, 0) + 1

    record_count = len(data)
    metadata_timestamp_count = len(metadata_timestamps)
    language_source_unknown_count = language_source_counts.get("unknown", 0)
    language_source_known_count = record_count - language_source_unknown_count

    manifest = {
        "dataset_file": os.path.basename(json_path),
        "dataset_sha256": _sha256_file(json_path),
        "video_count": record_count,
        "analysis_reference_date": str(REFERENCE_DATE),
        "analysis_text_source": text_source,
        "theme_detection_mode": THEME_DETECTION_MODE,
        "theme_rule_version": THEME_RULE_VERSION,
        "force_recompute": FORCE_RECOMPUTE,
        "random_seed": RANDOM_SEED,
        "clustering_n_clusters": N_CLUSTERS,
        "metadata_timestamp_count": metadata_timestamp_count,
        "metadata_timestamp_coverage_percent": round(
            100.0 * metadata_timestamp_count / record_count, 2
        ) if record_count else 0.0,
        "metadata_timestamp_min": metadata_timestamps[0] if metadata_timestamps else None,
        "metadata_timestamp_max": metadata_timestamps[-1] if metadata_timestamps else None,
        "language_source_counts": language_source_counts,
        "language_source_known_count": language_source_known_count,
        "language_source_unknown_count": language_source_unknown_count,
        "language_source_coverage_percent": round(
            100.0 * language_source_known_count / record_count, 2
        ) if record_count else 0.0,
        "python_version": platform.python_version(),
        "package_versions": package_versions,
    }
    manifest_path = os.path.join(analysis_dir, "reproducibility_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    logger.info(f"Reproducibility manifest saved to {manifest_path}")


def _coerce_bool_theme_series(series: pd.Series) -> pd.Series:
    """Return a real boolean Series from bool, numeric, or string theme values."""
    if series.dtype == bool:
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(float) > 0

    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _coerce_theme_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all known theme columns exist and are stored as booleans."""
    for col in THEME_COLUMNS:
        if col not in df.columns:
            df[col] = False
        df[col] = _coerce_bool_theme_series(df[col])
    return df


def _theme_flag_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact count/share table for all theme flags."""
    rows = []
    total = len(df)
    for col in THEME_COLUMNS:
        if col not in df.columns:
            count = 0
        else:
            count = int(_coerce_bool_theme_series(df[col]).sum())
        rows.append(
            {
                "theme": col,
                "true_count": count,
                "total_videos": total,
                "share": count / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _ensure_theme_flags_are_valid(df: pd.DataFrame, text_source: str, enriched_pickle: str) -> pd.DataFrame:
    """Refresh cached theme flags when they are missing or suspiciously all zero."""
    missing_before = [col for col in THEME_COLUMNS if col not in df.columns]
    df = _coerce_theme_columns(df)
    counts = _theme_flag_counts(df)
    logger.info("Theme flag counts before validation:\n" + counts.to_string(index=False))

    expected_method = (
        "english_lexical_rules"
        if THEME_DETECTION_MODE == "rule_based"
        else "spacy:en_core_web_sm"
    )
    method_mismatch = (
        "theme_detection_method" not in df.columns
        or "theme_rule_version" not in df.columns
        or not df["theme_detection_method"].astype(str).eq(expected_method).all()
        or not df["theme_rule_version"].astype(str).eq(THEME_RULE_VERSION).all()
    )
    should_refresh = (
        FORCE_RECOMPUTE
        or bool(missing_before)
        or method_mismatch
        or (int(counts["true_count"].sum()) == 0 and len(df) > 0)
    )
    if should_refresh:
        if FORCE_RECOMPUTE:
            logger.info("Recomputing theme labels because force_recompute is enabled.")
        elif missing_before:
            logger.warning(
                "The cached enriched dataset is missing theme columns: "
                f"{missing_before}. Refreshing theme detection from the text columns."
            )
        else:
            logger.warning(
                "All theme flags are zero. Refreshing theme detection from the text columns; "
                "this usually means the cached enriched pickle was created before theme detection "
                "worked, or spaCy was unavailable."
            )

        df = pre_process_class.add_theme_flags(
            df,
            text_source=text_source,
            detection_mode=THEME_DETECTION_MODE,
        )
        df = _coerce_theme_columns(df)
        df = _stamp_publication_settings(df)
        counts = _theme_flag_counts(df)
        logger.info("Theme flag counts after refresh:\n" + counts.to_string(index=False))
        df.to_pickle(enriched_pickle)

        if int(counts["true_count"].sum()) == 0:
            logger.warning(
                "Theme flags are still all zero after refresh. Check that title/description "
                "text is present and that the theme rules match your corpus."
            )

    return df

def run_wordcloud_pipeline(data: Dict[str, Any], text_source: str = "both") -> None:
    """Generate and save a word-cloud visualization for ASMR video text.

    This function orchestrates the complete word-cloud pipeline, including:
      - extracting raw text from titles, descriptions, or both;
      - applying text cleaning routines (e.g., lowercasing, stopword removal,
        punctuation/markup normalization);
      - rendering a word cloud using the project's custom visualization stack;
      - exporting interactive HTML, PNG, and EPS variants to the `figures/`
        and `output/` directories.

    Args:
        data: Dictionary of raw ASMR video metadata loaded from the JSON
            dataset. Each entry should include at least `title` and/or
            `description` fields.
        text_source: Determines which textual field(s) to use. Valid options:
            `"title"`, `"description"`, or `"both"`. Defaults to `"both"`.

    Notes:
        The function relies on utilities from:
          - `Preprocessing` for corpus construction;
          - `Tools` for text cleaning and custom stopword lists;
          - `Plots` for image generation and multi-format figure export.

        Figures are saved using the naming pattern:
            wordcloud_<text_source>.(html|png|eps)

    Returns:
        None. Outputs are written to disk and logged through the project's
        custom logger.
    """
    # Load domain-specific stopwords (e.g., ASMR jargon, channel branding terms)
    stopwords = tool_class.get_custom_stopwords()

    logger.info(f"Generating word cloud for text_source='{text_source}'")

    # Build a corpus from the desired text fields (title/description/both)
    raw_text = pre_process_class.build_corpus(data, source=text_source)

    # Apply configurable cleaning: lowercasing, stripping, punctuation filtering
    cleaned_text = tool_class.clean_text(raw_text)

    # Generate the word-cloud raster image (Pillow-based internally)
    img = plot_class.generate_wordcloud_image(cleaned_text, stopwords)

    # Wrap the raster image into an interactive Plotly container for export
    fig = plot_class.create_plotly_figure(img, title="")

    filename = f"wordcloud_{text_source}"

    # Save HTML, PNG, and EPS; open HTML automatically if config allows
    plot_class.save_plotly_figure(
        fig=fig,
        filename=filename,
        width=1600,
        height=900,
        scale=SCALE,
        save_final=True,
        save_png=True,
        save_eps=True,
    )

    logger.info(f"Wordcloud pipeline completed for text_source='{text_source}'")


def plot_spacy_keyword_bar(keyword_df: pd.DataFrame, filename: str = "spacy_keyword_bar") -> None:
    """Plot and save a bar chart of spaCy-derived keyword lemma frequencies.

    This function visualizes how often lemmatised content words (computed
    upstream via spaCy) appear across the ASMR video corpus. Each bar
    corresponds to a single lemma and its height reflects the number of
    distinct videos in which that lemma occurs at least once.

    The resulting figure is exported via the project's Plotly wrapper
    (`plot_class.save_plotly_figure`) as:
        - HTML (interactive),
        - PNG,
        - EPS,
    using the configured `SCALE`, font settings, and Plotly template.

    Args:
        keyword_df: DataFrame with at least two columns:
            - `lemma`: string lemma label (e.g., "whisper", "sleep").
            - `count`: integer count of videos containing that lemma.
        filename: Base filename (without extension) to use when saving
            the figure. Defaults to `"spacy_keyword_bar"`.

    Returns:
        None. The function writes figure files to disk and logs a summary
        message indicating how many lemmas were plotted.

    Notes:
        If `keyword_df` is empty, the function logs a warning and exits
        early to avoid generating an empty or misleading plot.
    """
    # Guard against missing or degenerate input to avoid empty plots.
    if keyword_df.empty:
        logger.warning("Keyword DataFrame is empty; no spaCy bar plot created.")
        return

    # Construct a simple bar chart: lemmas on the x-axis, counts on the y-axis.
    fig = px.bar(
        keyword_df,
        x="lemma",
        y="count",
        labels={"lemma": "Lemma", "count": "Number of videos containing lemma"},
        text="count"
    )

    # Place numeric counts above each bar for direct readability in exports.
    fig.update_traces(
        texttemplate="%{text}",
        textposition="outside",
    )

    # Rotate x-axis labels for better readability when there are many lemmas.
    fig.update_xaxes(tickangle=45)

    # Ensure the y-axis starts at zero so bar heights are interpretable.
    fig.update_yaxes(rangemode="tozero")

    # Delegate all saving (HTML/PNG/EPS) to the shared Plotly helper.
    plot_class.save_plotly_figure(fig, filename=filename, width=1600, height=900, scale=SCALE)

    # Log how many lemmas were included and under which base filename.
    logger.info(f"Saved spaCy keyword bar plot with {len(keyword_df)} lemmas as {filename}")


def summarize_by_theme_category(df: pd.DataFrame, theme_col: str) -> pd.DataFrame:
    """Compute descriptive statistics for videos split by a boolean theme flag.

    This function summarises multiple popularity and engagement indicators
    for two groups of videos: those where the thematic flag (e.g.,
    `"has_whisper"`, `"has_sleep"`, `"has_binaural"`) is `True` and those
    where it is `False`. For each group, it reports the mean and standard
    deviation of raw metrics and their log10-transformed counterparts:

        - views and views/day
        - likes and likes/day
        - engagement_rate
        - log10-transformed versions of all above metrics

    These summaries provide a structured way to quantify whether videos
    containing a given theme differ systematically from videos without it.

    Args:
        df: A DataFrame containing ASMR video metadata and derived metrics,
            including columns such as `views`, `likes`, `views_per_day`,
            `likes_per_day`, `engagement_rate`, and their log10 transforms.
        theme_col: The name of the boolean column marking presence (`True`)
            or absence (`False`) of a theme. Must already exist in `df`.

    Returns:
        A DataFrame with one row per group (`False`, `True`) and aggregated
        statistics for all listed metrics (mean and standard deviation).

    Raises:
        ValueError: If `theme_col` is not found in the provided DataFrame.

    Notes:
        The function logs a formatted table of results for traceability.
    """
    # Validate that the requested thematic column exists in the dataset.
    if theme_col not in df.columns:
        raise ValueError(f"Unknown theme column: {theme_col}")

    # Aggregate mean and SD across raw and log-transformed metrics,
    # grouped by whether the theme flag is present.
    agg = (
        df.groupby(theme_col)
        .agg(
            video_count=("video_id", "count"),

            mean_views=("views", "mean"),
            sd_views=("views", "std"),

            mean_views_per_day=("views_per_day", "mean"),
            sd_views_per_day=("views_per_day", "std"),

            mean_likes=("likes", "mean"),
            sd_likes=("likes", "std"),

            mean_likes_per_day=("likes_per_day", "mean"),
            sd_likes_per_day=("likes_per_day", "std"),

            mean_engagement_rate=("engagement_rate", "mean"),
            sd_engagement_rate=("engagement_rate", "std"),

            mean_log10_views=("log10_views", "mean"),
            sd_log10_views=("log10_views", "std"),

            mean_log10_views_per_day=("log10_views_per_day", "mean"),
            sd_log10_views_per_day=("log10_views_per_day", "std"),

            mean_log10_likes=("log10_likes", "mean"),
            sd_log10_likes=("log10_likes", "std"),

            mean_log10_likes_per_day=("log10_likes_per_day", "mean"),
            sd_log10_likes_per_day=("log10_likes_per_day", "std"),

            mean_log10_engagement_rate=("log10_engagement_rate", "mean"),
            sd_log10_engagement_rate=("log10_engagement_rate", "std"),
        )
        .reset_index()
    )

    # Log a readable textual summary for auditing and interpretation.
    logger.info(
        f"Thematic summary (raw + log10) for {theme_col} (False/True):\n"
        f"{agg.to_string(index=False)}"
    )

    return agg


def summarize_theme_by_duration_bucket(df: pd.DataFrame, theme_col: str) -> pd.DataFrame:
    """Summarise popularity and engagement metrics by theme flag and duration bucket.

    This function computes descriptive statistics for ASMR videos jointly
    stratified by a boolean theme indicator (e.g., ``"has_whisper"``,
    ``"has_sleep"``, ``"has_binaural"``) and by predefined duration
    buckets (e.g., under 10 minutes, 10–30 minutes, etc.).

    For each (theme, duration_bucket) combination, it reports:

    * ``video_count`` (number of videos in the cell)
    * Mean and standard deviation of raw metrics:
        * ``views``
        * ``views_per_day``
        * ``likes``
        * ``likes_per_day``
        * ``engagement_rate``
    * Mean and standard deviation of log10-transformed metrics:
        * ``log10_views``
        * ``log10_views_per_day``
        * ``log10_likes``
        * ``log10_likes_per_day``
        * ``log10_engagement_rate``

    These summaries are useful to assess how the interaction between
    thematic category and video duration relates to popularity and
    engagement patterns.

    Args:
        df: A pandas DataFrame containing one row per video and at least
            the following columns:

            * ``video_id``
            * ``views``, ``views_per_day``
            * ``likes``, ``likes_per_day``
            * ``engagement_rate``
            * ``log10_views``, ``log10_views_per_day``
            * ``log10_likes``, ``log10_likes_per_day``
            * ``log10_engagement_rate``
            * ``duration_bucket``: categorical duration bin for each video.
            * ``theme_col``: a boolean theme flag column (see below).

        theme_col: Name of the boolean column in ``df`` indicating whether
            a given theme is present (``True``) or absent (``False``),
            for example ``"has_whisper"`` or ``"has_sleep"``. This column
            must already exist in ``df``.

    Returns:
        A pandas DataFrame with one row per (theme_col, duration_bucket)
        combination. Each row contains the number of videos and the
        mean/standard deviation for all raw and log10-transformed metrics.

    Raises:
        ValueError: If ``theme_col`` is not present in ``df`` or if the
            ``duration_bucket`` column is missing.

    Notes:
        * The duration buckets are ordered explicitly as:

          ``["under_10min", "10_to_30min", "30_to_60min",
          "60_to_180min", "over_180min", "unknown"]``

        * A formatted summary table is logged via the project logger
          for reproducibility and quick inspection.
    """
    # Validate that the requested theme indicator exists in the DataFrame.
    if theme_col not in df.columns:
        raise ValueError(f"Unknown theme column: {theme_col}")

    # Ensure that duration_bucket is available for the joint stratification.
    if "duration_bucket" not in df.columns:
        raise ValueError("Column 'duration_bucket' is missing from DataFrame.")

    # Work on a copy to avoid mutating the original dataset.
    df_copy = df.copy()

    # Define a consistent, ordered set of duration buckets so output
    # tables and plots follow a predictable ordering.
    bucket_order = [
        "under_10min",
        "10_to_30min",
        "30_to_60min",
        "60_to_180min",
        "over_180min",
        "unknown",
    ]

    # Cast duration_bucket to a Categorical with the specified order.
    df_copy["duration_bucket"] = pd.Categorical(
        df_copy["duration_bucket"],
        categories=bucket_order,
        ordered=True,
    )

    # Group by (theme_col, duration_bucket) and aggregate counts plus
    # mean/SD for both raw and log10-transformed metrics.
    agg = (
        df_copy.groupby([theme_col, "duration_bucket"])
        .agg(
            video_count=("video_id", "count"),

            mean_views=("views", "mean"),
            sd_views=("views", "std"),

            mean_views_per_day=("views_per_day", "mean"),
            sd_views_per_day=("views_per_day", "std"),

            mean_likes=("likes", "mean"),
            sd_likes=("likes", "std"),

            mean_likes_per_day=("likes_per_day", "mean"),
            sd_likes_per_day=("likes_per_day", "std"),

            mean_engagement_rate=("engagement_rate", "mean"),
            sd_engagement_rate=("engagement_rate", "std"),

            mean_log10_views=("log10_views", "mean"),
            sd_log10_views=("log10_views", "std"),

            mean_log10_views_per_day=("log10_views_per_day", "mean"),
            sd_log10_views_per_day=("log10_views_per_day", "std"),

            mean_log10_likes=("log10_likes", "mean"),
            sd_log10_likes=("log10_likes", "std"),

            mean_log10_likes_per_day=("log10_likes_per_day", "mean"),
            sd_log10_likes_per_day=("log10_likes_per_day", "std"),

            mean_log10_engagement_rate=("log10_engagement_rate", "mean"),
            sd_log10_engagement_rate=("log10_engagement_rate", "std"),
        )
        .reset_index()
    )

    # Log a neatly formatted multi-line summary to aid interpretation
    # and to make the analysis traceable from the logs alone.
    logger.info(
        f"{theme_col} × duration_bucket summary (raw + log10):\n"
        f"{agg.to_string(index=False)}"
    )

    return agg


def run_elbow_analysis(text_source: str = "both") -> None:
    """Run K-means elbow analysis to select a reasonable number of clusters.

    This function prepares an enriched ASMR video dataset and computes
    a K-means elbow curve for a range of cluster counts. It first attempts
    to load a cached, pre-processed dataset from a pickle file; if that
    does not exist, it rebuilds the dataset from the raw JSON file using
    the project's preprocessing utilities.

    The analysis proceeds as follows:

    1. Load (or construct) an enriched DataFrame of ASMR videos containing
       at least the following fields:

       * Basic identifiers and metadata (e.g., ``video_id``, title, etc.).
       * Duration and engagement-related variables (e.g., ``duration_minutes``,
         ``views``, ``likes``, ``views_per_day``, ``engagement_rate``,
         ``language``).
       * Derived temporal features such as ``days_since_upload``.

    2. Ensure that ``likes_per_day`` is available. For older cached data
       where this column might be missing, it is backfilled as:

       ``likes_per_day = likes / days_since_upload`` for valid entries.

    3. Call :meth:`Clustering_utils.compute_kmeans_elbow` to fit K-means
       models over a predefined range of ``k`` values (here ``k=4..20``),
       using the same feature setup as in the main clustering pipeline.

    4. Log a formatted summary of the resulting inertia values (one per
       tested ``k``) to facilitate manual inspection of the elbow region.

    Args:
        text_source: String specifying which text field(s) to use in the
            clustering features. Typical values are:

            * ``"title"`` – use video titles only,
            * ``"description"`` – use descriptions only,
            * ``"both"`` – use concatenated title and description.

            This parameter must be consistent with the rest of the
            preprocessing and clustering configuration.

    Returns:
        None. The function is called for its side effects: loading or
        constructing the dataset, computing the K-means inertia curve,
        and logging the results. The full elbow DataFrame is returned
        by :meth:`Clustering_utils.compute_kmeans_elbow` and captured
        in the local variable ``elbow_df``, but it is not written to
        disk directly here.

    Notes:
        * The enriched dataset is expected to be stored in (and loaded
          from) ``output/analysis/asmr_videos_enriched_{text_source}.pkl``.
        * The range of ``k`` (here 4–20) can be adjusted if finer or
          coarser exploration of cluster counts is required.
        * The function assumes that global instances of the preprocessing
          and clustering utility classes (``pre_process_class`` and
          ``clustering_class``) are available in the module scope.
    """
    # Derive the path to the analysis directory inside the project output.
    analysis_dir = os.path.join(common.output_dir, "analysis")

    # Path to the enriched dataset pickle for the selected text source.
    enriched_pickle = os.path.join(
        analysis_dir,
        f"asmr_videos_enriched_{text_source}.pkl",
    )

    # Try to load the enriched DataFrame from disk; if not present,
    # build it from the raw JSON file using the preprocessing utilities.
    if os.path.isfile(enriched_pickle) and not FORCE_RECOMPUTE:
        logger.info(f"Loading enriched dataset from pickle {enriched_pickle}")
        df = pd.read_pickle(enriched_pickle)
        if not _cache_matches_publication_settings(df):
            logger.info("Cached enriched dataset uses different analysis settings; rebuilding it.")
            data = pre_process_class.load_asmr_data(
                os.path.join(common.get_configs("data"), "asmr_results.json")
            )
            df = pre_process_class.json_to_dataframe(
                data,
                reference_date=pre_process_class.parse_reference_datetime(REFERENCE_DATE),
                text_source=text_source,
            )
            df = _stamp_publication_settings(df)
            df.to_pickle(enriched_pickle)
    else:
        logger.info("No enriched pickle found; building DataFrame from JSON...")
        json_path = os.path.join(common.get_configs("data"), "asmr_results.json")
        data = pre_process_class.load_asmr_data(json_path)
        df = pre_process_class.json_to_dataframe(
            data,
            reference_date=pre_process_class.parse_reference_datetime(REFERENCE_DATE),
            text_source=text_source,
        )
        df = _stamp_publication_settings(df)

    # Backfill likes_per_day for older pickles if needed. Some historical
    # cached datasets may be missing this derived column, so we recompute
    # it whenever days_since_upload is available.
    if "likes_per_day" not in df.columns and "days_since_upload" in df.columns:
        df["likes_per_day"] = np.where(
            df["days_since_upload"] > 0,
            df["likes"] / df["days_since_upload"],
            np.nan,
        )

    # Define the range of k values for which we want to compute
    # K-means inertia. Here, we scan from k=4 to k=20 inclusive.
    k_values = range(4, 21)  # for example, k = 4..20

    # Compute the elbow curve (k vs inertia) using the project-level
    # clustering utilities, reusing the same feature engineering as
    # in the main clustering pipeline.
    elbow_df = clustering_class.compute_kmeans_elbow(
        df,
        k_values,
        text_source=text_source,
        random_state=RANDOM_SEED,
    )

    # Log a neatly formatted summary of the elbow results. The logged
    # table shows inertia for each tested k, which helps in visually
    # identifying the elbow region (where marginal gains diminish).
    logger.info(f"Elbow results:\n{elbow_df.to_string(index=False)}")


def plot_language_stats(lang_stats: pd.DataFrame, min_videos: int = 20) -> None:
    """Plot language-level engagement and growth statistics.

    This function visualizes language-level performance metrics for ASMR
    videos based on a precomputed summary DataFrame. It produces two bar
    charts using Plotly:

    1. Mean views per day by language (for languages with sufficient data).
    2. Mean engagement rate (likes / views) by language.

    The input ``lang_stats`` DataFrame is expected to contain one row per
    language with summary statistics computed upstream, including at least:

    * ``language``: Language label (string).
    * ``video_count``: Number of videos for the language.
    * ``mean_views_per_day``: Average views per day across videos.
    * ``mean_engagement_rate``: Average engagement rate (likes / views).

    The function applies a minimum video-count threshold so that very rare
    languages (with few videos) do not dominate or clutter the plots.
    It also logs which languages have the highest mean views per day and
    highest mean engagement rate, providing a textual summary that can be
    inspected alongside the figures.

    Args:
        lang_stats: A pandas DataFrame with per-language summary statistics
            (e.g., as returned by a language-level aggregation function).
            Must include at least ``language``, ``video_count``,
            ``mean_views_per_day``, and ``mean_engagement_rate`` columns.
        min_videos: Minimum number of videos required for a language to be
            included in the plots. Languages with fewer than this number of
            videos are filtered out to avoid unstable or noisy estimates.

    Returns:
        None. The function is executed for its side effects:
        it filters and logs summary information and writes two Plotly-based
        bar charts to disk via ``plot_class.save_plotly_figure``:

        * ``language_mean_views_per_day.*``
        * ``language_mean_engagement_rate.*``

    Notes:
        * If ``lang_stats`` is empty or no languages meet the ``min_videos``
          threshold, the function logs a warning and exits without
          generating plots.
        * The global ``plot_class`` instance and ``SCALE`` constant are
          assumed to be defined at module level and encapsulate project-wide
          figure styling and export behavior.
    """
    # If there is no language-level data at all, we cannot produce any plot.
    if lang_stats.empty:
        logger.warning("Language stats DataFrame is empty; skipping plots.")
        return

    # Work on a copy to avoid modifying the original DataFrame.
    df_plot = lang_stats.copy()

    # Keep only languages with at least `min_videos` videos to ensure
    # that plotted statistics are based on reasonably sized samples.
    df_plot = df_plot[df_plot["video_count"] >= min_videos]
    if df_plot.empty:
        logger.warning(
            f"No languages with at least {min_videos} videos; skipping language plots."
        )
        return

    # Log how many languages pass the threshold and will appear in the plots.
    logger.info(
        f"Language stats include {len(df_plot)} languages with at least "
        f"{min_videos} videos each."
    )

    # ----------------------------------------------------------------------
    # Identify and log the language with highest mean views per day.
    # ----------------------------------------------------------------------
    df_views = df_plot.sort_values("mean_views_per_day", ascending=False)
    top_vpd = df_views.iloc[0]
    logger.info(
        "Language with highest mean views per day: "
        f"{top_vpd['language']} "
        f"(videos={int(top_vpd['video_count'])}, "
        f"mean_views_per_day={top_vpd['mean_views_per_day']:.2f})"
    )

    # ----------------------------------------------------------------------
    # Identify and log the language with highest mean engagement rate.
    # ----------------------------------------------------------------------
    df_eng = df_plot.sort_values("mean_engagement_rate", ascending=False)
    top_eng = df_eng.iloc[0]
    logger.info(
        "Language with highest mean engagement rate: "
        f"{top_eng['language']} "
        f"(videos={int(top_eng['video_count'])}, "
        f"mean_engagement_rate={top_eng['mean_engagement_rate']:.4f})"
    )

    # ----------------------------------------------------------------------
    # Plot: mean views per day by language.
    # ----------------------------------------------------------------------
    fig = px.bar(
        df_views,
        x="language",
        y="mean_views_per_day",
        title="",
        labels={"language": "Language", "mean_views_per_day": "Mean views per day"},
    )
    # Angle x-axis labels to improve readability when there are many languages.
    fig.update_xaxes(tickangle=45)
    plot_class.save_plotly_figure(fig, "language_mean_views_per_day", width=1600, height=900, scale=SCALE)

    # ----------------------------------------------------------------------
    # Plot: mean engagement rate by language.
    # ----------------------------------------------------------------------
    fig = px.bar(
        df_eng,
        x="language",
        y="mean_engagement_rate",
        title="",
        labels={
            "language": "Language",
            "mean_engagement_rate": "Mean engagement rate",
        },
    )
    # Again, rotate the labels to prevent overlap.
    fig.update_xaxes(tickangle=45)
    plot_class.save_plotly_figure(fig, "language_mean_engagement_rate", width=1600, height=900, scale=SCALE)


def plot_monthly_counts(monthly_counts: pd.DataFrame) -> None:
    """Line chart for number of videos per month (community growth)."""
    if monthly_counts.empty:
        logger.warning("Monthly counts DataFrame is empty; skipping plot.")
        return

    df_plot = monthly_counts.copy()
    df_plot = df_plot.sort_values("year_month")

    logger.info(
        "Monthly counts cover "
        f"{len(df_plot)} months from "
        f"{df_plot['year_month'].min().date()} "
        f"to {df_plot['year_month'].max().date()}."
    )

    fig = px.line(
        df_plot,
        x="year_month",
        y="video_count",
        title="",
        labels={"year_month": "Month", "video_count": "Number of videos"},
    )
    plot_class.save_plotly_figure(fig, "monthly_video_counts", width=1600, height=900, scale=SCALE)

# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Log basic summary: N videos, oldest/newest video, mean/SD duration, views, likes."""
    n_videos = len(df)
    logger.info("===== DATASET SUMMARY =====")
    logger.info(f"Total number of videos: {n_videos}")

    n_channels = df["channel_id"].nunique(dropna=True)
    n_authors = df["author"].nunique(dropna=True)
    n_languages = df["language"].nunique(dropna=True)
    logger.info(
        f"Number of unique channels: {n_channels}, "
        f"unique authors: {n_authors}, unique languages: {n_languages}"
    )

    if df["upload_datetime"].notna().any():
        oldest_idx = df["upload_datetime"].idxmin()
        newest_idx = df["upload_datetime"].idxmax()
        oldest = df.loc[oldest_idx]
        newest = df.loc[newest_idx]
        logger.info(
            "Oldest video: "
            f"'{oldest.get('title', '')}' "
            f"(id={oldest.get('video_id', '')}), "
            f"uploaded on {oldest.get('upload_datetime')}"
        )
        logger.info(
            "Newest video: "
            f"'{newest.get('title', '')}' "
            f"(id={newest.get('video_id', '')}), "
            f"uploaded on {newest.get('upload_datetime')}"
        )
    else:
        logger.info("Oldest/newest video: upload dates are unavailable.")

    if df["duration_seconds"].notna().any():
        mean_dur_sec = df["duration_seconds"].mean()
        std_dur_sec = df["duration_seconds"].std()
        logger.info(
            f"Duration (seconds): mean = {mean_dur_sec:.2f}, SD = {std_dur_sec:.2f}"
        )
    else:
        logger.info("Duration statistics: not available.")

    if df["views"].notna().any():
        mean_views = df["views"].mean()
        std_views = df["views"].std()
        logger.info(
            f"Views: mean = {mean_views:.2f}, SD = {std_views:.2f}"
        )
    else:
        logger.info("View statistics: not available.")

    if df["likes"].notna().any():
        mean_likes = df["likes"].mean()
        std_likes = df["likes"].std()
        logger.info(
            f"Likes: mean = {mean_likes:.2f}, SD = {std_likes:.2f}"
        )
    else:
        logger.info("Like statistics: not available.")

    if df["views_per_day"].notna().any():
        mean_vpd = df["views_per_day"].mean()
        std_vpd = df["views_per_day"].std()
        logger.info(
            f"Views per day: mean = {mean_vpd:.2f}, SD = {std_vpd:.2f}"
        )
    else:
        logger.info("Views-per-day statistics: not available.")

    if "likes_per_day" in df.columns and df["likes_per_day"].notna().any():
        mean_lpd = df["likes_per_day"].mean()
        std_lpd = df["likes_per_day"].std()
        logger.info(
            f"Likes per day: mean = {mean_lpd:.2f}, SD = {std_lpd:.2f}"
        )
    else:
        logger.info("Likes-per-day statistics: not available.")

    logger.info("===== END DATASET SUMMARY =====")


# ---------------------------------------------------------------------------
# Analytics pipeline: CSVs + figures
# ---------------------------------------------------------------------------

def run_analytics_pipeline(data: Dict[str, Any], text_source: str = "both") -> None:
    """Run all analytics, write CSVs into output/analysis, and create Plotly figures."""
    analysis_dir = os.path.join(common.output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    enriched_pickle = os.path.join(
        analysis_dir,
        f"asmr_videos_enriched_{text_source}.pkl",
    )

    if os.path.isfile(enriched_pickle) and not FORCE_RECOMPUTE:
        logger.info(f"Loading enriched dataset from pickle {enriched_pickle}")
        df = pd.read_pickle(enriched_pickle)
        if not _cache_matches_publication_settings(df):
            logger.info("Cached enriched dataset uses different analysis settings; rebuilding it.")
            df = pre_process_class.json_to_dataframe(
                data,
                reference_date=pre_process_class.parse_reference_datetime(REFERENCE_DATE),
                text_source=text_source,
            )
            df = _stamp_publication_settings(df)
            df.to_pickle(enriched_pickle)
    else:
        logger.info("No enriched pickle found; building DataFrame from JSON...")
        df = pre_process_class.json_to_dataframe(
            data,
            reference_date=pre_process_class.parse_reference_datetime(REFERENCE_DATE),
            text_source=text_source,
        )
        df = _stamp_publication_settings(df)
        logger.info(f"Saving enriched dataset to pickle {enriched_pickle}")
        df.to_pickle(enriched_pickle)

    df = _ensure_theme_flags_are_valid(df, text_source=text_source, enriched_pickle=enriched_pickle)

    logger.info(
        f"Enriched DataFrame ready with {len(df)} rows and {len(df.columns)} columns."
    )

    theme_counts_path = os.path.join(analysis_dir, "theme_flag_counts.csv")
    _theme_flag_counts(df).to_csv(theme_counts_path, index=False)
    logger.info(f"Theme flag counts saved to {theme_counts_path}")

    print_dataset_summary(df)

    enriched_csv = os.path.join(analysis_dir, "asmr_videos_enriched.csv")
    logger.info(f"Saving enriched dataset CSV to {enriched_csv}")
    df.to_csv(enriched_csv, index=False)

    duration_stats = summary_class.summarize_by_duration_bucket(df)
    duration_stats.to_csv(
        os.path.join(analysis_dir, "duration_stats.csv"),
        index=False,
    )

    plot_class.plot_duration_vs_views(df)

    plot_class.analyze_log_views_normality(df)

    lang_stats = summary_class.summarize_by_language(df)
    lang_stats.to_csv(
        os.path.join(analysis_dir, "language_stats.csv"),
        index=False,
    )
    plot_language_stats(lang_stats)

    title_stats = summary_class.summarize_title_styles(df)
    title_stats.to_csv(
        os.path.join(analysis_dir, "title_style_stats.csv"),
        index=False,
    )
    viz_summary.plot_title_style_stats(title_stats)

    for theme in ["has_whisper", "has_no_talking", "has_sleep", "has_binaural", "has_drive"]:
        if theme in df.columns:
            theme_stats = summary_class.summarize_theme_vs_growth(df, theme)
            theme_stats.to_csv(
                os.path.join(analysis_dir, f"{theme}_growth_stats.csv"),
                index=False,
            )

    # NEW: per-theme and theme × duration summaries in raw + log10 scale
    for theme in ["has_whisper", "has_no_talking", "has_sleep", "has_binaural", "has_drive"]:
        if theme in df.columns:
            theme_cat_stats = summarize_by_theme_category(df, theme)
            theme_cat_stats.to_csv(
                os.path.join(analysis_dir, f"{theme}_summary_raw_log.csv"),
                index=False,
            )

            theme_duration_stats = summarize_theme_by_duration_bucket(df, theme)
            theme_duration_stats.to_csv(
                os.path.join(analysis_dir, f"{theme}_by_duration_summary_raw_log.csv"),
                index=False,
            )

            viz_summary.plot_theme_growth_box(df, theme)

    monthly_counts = summary_class.compute_monthly_video_counts(df)
    monthly_counts.to_csv(
        os.path.join(analysis_dir, "monthly_video_counts.csv"),
        index=False,
    )
    plot_monthly_counts(monthly_counts)

    lang_growth = summary_class.compute_language_growth(df)
    lang_growth.to_csv(
        os.path.join(analysis_dir, "language_growth.csv"),
        index=False,
    )
    viz_summary.plot_language_growth(lang_growth)

    for theme in THEME_COLUMNS:
        if theme in df.columns:
            trend_all = summary_class.compute_theme_trend_over_time(
                df, theme_col=theme, by_language=False
            )
            trend_lang = summary_class.compute_theme_trend_over_time(
                df, theme_col=theme, by_language=True
            )

            trend_all.to_csv(
                os.path.join(analysis_dir, f"{theme}_trend_overall.csv"),
                index=False,
            )
            trend_lang.to_csv(
                os.path.join(analysis_dir, f"{theme}_trend_by_language.csv"),
                index=False,
            )

            viz_summary.plot_theme_trend_overall(trend_all, theme)
            viz_summary.plot_theme_trend_by_language(trend_lang, theme)

            if theme == "has_drive":
                # Keep the historical output names expected by earlier runs.
                trend_all.to_csv(
                    os.path.join(analysis_dir, "drive_trend_overall.csv"),
                    index=False,
                )
                viz_summary.plot_theme_trend_overall(trend_all, theme_col="drive")

    # ------------------------------------------------------------------
    # KMeans clustering (PCA embedding) with PKL caching
    # ------------------------------------------------------------------
    cluster_pickle = os.path.join(
        analysis_dir,
        f"asmr_videos_with_clusters_{text_source}.pkl",
    )

    if os.path.isfile(cluster_pickle) and not FORCE_RECOMPUTE:
        logger.info(f"Loading clustered dataset from pickle {cluster_pickle}")
        clustered_df = pd.read_pickle(cluster_pickle)
        pipeline = None
        pca_info = None
    else:
        logger.info("No clustered pickle found; running KMeans clustering...")
        clustered_df, pipeline, pca_info = clustering_class.cluster_videos(
            df,
            n_clusters=N_CLUSTERS,
            random_state=RANDOM_SEED,
            text_source=text_source,
        )
        logger.info(f"Saving clustered dataset to pickle {cluster_pickle}")
        clustered_df.to_pickle(cluster_pickle)

    # Backfill likes_per_day on clustered_df if needed
    if "likes_per_day" not in clustered_df.columns and "days_since_upload" in clustered_df.columns:
        clustered_df["likes_per_day"] = np.where(
            clustered_df["days_since_upload"] > 0,
            clustered_df["likes"] / clustered_df["days_since_upload"],
            np.nan,
        )

    clustered_csv = os.path.join(analysis_dir, "asmr_videos_with_clusters.csv")
    clustered_df.to_csv(
        clustered_csv,
        index=False,
    )
    logger.info(f"Clustered dataset (PCA) saved to {clustered_csv}")

    if "cluster" in clustered_df.columns:
        cluster_summary = (
            clustered_df.groupby("cluster")
            .agg(
                video_count=("video_id", "count"),
                mean_views=("views", "mean"),
                sd_views=("views", "std"),
                median_views=("views", "median"),
                mean_views_per_day=("views_per_day", "mean"),
                sd_views_per_day=("views_per_day", "std"),
                mean_likes=("likes", "mean"),
                sd_likes=("likes", "std"),
                mean_likes_per_day=("likes_per_day", "mean"),
                sd_likes_per_day=("likes_per_day", "std"),
                mean_duration_minutes=("duration_minutes", "mean"),
            )
            .reset_index()
        )

        cluster_summary_csv = os.path.join(analysis_dir, "cluster_summary.csv")
        cluster_summary.to_csv(cluster_summary_csv, index=False)
        logger.info(
            "Cluster summary table:\n"
            f"{cluster_summary.to_string(index=False)}"
        )
        logger.info(f"Cluster summary saved to {cluster_summary_csv}")

        cluster_desc = clustering_class.describe_clusters(clustered_df)
        cluster_desc_csv = os.path.join(analysis_dir, "cluster_descriptions.csv")
        cluster_desc.to_csv(cluster_desc_csv, index=False)
        logger.info(f"Cluster descriptions saved to {cluster_desc_csv}")

    if pca_info is not None and not pca_info.empty:
        pca_csv = os.path.join(analysis_dir, "cluster_pca_variance.csv")
        pca_info.to_csv(pca_csv, index=False)
        logger.info(f"PCA variance info saved to {pca_csv}")

    needed = {"video_id", "cluster", "embedding_x", "embedding_y"}
    if needed.issubset(clustered_df.columns):
        emb_cols = [
            "video_id",
            "cluster",
            "embedding_x",
            "embedding_y",
            "views",
            "views_per_day",
            "duration_minutes",
            "language",
            "title",
        ]
        emb_cols = [c for c in emb_cols if c in clustered_df.columns]
        embedding_csv = os.path.join(analysis_dir, "cluster_embedding_2d.csv")
        clustered_df[emb_cols].to_csv(embedding_csv, index=False)
        logger.info(f"2D PCA embedding for clusters saved to {embedding_csv}")

        # Stable colors; highlight_cluster can be set here if desired
        clustering_class.plot_cluster_distribution(clustered_df, name_suffix="pca")

        # ------------------------------------------------------------------
        # UMAP embedding of clusters with PKL caching
        # ------------------------------------------------------------------
        umap_svd_components = 50
        umap_n_neighbors = 30
        umap_min_dist = 0.1
        umap_metric = "cosine"
        # Full coverage is preserved in the saved UMAP coordinates and static PNG.
        # Interactive browser plots are capped so they remain openable.
        umap_interactive_plot_max_points = 12000

        umap_cluster_pickle = os.path.join(
            analysis_dir,
            (
                f"asmr_videos_with_clusters_umap_{text_source}_"
                f"svd{umap_svd_components}_nn{umap_n_neighbors}_mindist{str(umap_min_dist).replace('.', 'p')}.pkl"
            ),
        )

        recompute_umap = True
        if os.path.isfile(umap_cluster_pickle) and not FORCE_RECOMPUTE:
            logger.info(f"Loading full UMAP clustered dataset from pickle {umap_cluster_pickle}")
            clustered_umap_df = pd.read_pickle(umap_cluster_pickle)
            if {"embedding_x", "embedding_y"}.issubset(clustered_umap_df.columns):
                n_valid_umap = clustered_umap_df.dropna(subset=["embedding_x", "embedding_y"]).shape[0]
                recompute_umap = n_valid_umap == 0
                if recompute_umap:
                    logger.warning(
                        "Cached UMAP pickle has no valid 2D coordinates; recomputing UMAP."
                    )
                else:
                    logger.info(
                        f"Cached UMAP pickle contains {n_valid_umap} rows with valid 2D coordinates."
                    )
            else:
                logger.warning(
                    "Cached UMAP pickle is missing embedding_x / embedding_y; recomputing UMAP."
                )

        if recompute_umap:
            logger.info(
                "Computing UMAP embedding for every video in the dataset..."
            )
            clustered_umap_df, umap_pipeline = clustering_class.cluster_videos_umap(
                df,
                n_clusters=N_CLUSTERS,
                random_state=RANDOM_SEED,
                text_source=text_source,
                umap_n_neighbors=umap_n_neighbors,
                umap_min_dist=umap_min_dist,
                umap_metric=umap_metric,
                svd_components=umap_svd_components,
            )
            logger.info(f"Saving full UMAP clustered dataset to pickle {umap_cluster_pickle}")
            clustered_umap_df.to_pickle(umap_cluster_pickle)

        # Full-dataset static plot: includes every video with UMAP coordinates.
        clustering_class.plot_embedding_static_full(
            clustered_umap_df,
            name_suffix="umap_full",
            embedding_name="UMAP",
            filename_prefix="cluster_umap_full_static",
        )

        # Interactive research plot: sampled only for browser responsiveness.
        clustering_class.plot_umap_research(
            clustered_umap_df,
            max_points=umap_interactive_plot_max_points,
        )

        umap_embedding_csv = os.path.join(analysis_dir, "cluster_embedding_2d_umap.csv")

        if {"embedding_x", "embedding_y"}.issubset(clustered_umap_df.columns):
            umap_emb_df = clustered_umap_df.dropna(subset=["embedding_x", "embedding_y"])
            umap_emb_df[emb_cols].to_csv(umap_embedding_csv, index=False)
            logger.info(
                f"2D UMAP embedding for full clusters saved to {umap_embedding_csv} "
                f"({len(umap_emb_df)} rows with non-null embeddings)"
            )

        clustering_class.plot_cluster_distribution(
            clustered_umap_df,
            name_suffix="umap",
            max_scatter_points=umap_interactive_plot_max_points,
        )

    logger.info(
        f"Analytics pipeline complete. CSVs and figures written to {analysis_dir}"
    )


# ============================================================================
# MAIN — run wordclouds + analytics + spaCy keyword bar plot
# ============================================================================

def main() -> None:
    json_path = os.path.join(common.get_configs("data"), "asmr_results.json")
    logger.info(f"Loading ASMR data from {json_path}")
    data = pre_process_class.load_asmr_data(json_path)

    analysis_dir = os.path.join(common.output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    write_reproducibility_manifest(json_path, data, TEXT_SOURCE)

    # Always generate wordclouds for title, description, and both
    for src in ["title", "description", "both"]:
        logger.info(f"Generating wordclouds for text_source='{src}'")

        # Raw-text wordcloud
        run_wordcloud_pipeline(data, text_source=src)

        # Verb-only lemma wordcloud (cached per text_source)
        key_class.run_verb_lemma_wordcloud_pipeline(
            data,
            text_source=src,
            model_name="en_core_web_sm",
            top_k=200,
            use_cache=not FORCE_RECOMPUTE,
        )

    keyword_pickle = os.path.join(
        analysis_dir,
        f"spacy_keywords_{TEXT_SOURCE}.pkl",
    )

    if os.path.isfile(keyword_pickle) and not FORCE_RECOMPUTE:
        logger.info(f"Loading spaCy keyword counts from pickle {keyword_pickle}")
        keyword_df = pd.read_pickle(keyword_pickle)
    else:
        logger.info("No spaCy keyword pickle found; computing keyword counts...")
        keyword_df = key_class.compute_spacy_keyword_counts(
            data,
            target_lemmas=None,
            text_source=TEXT_SOURCE,
            model_name="en_core_web_sm",
            top_k=30,
            extra_stopwords=tool_class.get_custom_stopwords(),
        )
        logger.info(f"Saving spaCy keyword counts to pickle {keyword_pickle}")
        keyword_df.to_pickle(keyword_pickle)

    plot_spacy_keyword_bar(
        keyword_df,
        filename=f"spacy_keywords_{TEXT_SOURCE}",
    )

    run_analytics_pipeline(data, text_source=TEXT_SOURCE)
    run_elbow_analysis(text_source=TEXT_SOURCE)


if __name__ == "__main__":

    main()
