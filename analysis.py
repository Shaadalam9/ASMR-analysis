"""
ASMR full analysis:

- Word clouds (titles / descriptions / both)
- Metrics:
    - engagement_rate = likes / views
    - views_per_day = views / days_since_upload
    - duration buckets: short / normal / long / other
    - title style features
    - theme flags (whisper / no-talking / sleep / binaural / etc.)
    - growth_category (fast / slow / medium)
- Aggregations:
    - by duration bucket
    - by language
    - by title length
    - theme vs growth
    - monthly counts, language growth, theme trends, seasonal sleep
- Clustering using title+description text, duration, engagement, language
- Plotly figures for all of the above
"""

import json
import logging
import os
import re
import shutil
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly as py
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS

import common  # your existing project config module

# Optional: ML imports for clustering.
try:
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Default scaling factor for saved PNG images.
SCALE = 3

logger = logging.getLogger(__name__)

# ============================================================================
# Shared: load JSON
# ============================================================================


def load_asmr_data(json_path: str) -> Dict[str, Any]:
    """Load ASMR results from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ============================================================================
# PART 1 — WORDCLOUD PIPELINE (original logic)
# ============================================================================


def build_corpus(data: Dict[str, Any], source: str) -> str:
    """Build a text corpus from titles, descriptions, or both.

    Args:
        data: Dict mapping video_id -> metadata.
        source: 'title', 'description', or 'both'.

    Returns:
        Concatenated text.
    """
    texts = []

    for _, info in data.items():
        raw_title = info.get("title")
        raw_description = info.get("description")

        # Coerce to string; treat None as empty string.
        title = raw_title if isinstance(raw_title, str) else ""
        description = raw_description if isinstance(raw_description, str) else ""

        if source == "title":
            texts.append(title)
        elif source == "description":
            texts.append(description)
        elif source == "both":
            texts.append(f"{title} {description}")
        else:
            raise ValueError(f"Unsupported corpus source: {source}")

    raw_text = " ".join(texts)
    return raw_text


def clean_text(text: str) -> str:
    """Simple cleaning for wordcloud text."""
    # Remove URLs.
    text = re.sub(r"http\S+", " ", text)

    # Replace actual newlines with spaces.
    text = re.sub(r"[\r\n]+", " ", text)

    return text


def get_custom_stopwords() -> Set[str]:
    """Create a combined stopword set for word cloud generation."""
    stopwords = set(STOPWORDS)

    custom_stopwords = {
        # Domain-specific.
        "asmr", "ASMR",

        # Basic English stopwords.
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "could", "couldn't", "did",
        "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each",
        "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
        "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
        "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
        "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
        "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
        "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
        "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
        "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
        "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
        "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
        "those", "through", "to", "too", "under", "until", "up", "very", "was",
        "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
        "what's", "when", "when's", "where", "where's", "which", "while", "who",
        "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
        "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
        "yourselves",

        # Social media fillers.
        "thanks", "thank", "thankyou", "thanksgiving", "subscribe", "sub", "follow",
        "like", "likes", "watch", "watching", "video", "videos", "link", "please",
        "dm", "instagram", "tiktok", "channel",

        # French fillers.
        "le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "dans", "ce",
        "ces", "je", "tu", "que", "qui", "au", "aux", "pour", "mais",
    }
    stopwords.update(custom_stopwords)

    # Single-letter words.
    single_letters = {chr(i) for i in range(ord("a"), ord("z") + 1)}
    single_letters |= {chr(i) for i in range(ord("A"), ord("Z") + 1)}
    stopwords.update(single_letters)

    # Punctuation tokens.
    punctuation_tokens = {
        ".", ",", "!", "?", ":", ";",
        "-", "_", "(", ")", "[", "]", "{", "}",
        "'", '"', "/", "\\", "|", "&", "*", "#", "@",
        "...", "..",
    }
    stopwords.update(punctuation_tokens)

    # Digits.
    digits = {str(i) for i in range(10)}
    stopwords.update(digits)

    return stopwords


def generate_wordcloud_image(text: str, stopwords: Set[str]):
    """Generate a word cloud image array from text."""
    wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        stopwords=stopwords,
        collocations=False,  # Treat word pairs separately.
    ).generate(text)

    img = wordcloud.to_array()
    return img


def create_plotly_figure(img, title: str = "") -> Any:
    """Create a Plotly figure to display the word cloud image."""
    fig = px.imshow(img)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def save_plotly_figure(
    fig: Any,
    filename: str,
    width: int = 1600,
    height: int = 900,
    scale: int = SCALE,
    save_final: bool = True,
    save_png: bool = True,
    save_eps: bool = True,
) -> None:
    """Save a Plotly figure as HTML, PNG, and EPS formats."""
    output_final = os.path.join(common.root_dir, "figures")
    os.makedirs(common.output_dir, exist_ok=True)
    os.makedirs(output_final, exist_ok=True)

    # Save as HTML.
    logger.info("Saving HTML file for %s.", filename)
    py.offline.plot(
        fig,
        filename=os.path.join(common.output_dir, filename + ".html"),
        auto_open=True,
    )

    if save_final:
        py.offline.plot(
            fig,
            filename=os.path.join(output_final, filename + ".html"),
            auto_open=False,
        )

    try:
        if save_png:
            logger.info("Saving PNG file for %s.", filename)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Support for Kaleido versions less than 1.0.0.*",
                    category=DeprecationWarning,
                )
                fig.write_image(
                    os.path.join(common.output_dir, filename + ".png"),
                    width=width,
                    height=height,
                    scale=scale,
                )
            if save_final:
                shutil.copy(
                    os.path.join(common.output_dir, filename + ".png"),
                    os.path.join(output_final, filename + ".png"),
                )

        if save_eps:
            logger.info("Saving EPS file for %s.", filename)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Support for Kaleido versions less than 1.0.0.*",
                    category=DeprecationWarning,
                )
                fig.write_image(
                    os.path.join(common.output_dir, filename + ".eps"),
                    width=width,
                    height=height,
                )
            if save_final:
                shutil.copy(
                    os.path.join(common.output_dir, filename + ".eps"),
                    os.path.join(output_final, filename + ".eps"),
                )
    except ValueError as exc:
        logger.error(
            "Value error raised when attempted to save image %s: %s",
            filename,
            exc,
        )


def run_wordcloud_pipeline(data: Dict[str, Any]) -> None:
    """Run the three word clouds (title / description / both)."""
    stopwords = get_custom_stopwords()

    configs = [
        {"source": "title", "filename": "wordcloud_titles"},
        {"source": "description", "filename": "wordcloud_descriptions"},
        {"source": "both", "filename": "wordcloud_titles_descriptions"},
    ]

    for cfg in configs:
        logger.info("Generating word cloud for source='%s'.", cfg["source"])

        raw_text = build_corpus(data, source=cfg["source"])
        cleaned_text = clean_text(raw_text)

        img = generate_wordcloud_image(cleaned_text, stopwords)
        fig = create_plotly_figure(img)

        save_plotly_figure(
            fig=fig,
            filename=cfg["filename"],
            width=1600,
            height=900,
            scale=SCALE,
            save_final=True,
            save_png=True,
            save_eps=True,
        )

# ============================================================================
# PART 2 — ANALYTICS & CLUSTERING
# ============================================================================


def _parse_upload_datetime(upload_date_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 uploadDate string into a timezone-aware datetime."""
    if not isinstance(upload_date_str, str) or not upload_date_str:
        return None
    try:
        if upload_date_str.endswith("Z"):
            return datetime.fromisoformat(upload_date_str.replace("Z", "+00:00"))
        return datetime.fromisoformat(upload_date_str)
    except Exception as exc:
        logger.warning("Could not parse uploadDate '%s': %s", upload_date_str, exc)
        return None


def _month_to_season(month: Optional[int]) -> str:
    """Map month to a simple meteorological season."""
    if month is None or pd.isna(month):
        return "unknown"
    m = int(month)
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    if m in (9, 10, 11):
        return "autumn"
    return "unknown"


def _duration_bucket(minutes: float) -> str:
    """Bucket video duration."""
    if pd.isna(minutes):
        return "unknown"
    if minutes < 5:
        return "short"
    if 10 <= minutes <= 20:
        return "normal"
    if 60 <= minutes <= 180:
        return "long"
    return "other"


def json_to_dataframe(
    data: Dict[str, Any],
    reference_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Convert the raw JSON dict into a pandas DataFrame with derived fields."""
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)

    rows = []
    for video_id, info in data.items():
        title = info.get("title") or ""
        description = info.get("description") or ""
        duration = info.get("duration")
        channel_id = info.get("channelId")
        author = info.get("author")
        views = info.get("views")
        likes = info.get("likes")
        language = info.get("language") or "unknown"
        upload_date_str = info.get("uploadDate")
        channel_avg_views = info.get("channel_average_views")

        upload_dt = _parse_upload_datetime(upload_date_str)
        if upload_dt is not None:
            days_since_upload = (reference_date - upload_dt).total_seconds() / 86400.0
            days_since_upload = max(days_since_upload, 1e-6)
        else:
            days_since_upload = np.nan

        rows.append(
            {
                "video_id": video_id,
                "title": title,
                "description": description,
                "language": language,
                "views": views,
                "likes": likes,
                "duration_seconds": duration,
                "channel_id": channel_id,
                "author": author,
                "upload_datetime": upload_dt,
                "days_since_upload": days_since_upload,
                "channel_average_views": channel_avg_views,
            }
        )

    df = pd.DataFrame(rows)

    # Ensure upload_datetime is datetimelike
    df["upload_datetime"] = pd.to_datetime(
        df["upload_datetime"], errors="coerce", utc=True
    )

    # Numeric conversions.
    df["views"] = pd.to_numeric(df["views"], errors="coerce")
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce")
    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")
    df["channel_average_views"] = pd.to_numeric(
        df["channel_average_views"], errors="coerce"
    )

    # Duration metrics & bucket.
    df["duration_minutes"] = df["duration_seconds"] / 60.0
    df["duration_bucket"] = df["duration_minutes"].apply(_duration_bucket)

    # Engagement metrics.
    df["engagement_rate"] = np.where(
        df["views"] > 0,
        df["likes"] / df["views"],
        np.nan,
    )
    df["views_per_day"] = np.where(
        df["days_since_upload"] > 0,
        df["views"] / df["days_since_upload"],
        np.nan,
    )
    df["rel_views_vs_channel_avg"] = np.where(
        (df["channel_average_views"] > 0) & df["channel_average_views"].notna(),
        df["views"] / df["channel_average_views"],
        np.nan,
    )

    # Time-based fields.
    df["upload_year"] = df["upload_datetime"].dt.year  # type: ignore
    df["upload_month"] = df["upload_datetime"].dt.month  # type: ignore
    df["upload_day"] = df["upload_datetime"].dt.day  # type: ignore
    df["upload_date"] = df["upload_datetime"].dt.date  # type: ignore
    df["upload_season"] = df["upload_month"].apply(_month_to_season)

    # Title style & themes.
    df = add_title_style_features(df)
    df = add_theme_flags(df)

    # Growth category.
    df = add_growth_category(df)

    return df


def add_title_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features capturing 'title style'."""
    titles = df["title"].fillna("")

    df["title_word_count"] = titles.str.split().str.len()
    df["title_char_count"] = titles.str.len()
    df["title_has_brackets"] = titles.str.contains(r"[\[\]\(\)]", regex=True)
    df["title_has_all_caps_word"] = titles.str.contains(r"\b[A-Z]{3,}\b")
    df["title_has_exclamation"] = titles.str.contains("!")
    df["title_has_question"] = titles.str.contains(r"\?")
    df["title_has_hashtag"] = titles.str.contains("#")
    # Use non-capturing group to avoid regex group warning
    df["title_has_no_talking_tag"] = titles.str.contains(
        r"no[-\s]?talk(?:ing)?", case=False, regex=True
    )

    return df


THEME_KEYWORDS = {
    "has_whisper": ["whisper", "whispered", "whispering", "susurro", "susurros", "chuchotement", "flüstern"],
    "has_no_talking": ["no talking", "no-talking", "no talk", "sin hablar", "sans parler", "senza parlare"],
    "has_sleep": ["sleep", "sleepy", "for sleep", "insomnia", "insomnio", "dormir", "sommeil", "para dormir"],
    "has_binaural": ["binaural", "3dio", "3d sound", "3d audio", "8d audio", "8d sound"],
    "has_roleplay": ["roleplay", "rp ", " rp", "doctor roleplay", "nurse roleplay", "medical roleplay",
                     "exam", "check up", "check-up", "haircut", "barber", "spa roleplay"],
    "has_ear_cleaning": ["ear cleaning", "ear exam", "ear massage", "ear attention", "ear brushing", "otoscope"],
    "has_mukbang": ["mukbang", "eating asmr", "eating sounds", "chewing", "eating show"],
    "has_keyboard": ["keyboard", "typing", "key sounds", "mechanical keyboard"],
    "has_visual": ["visual triggers", "hand movements", "visuals", "slow movements", "trigger assortment"],
}


def add_theme_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean columns for content themes based on title+description."""
    text_all = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()  # type: ignore
    for col, keywords in THEME_KEYWORDS.items():
        pattern = "|".join(map(re.escape, keywords))
        df[col] = text_all.str.contains(pattern, regex=True)
    return df


def add_growth_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'growth_category' based on views_per_day quantiles."""
    vpd = df["views_per_day"]
    if vpd.notna().sum() == 0:
        df["growth_category"] = "unknown"
        return df

    fast_thr = vpd.quantile(0.8)
    slow_thr = vpd.quantile(0.2)

    def _cat(x):
        if pd.isna(x):
            return "unknown"
        if x >= fast_thr:
            return "fast_growth"
        if x <= slow_thr:
            return "slow_growth"
        return "medium"

    df["growth_category"] = vpd.apply(_cat)
    return df


def summarize_by_duration_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Are long videos more popular / more engaging?"""
    agg = (
        df.groupby("duration_bucket")
        .agg(
            video_count=("video_id", "count"),
            mean_views=("views", "mean"),
            median_views=("views", "median"),
            mean_views_per_day=("views_per_day", "mean"),
            mean_engagement_rate=("engagement_rate", "mean"),
        )
        .reset_index()
    )
    return agg


def summarize_by_language(df: pd.DataFrame) -> pd.DataFrame:
    """Which languages get more engagement / growth?"""
    agg = (
        df.groupby("language")
        .agg(
            video_count=("video_id", "count"),
            mean_views=("views", "mean"),
            median_views=("views", "median"),
            mean_views_per_day=("views_per_day", "mean"),
            mean_engagement_rate=("engagement_rate", "mean"),
        )
        .reset_index()
    )
    return agg


def summarize_title_styles(df: pd.DataFrame) -> pd.DataFrame:
    """Compare engagement across 'title style' bins."""
    df_copy = df.copy()
    df_copy["title_length_bucket"] = pd.cut(
        df_copy["title_word_count"],
        bins=[0, 5, 10, 20, 1000],
        labels=["<=5 words", "6–10 words", "11–20 words", ">20 words"],
        include_lowest=True,
    )

    agg = (
        df_copy.groupby("title_length_bucket")
        .agg(
            video_count=("video_id", "count"),
            mean_views=("views", "mean"),
            mean_engagement_rate=("engagement_rate", "mean"),
        )
        .reset_index()
    )
    return agg


def summarize_theme_vs_growth(df: pd.DataFrame, theme_col: str) -> pd.DataFrame:
    """Compare views_per_day for videos with vs without a given theme flag."""
    if theme_col not in df.columns:
        raise ValueError(f"Unknown theme column: {theme_col}")

    agg = (
        df.groupby(theme_col)["views_per_day"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .reset_index()
    )
    return agg


def compute_monthly_video_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Number of ASMR videos per year/month."""
    tmp = df.dropna(subset=["upload_year", "upload_month"])

    monthly = (
        tmp.groupby(["upload_year", "upload_month"])
        .size()
        .rename("video_count")
        .reset_index()
    )

    monthly["upload_year"] = monthly["upload_year"].astype(int)
    monthly["upload_month"] = monthly["upload_month"].astype(int)

    year_str = monthly["upload_year"].astype(str)
    month_str = monthly["upload_month"].astype(str).str.zfill(2)

    monthly["year_month"] = pd.to_datetime(
        year_str + "-" + month_str + "-01",
        errors="coerce",
    )

    return monthly


def compute_language_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Growth of ASMR per language (e.g. Spanish vs English)."""
    growth = (
        df.groupby(["upload_year", "language"])
        .size()
        .rename("video_count")
        .reset_index()
    )
    return growth


def compute_theme_trend_over_time(
    df: pd.DataFrame,
    theme_col: str,
    by_language: bool = False,
) -> pd.DataFrame:
    """Share of videos with a given theme per year (optionally per language)."""
    if theme_col not in df.columns:
        raise ValueError(f"Unknown theme column: {theme_col}")

    if by_language:
        trend = (
            df.groupby(["upload_year", "language"])[theme_col]
            .mean()
            .reset_index(name=f"{theme_col}_share")
        )
    else:
        trend = (
            df.groupby("upload_year")[theme_col]
            .mean()
            .reset_index(name=f"{theme_col}_share")
        )
    return trend


def compute_seasonal_sleep_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Are 'sleep' videos more common in winter?"""
    if "has_sleep" not in df.columns:
        raise ValueError("Theme flag 'has_sleep' not present.")

    agg = (
        df.groupby("upload_season")["has_sleep"]
        .mean()
        .reset_index(name="sleep_share")
    )
    agg["sleep_share"] = agg["sleep_share"].astype(float)
    return agg


def cluster_videos(
    df: pd.DataFrame,
    n_clusters: int = 10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Optional[Pipeline]]:
    """Cluster videos using title text, duration, engagement, and language.

    Also computes a 2D embedding (PCA) and stores it in columns
    'embedding_x' and 'embedding_y' for visualization.
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn is not installed; clustering skipped.")
        df_copy = df.copy()
        df_copy["cluster"] = -1
        return df_copy, None

    df_copy = df.copy()
    df_copy["text_all"] = (
        df_copy["title"].fillna("") + " " + df_copy["description"].fillna("")  # type: ignore
    )

    feature_cols = ["text_all", "duration_minutes", "engagement_rate", "views_per_day", "language"]

    # Ensure numeric columns exist and fill NaNs for clustering.
    for col in ["duration_minutes", "engagement_rate", "views_per_day"]:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0.0)

    preprocess = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=5,
                ),
                "text_all",
            ),
            (
                "numeric",
                StandardScaler(with_mean=False),
                ["duration_minutes", "engagement_rate", "views_per_day"],
            ),
            (
                "lang",
                OneHotEncoder(handle_unknown="ignore"),
                ["language"],
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "cluster",
                KMeans(
                    n_clusters=n_clusters,
                    random_state=random_state,
                    n_init=10,
                ),
            ),
        ]
    )

    X = df_copy[feature_cols]
    logger.info("Fitting clustering model on %d videos...", len(df_copy))
    pipeline.fit(X)

    logger.info("Assigning cluster labels...")
    df_copy["cluster"] = pipeline.predict(X)

    # ---- 2D embedding for visualization (PCA) ----
    try:
        logger.info("Computing 2D PCA embedding for cluster visualization...")
        # Use the fitted preprocessor to get feature vectors
        features = pipeline.named_steps["preprocess"].transform(X)

        reducer = PCA(n_components=2)
        embedding_2d = reducer.fit_transform(features)

        df_copy["embedding_x"] = embedding_2d[:, 0]
        df_copy["embedding_y"] = embedding_2d[:, 1]
    except Exception as exc:
        logger.warning("Could not compute 2D embedding for clusters: %s", exc)
        df_copy["embedding_x"] = np.nan
        df_copy["embedding_y"] = np.nan

    return df_copy, pipeline

# ---------------------------------------------------------------------------
# Plotly figure helpers for analytics
# ---------------------------------------------------------------------------


def plot_duration_stats(duration_stats: pd.DataFrame) -> None:
    """Create Plotly figures for duration bucket stats."""
    if duration_stats.empty:
        return

    df_plot = duration_stats.copy()
    order = ["short", "normal", "long", "other", "unknown"]
    df_plot["duration_bucket"] = df_plot["duration_bucket"].astype(str)
    df_plot["duration_bucket"] = pd.Categorical(df_plot["duration_bucket"], categories=order, ordered=True)
    df_plot = df_plot.sort_values("duration_bucket")

    fig = px.bar(
        df_plot,
        x="duration_bucket",
        y="mean_views",
        title="Mean views by duration bucket",
        labels={"duration_bucket": "Duration bucket", "mean_views": "Mean views"},
    )
    save_plotly_figure(fig, "duration_mean_views", width=1600, height=900, scale=SCALE)

    fig = px.bar(
        df_plot,
        x="duration_bucket",
        y="mean_views_per_day",
        title="Mean views per day by duration bucket",
        labels={"duration_bucket": "Duration bucket", "mean_views_per_day": "Mean views per day"},
    )
    save_plotly_figure(fig, "duration_mean_views_per_day", width=1600, height=900, scale=SCALE)

    fig = px.bar(
        df_plot,
        x="duration_bucket",
        y="mean_engagement_rate",
        title="Mean engagement rate by duration bucket",
        labels={"duration_bucket": "Duration bucket", "mean_engagement_rate": "Mean engagement rate (likes / views)"},
    )
    save_plotly_figure(fig, "duration_mean_engagement_rate", width=1600, height=900, scale=SCALE)


def plot_language_stats(lang_stats: pd.DataFrame, min_videos: int = 20) -> None:
    """Plot language-level engagement / growth statistics."""
    if lang_stats.empty:
        return

    df_plot = lang_stats.copy()
    df_plot = df_plot[df_plot["video_count"] >= min_videos]
    if df_plot.empty:
        return

    df_plot = df_plot.sort_values("mean_views_per_day", ascending=False)

    fig = px.bar(
        df_plot,
        x="language",
        y="mean_views_per_day",
        title=f"Mean views per day by language (>= {min_videos} videos)",
        labels={"language": "Language", "mean_views_per_day": "Mean views per day"},
    )
    save_plotly_figure(fig, "language_mean_views_per_day", width=1600, height=900, scale=SCALE)

    fig = px.bar(
        df_plot,
        x="language",
        y="mean_engagement_rate",
        title=f"Mean engagement rate by language (>= {min_videos} videos)",
        labels={"language": "Language", "mean_engagement_rate": "Mean engagement rate (likes / views)"},
    )
    save_plotly_figure(fig, "language_mean_engagement_rate", width=1600, height=900, scale=SCALE)


def plot_title_style_stats(title_stats: pd.DataFrame) -> None:
    """Plot engagement vs title length buckets."""
    if title_stats.empty:
        return

    df_plot = title_stats.copy()
    order = ["<=5 words", "6–10 words", "11–20 words", ">20 words"]
    df_plot["title_length_bucket"] = pd.Categorical(df_plot["title_length_bucket"], categories=order, ordered=True)
    df_plot = df_plot.sort_values("title_length_bucket")

    fig = px.bar(
        df_plot,
        x="title_length_bucket",
        y="mean_engagement_rate",
        title="Mean engagement rate by title length",
        labels={"title_length_bucket": "Title length", "mean_engagement_rate": "Mean engagement rate (likes / views)"},
    )
    save_plotly_figure(fig, "title_length_mean_engagement_rate", width=1600, height=900, scale=SCALE)

    fig = px.bar(
        df_plot,
        x="title_length_bucket",
        y="mean_views",
        title="Mean views by title length",
        labels={"title_length_bucket": "Title length", "mean_views": "Mean views"},
    )
    save_plotly_figure(fig, "title_length_mean_views", width=1600, height=900, scale=SCALE)


def plot_theme_growth_box(df: pd.DataFrame, theme_col: str) -> None:
    """Boxplot of views_per_day for videos with/without a given theme."""
    if theme_col not in df.columns:
        return

    df_plot = df[["views_per_day", theme_col]].dropna(subset=["views_per_day"])
    if df_plot.empty:
        return

    fig = px.box(
        df_plot,
        x=theme_col,
        y="views_per_day",
        title=f"Views per day distribution by theme: {theme_col}",
        labels={theme_col: f"{theme_col} (False / True)", "views_per_day": "Views per day"},
    )
    filename = f"{theme_col}_views_per_day_boxplot"
    save_plotly_figure(fig, filename, width=1600, height=900, scale=SCALE)


def plot_monthly_counts(monthly_counts: pd.DataFrame) -> None:
    """Line chart for number of videos per month (community growth)."""
    if monthly_counts.empty:
        return

    df_plot = monthly_counts.copy()
    df_plot = df_plot.sort_values("year_month")

    fig = px.line(
        df_plot,
        x="year_month",
        y="video_count",
        title="Number of ASMR videos per month",
        labels={"year_month": "Month", "video_count": "Number of videos"},
    )
    save_plotly_figure(fig, "monthly_video_counts", width=1600, height=900, scale=SCALE)


def plot_language_growth(lang_growth: pd.DataFrame, min_total_videos: int = 50) -> None:
    """Line chart for growth of ASMR per language (videos per year)."""
    if lang_growth.empty:
        return

    df_plot = lang_growth.copy()
    totals = df_plot.groupby("language")["video_count"].sum().reset_index()
    keep_langs = totals[totals["video_count"] >= min_total_videos]["language"]
    df_plot = df_plot[df_plot["language"].isin(keep_langs)]
    if df_plot.empty:
        return

    df_plot = df_plot.sort_values(["language", "upload_year"])

    fig = px.line(
        df_plot,
        x="upload_year",
        y="video_count",
        color="language",
        markers=True,
        title=f"ASMR video uploads per year by language (>= {min_total_videos} videos total)",
        labels={"upload_year": "Year", "video_count": "Number of videos", "language": "Language"},
    )
    save_plotly_figure(fig, "language_growth_over_years", width=1600, height=900, scale=SCALE)


def plot_theme_trend_overall(trend_df: pd.DataFrame, theme_col: str) -> None:
    """Trend of theme share over years (all languages combined)."""
    if trend_df.empty:
        return

    df_plot = trend_df.copy()
    share_col = f"{theme_col}_share"

    fig = px.line(
        df_plot,
        x="upload_year",
        y=share_col,
        title=f"Share of videos with theme '{theme_col}' over time (all languages)",
        labels={"upload_year": "Year", share_col: "Share of videos"},
        markers=True,
    )
    filename = f"{theme_col}_trend_overall_fig"
    save_plotly_figure(fig, filename, width=1600, height=900, scale=SCALE)


def plot_theme_trend_by_language(trend_df: pd.DataFrame, theme_col: str, min_videos: int = 30) -> None:
    """Trend of theme share over years by language."""
    if trend_df.empty:
        return

    df_plot = trend_df.copy()
    share_col = f"{theme_col}_share"

    counts = df_plot.groupby("language")[share_col].count().reset_index(name="n")
    keep_langs = counts[counts["n"] >= min_videos]["language"]
    df_plot = df_plot[df_plot["language"].isin(keep_langs)]
    if df_plot.empty:
        return

    fig = px.line(
        df_plot,
        x="upload_year",
        y=share_col,
        color="language",
        markers=True,
        title=f"Share of videos with theme '{theme_col}' over time by language",
        labels={"upload_year": "Year", share_col: "Share of videos", "language": "Language"},
    )
    filename = f"{theme_col}_trend_by_language_fig"
    save_plotly_figure(fig, filename, width=1600, height=900, scale=SCALE)


def plot_sleep_seasonal(sleep_seasonal: pd.DataFrame) -> None:
    """Bar chart: share of 'sleep' videos by season."""
    if sleep_seasonal.empty:
        return

    df_plot = sleep_seasonal.copy()
    order = ["winter", "spring", "summer", "autumn", "unknown"]
    df_plot["upload_season"] = pd.Categorical(df_plot["upload_season"], categories=order, ordered=True)
    df_plot = df_plot.sort_values("upload_season")

    fig = px.bar(
        df_plot,
        x="upload_season",
        y="sleep_share",
        title="Share of 'sleep' ASMR videos by season",
        labels={"upload_season": "Season", "sleep_share": "Share of videos tagged as sleep"},
    )
    save_plotly_figure(fig, "sleep_seasonal_pattern_fig", width=1600, height=900, scale=SCALE)


def plot_cluster_distribution(clustered_df: pd.DataFrame) -> None:
    """Visualize clusters: bar charts + 2D scatter with circles around clusters."""
    if "cluster" not in clustered_df.columns:
        return

    df_plot = clustered_df.copy()

    # --- 1) Bar charts: sizes and mean views ---
    agg = (
        df_plot.groupby("cluster")
        .agg(
            video_count=("video_id", "count"),
            mean_views=("views", "mean"),
            mean_views_per_day=("views_per_day", "mean"),
        )
        .reset_index()
    )

    # Cluster sizes
    fig = px.bar(
        agg,
        x="cluster",
        y="video_count",
        title="Number of videos per cluster",
        labels={"cluster": "Cluster", "video_count": "Number of videos"},
    )
    save_plotly_figure(fig, "cluster_sizes", width=1600, height=900, scale=SCALE)

    # Mean views per day per cluster
    fig = px.bar(
        agg,
        x="cluster",
        y="mean_views_per_day",
        title="Mean views per day by cluster",
        labels={"cluster": "Cluster", "mean_views_per_day": "Mean views per day"},
    )
    save_plotly_figure(fig, "cluster_mean_views_per_day", width=1600, height=900, scale=SCALE)

    # --- 2) 2D scatter embedding with circles ---
    if "embedding_x" not in df_plot.columns or "embedding_y" not in df_plot.columns:
        logger.warning("No embedding_x / embedding_y columns found; skipping cluster scatter plot.")
        return

    df_emb = df_plot.dropna(subset=["embedding_x", "embedding_y"]).copy()
    if df_emb.empty:
        logger.warning("Embedding columns are empty; skipping cluster scatter plot.")
        return

    # Scatter of videos in 2D embedding space
    fig = px.scatter(
        df_emb,
        x="embedding_x",
        y="embedding_y",
        color="cluster",
        hover_data=["video_id", "title", "language", "views", "duration_minutes"],
        title="ASMR video clusters (2D embedding of text, duration, engagement, language)",
        labels={
            "embedding_x": "Embedding axis 1 (text + duration + engagement + language)",
            "embedding_y": "Embedding axis 2 (text + duration + engagement + language)",
            "cluster": "Cluster",
        },
    )

    # Add circles around each cluster (based on centroid + radius)
    shapes = []
    for cluster_id, group in df_emb.groupby("cluster"):
        if len(group) < 2:
            continue

        cx = group["embedding_x"].mean()
        cy = group["embedding_y"].mean()
        distances = np.sqrt(
            (group["embedding_x"] - cx) ** 2 + (group["embedding_y"] - cy) ** 2
        )

        # Use 80th percentile as a "cluster radius"
        radius = float(distances.quantile(0.8))  # type: ignore
        if not np.isfinite(radius) or radius <= 0:
            continue

        shapes.append(
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=cx - radius,
                y0=cy - radius,
                x1=cx + radius,
                y1=cy + radius,
                line=dict(width=1, dash="dot"),
                opacity=0.3,
            )
        )

    if shapes:
        fig.update_layout(shapes=shapes)

    save_plotly_figure(fig, "cluster_scatter_embedding", width=1600, height=900, scale=SCALE)

# ---------------------------------------------------------------------------
# Analytics pipeline: CSVs + figures
# ---------------------------------------------------------------------------


def run_analytics_pipeline(data: Dict[str, Any]) -> None:
    """Run all analytics, write CSVs into output/analysis, and create Plotly figures."""
    analysis_dir = os.path.join(common.output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    df = json_to_dataframe(data)

    # Enriched dataset
    enriched_csv = os.path.join(analysis_dir, "asmr_videos_enriched.csv")
    logger.info("Saving enriched dataset to %s", enriched_csv)
    df.to_csv(enriched_csv, index=False)

    # Duration stats
    duration_stats = summarize_by_duration_bucket(df)
    duration_stats.to_csv(
        os.path.join(analysis_dir, "duration_stats.csv"),
        index=False,
    )
    plot_duration_stats(duration_stats)

    # Language stats
    lang_stats = summarize_by_language(df)
    lang_stats.to_csv(
        os.path.join(analysis_dir, "language_stats.csv"),
        index=False,
    )
    plot_language_stats(lang_stats)

    # Title style stats
    title_stats = summarize_title_styles(df)
    title_stats.to_csv(
        os.path.join(analysis_dir, "title_style_stats.csv"),
        index=False,
    )
    plot_title_style_stats(title_stats)

    # Theme vs growth (boxplots)
    for theme in ["has_whisper", "has_no_talking", "has_sleep", "has_binaural"]:
        if theme in df.columns:
            theme_stats = summarize_theme_vs_growth(df, theme)
            theme_stats.to_csv(
                os.path.join(analysis_dir, f"{theme}_growth_stats.csv"),
                index=False,
            )
            plot_theme_growth_box(df, theme)

    # Community growth over time
    monthly_counts = compute_monthly_video_counts(df)
    monthly_counts.to_csv(
        os.path.join(analysis_dir, "monthly_video_counts.csv"),
        index=False,
    )
    plot_monthly_counts(monthly_counts)

    lang_growth = compute_language_growth(df)
    lang_growth.to_csv(
        os.path.join(analysis_dir, "language_growth.csv"),
        index=False,
    )
    plot_language_growth(lang_growth)

    # Theme trends over time
    for theme in ["has_no_talking", "has_binaural"]:
        if theme in df.columns:
            trend_all = compute_theme_trend_over_time(df, theme_col=theme, by_language=False)
            trend_lang = compute_theme_trend_over_time(df, theme_col=theme, by_language=True)

            trend_all.to_csv(
                os.path.join(analysis_dir, f"{theme}_trend_overall.csv"),
                index=False,
            )
            trend_lang.to_csv(
                os.path.join(analysis_dir, f"{theme}_trend_by_language.csv"),
                index=False,
            )

            plot_theme_trend_overall(trend_all, theme)
            plot_theme_trend_by_language(trend_lang, theme)

    # Seasonal pattern for sleep videos
    if "has_sleep" in df.columns:
        sleep_seasonal = compute_seasonal_sleep_pattern(df)
        sleep_seasonal.to_csv(
            os.path.join(analysis_dir, "sleep_seasonal_pattern.csv"),
            index=False,
        )
        plot_sleep_seasonal(sleep_seasonal)

    # Clustering
    clustered_df, _ = cluster_videos(df, n_clusters=12)
    clustered_df.to_csv(
        os.path.join(analysis_dir, "asmr_videos_with_clusters.csv"),
        index=False,
    )
    plot_cluster_distribution(clustered_df)

    logger.info("Analytics pipeline complete. CSVs and figures written to %s", analysis_dir)

# ============================================================================
# MAIN — run wordclouds + analytics
# ============================================================================


def main() -> None:
    # Path to JSON from your config.
    json_path = os.path.join(common.get_configs("data"), "asmr_results.json")
    logger.info("Loading ASMR data from %s", json_path)
    data = load_asmr_data(json_path)

    # 1) Original wordcloud pipeline.
    run_wordcloud_pipeline(data)

    # 2) Analytics & clustering + figures.
    run_analytics_pipeline(data)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
