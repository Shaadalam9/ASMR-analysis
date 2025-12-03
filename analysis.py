import json
import logging
import os
import re
import shutil
import warnings
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly as py
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import common
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

import spacy
from scipy import stats


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

# Choose which text source to use for text-based analyses:
#   "title"       -> titles only
#   "description" -> descriptions only
#   "both"        -> title + description
TEXT_SOURCE = common.get_configs("analysis_text_source")

# Default scaling factor for saved PNG images.
SCALE = 3

font_family = common.get_configs("font_family")
font_size = common.get_configs("font_size")

logger = logging.getLogger(__name__)

_NLP_CACHE: Optional["spacy.language.Language"] = None  # type: ignore[valid-type]

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"plotly\.io\._kaleido",
)


# ============================================================================
# Language normalization
# ============================================================================

_LANGUAGE_MAP: Dict[str, str] = {
    # English variants
    "en": "English",
    "eng": "English",
    "en-us": "English",
    "en-gb": "English",

    # Japanese
    "jp": "Japanese",
    "ja": "Japanese",

    # Spanish
    "es": "Spanish",
    "es-es": "Spanish",
    "es-mx": "Spanish (Mexico)",

    # Major European languages
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pt-br": "Portuguese (Brazil)",
    "pt-pt": "Portuguese (Portugal)",

    # Asian languages
    "ru": "Russian",
    "ko": "Korean",
    "kr": "Korean",
    "zh": "Chinese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",

    # Northern European
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",

    # Other common
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "hi": "Hindi",
    "id": "Indonesian",
    "th": "Thai",
    "vi": "Vietnamese",

    # Central / Eastern / misc.
    "cs": "Czech",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "bg": "Bulgarian",

    # Tagalog / Filipino
    "tl": "Tagalog / Filipino",

    # Fallback
    "unknown": "Unknown",
}


def normalize_language_code(lang: Any) -> str:
    """Map short codes (en, jp, tl, ...) to human-readable language names."""
    if not isinstance(lang, str):
        code = "unknown"
    else:
        code = lang.strip().lower() or "unknown"

    label = _LANGUAGE_MAP.get(code)
    if label is not None:
        return label

    # Fallback: title-case whatever is left
    return code.title()


# ============================================================================
# Shared: load JSON
# ============================================================================

def load_asmr_data(json_path: str) -> Dict[str, Any]:
    """Load ASMR results from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded JSON with {len(data)} video entries from {json_path}")
    return data


# ============================================================================
# PART 1 — WORDCLOUD PIPELINE
# ============================================================================

def build_corpus(data: Dict[str, Any], source: str) -> str:
    """Build a text corpus from titles, descriptions, or both."""
    texts = []

    for _, info in data.items():
        raw_title = info.get("title")
        raw_description = info.get("description")

        title = raw_title if isinstance(raw_title, str) else ""
        description = raw_description if isinstance(raw_description, str) else ""

        if source == "title":
            texts.append(title)
        elif source == "description":
            texts.append(description)
        elif source == "both":
            texts.append(f"{title} {description}")
        else:
            raise ValueError(f"Unsupported corpus source: {source!r}")

    raw_text = " ".join(texts)
    logger.info(
        f"Built text corpus from {len(texts)} videos using source='{source}', "
        f"total characters={len(raw_text)}"
    )
    return raw_text


def clean_text(text: str) -> str:
    """Simple cleaning for wordcloud / spaCy text."""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[\r\n]+", " ", text)
    return text


def get_custom_stopwords() -> Set[str]:
    """Create a combined stopword set for word cloud and spaCy lemma filtering."""
    stopwords = set(STOPWORDS)

    custom_stopwords = {
        # Domain-specific.
        "asmr", "ASMR", "gmail", "comment", "twitter", "facebook", "patreon", "help",
        "new", "youtube", "enjoy", "let", "spotify", "email",

        # Basic English stopwords.
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before",
        "being", "below", "between", "both", "but", "by", "can", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
        "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
        "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
        "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
        "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves",

        # Social media fillers.
        "thanks", "thank", "thankyou", "thanksgiving", "subscribe", "sub", "follow",
        "like", "likes", "watch", "watching", "video", "videos", "link", "please",
        "dm", "instagram", "tiktok", "channel",

        # French fillers.
        "le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "dans",
        "ce", "ces", "je", "tu", "que", "qui", "au", "aux", "pour", "mais",
    }
    stopwords.update(custom_stopwords)

    single_letters = {chr(i) for i in range(ord("a"), ord("z") + 1)}
    single_letters |= {chr(i) for i in range(ord("A"), ord("Z") + 1)}
    stopwords.update(single_letters)

    punctuation_tokens = {
        ".", ",", "!", "?", ":", ";", "-", "_", "(", ")", "[", "]", "{", "}", "'",
        '"', "/", "\\", "|", "&", "*", "#", "@", "...", "..",
    }
    stopwords.update(punctuation_tokens)

    digits = {str(i) for i in range(10)}
    stopwords.update(digits)

    logger.info(f"Custom stopword set size: {len(stopwords)} tokens")
    return stopwords


def generate_wordcloud_image(text: str, stopwords: Set[str]):
    """Generate a word cloud image array from text."""
    wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        stopwords=stopwords,
        collocations=False,
    ).generate(text)
    img = wordcloud.to_array()
    logger.info(
        f"Generated word cloud with {len(wordcloud.words_)} unique words "
        f"(highest weight word='{next(iter(wordcloud.words_))}' if any)."
    )
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


def save_plotly_figure(fig: Any, filename: str, width: int = 1600, height: int = 900, scale: int = SCALE,
                       save_final: bool = True, save_png: bool = True, save_eps: bool = True) -> None:
    """Save a Plotly figure as HTML, PNG, and EPS formats."""
    output_final = os.path.join(common.root_dir, "figures")
    os.makedirs(common.output_dir, exist_ok=True)
    os.makedirs(output_final, exist_ok=True)

    fig.update_layout(
        template=common.get_configs("plotly_template"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family=font_family,
            size=font_size,
        ),
    )

    fig.update_xaxes(
        tickfont=dict(
            family=font_family,
            size=font_size,
        ),
        title_font=dict(
            family=font_family,
            size=font_size,
        ),
    )
    fig.update_yaxes(
        tickfont=dict(
            family=font_family,
            size=font_size,
        ),
        title_font=dict(
            family=font_family,
            size=font_size,
        ),
    )

    html_path = os.path.join(common.output_dir, filename + ".html")
    py.offline.plot(
        fig,
        filename=html_path,
        auto_open=True,
    )

    if save_final:
        final_html_path = os.path.join(output_final, filename + ".html")
        py.offline.plot(
            fig,
            filename=final_html_path,
            auto_open=False,
        )

    try:
        if save_png:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Support for Kaleido versions less than 1.0.0.*",
                    category=DeprecationWarning,
                )
                png_path = os.path.join(common.output_dir, filename + ".png")
                fig.write_image(
                    png_path,
                    width=width,
                    height=height,
                    scale=scale,
                )
            if save_final:
                final_png_path = os.path.join(output_final, filename + ".png")
                shutil.copy(png_path, final_png_path)

        if save_eps:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                eps_path = os.path.join(common.output_dir, filename + ".eps")
                fig.write_image(
                    eps_path,
                    width=width,
                    height=height,
                )
            if save_final:
                final_eps_path = os.path.join(output_final, filename + ".eps")
                shutil.copy(eps_path, final_eps_path)
    except ValueError as exc:
        logger.error(
            f"Value error raised when attempted to save image {filename}: {exc}"
        )


def run_wordcloud_pipeline(data: Dict[str, Any], text_source: str = "both") -> None:
    """Run a word cloud for the chosen text source (title / description / both)."""
    stopwords = get_custom_stopwords()

    logger.info(f"Generating word cloud for text_source='{text_source}'")
    raw_text = build_corpus(data, source=text_source)
    cleaned_text = clean_text(raw_text)

    img = generate_wordcloud_image(cleaned_text, stopwords)
    fig = create_plotly_figure(img, title="")

    filename = f"wordcloud_{text_source}"
    save_plotly_figure(
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


# ============================================================================
# spaCy helpers: model + lemma keyword counts
# ============================================================================

def get_spacy_nlp(model_name: str = "en_core_web_sm"):
    """Lazy-load and cache a spaCy model."""
    global _NLP_CACHE
    if _NLP_CACHE is not None:
        return _NLP_CACHE

    try:
        nlp = spacy.load(model_name, disable=["parser", "ner"])
        nlp.max_length = max(nlp.max_length, 2_000_000)
        logger.info(f"Loaded spaCy model '{model_name}' with max_length={nlp.max_length}")
    except Exception as exc:
        logger.warning(f"Could not load spaCy model '{model_name}': {exc}")
        return None

    _NLP_CACHE = nlp
    return nlp


def compute_spacy_keyword_counts(data: Dict[str, Any], target_lemmas: Optional[Set[str]] = None,
                                 text_source: str = "both", model_name: str = "en_core_web_sm",
                                 top_k: int = 30, extra_stopwords: Optional[Set[str]] = None) -> pd.DataFrame:
    """
    Count lemmas across videos using spaCy. Each lemma is counted
    at most once per video.
    """
    nlp = get_spacy_nlp(model_name)
    if nlp is None:
        return pd.DataFrame(columns=["lemma", "count"])

    texts: list[str] = []
    for _, info in data.items():
        title = info.get("title") or ""
        description = info.get("description") or ""

        if text_source == "title":
            txt = title
        elif text_source == "description":
            txt = description
        elif text_source == "both":
            txt = f"{title} {description}"
        else:
            raise ValueError(f"Unsupported text_source: {text_source!r}")

        texts.append(clean_text(txt))

    mode = "explicit" if target_lemmas else "auto-topk"
    logger.info(
        f"Running spaCy over {len(texts)} videos for lemma counts "
        f"(text_source={text_source}, mode={mode})"
    )

    if target_lemmas is not None:
        target_lemmas = {w.lower() for w in target_lemmas}

    if extra_stopwords is None:
        extra_stopwords_lc: Set[str] = set()
    else:
        extra_stopwords_lc = {w.lower() for w in extra_stopwords}

    counts: Counter = Counter()

    for doc in nlp.pipe(texts, batch_size=256):
        lemma_set: Set[str] = set()

        for token in doc:
            if not token.is_alpha:
                continue
            if token.is_stop:
                continue

            lemma = token.lemma_.lower()
            if lemma in extra_stopwords_lc:
                continue

            lemma_set.add(lemma)

        if not lemma_set:
            continue

        if target_lemmas is not None:
            for lemma in lemma_set.intersection(target_lemmas):
                counts[lemma] += 1
        else:
            for lemma in lemma_set:
                counts[lemma] += 1

    if not counts:
        if target_lemmas is not None:
            logger.warning(
                f"No occurrences found for target lemmas: {sorted(target_lemmas)}"
            )
        else:
            logger.warning("No eligible lemmas found in corpus.")
        return pd.DataFrame(columns=["lemma", "count"])

    if target_lemmas is None:
        counts = Counter(dict(counts.most_common(top_k)))

    df = (
        pd.DataFrame(
            {"lemma": list(counts.keys()), "count": list(counts.values())}
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        "Top spaCy keyword lemmas (first 10 rows):\n"
        f"{df.head(10).to_string(index=False)}"
    )
    logger.info(
        f"Total unique lemmas in table: {len(df)}, "
        f"min count={df['count'].min()}, "
        f"median count={df['count'].median():.1f}, "
        f"max count={df['count'].max()}"
    )

    return df


def plot_spacy_keyword_bar(keyword_df: pd.DataFrame, filename: str = "spacy_keyword_bar") -> None:
    if keyword_df.empty:
        logger.warning("Keyword DataFrame is empty; no spaCy bar plot created.")
        return

    fig = px.bar(
        keyword_df,
        x="lemma",
        y="count",
        labels={"lemma": "Lemma", "count": "Number of videos containing lemma"},
    )

    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(rangemode="tozero")

    save_plotly_figure(fig, filename=filename, width=1600, height=900, scale=SCALE)
    logger.info(
        f"Saved spaCy keyword bar plot with {len(keyword_df)} lemmas as {filename}"
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
        logger.warning(f"Could not parse uploadDate '{upload_date_str}': {exc}")
        return None


def _month_to_season(month: Optional[int]) -> str:
    """Map month to a simple meteorological season."""
    if month is None or pd.isna(month):
        return "unknown"
    m = int(month)
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    if m in (9, 10, 11):
        return "Autumn"
    return "unknown"


def _duration_bucket(minutes: float) -> str:
    """
    Bucket video duration into fixed ranges (in minutes):

    - under_10min  : < 10
    - 10_to_30min  : 10–30
    - 30_to_60min  : 30–60
    - 60_to_180min : 60–180
    - over_180min  : > 180
    - unknown      : missing / non-positive
    """
    if pd.isna(minutes) or minutes <= 0:
        return "unknown"

    m = float(minutes)
    if m < 10:
        return "under_10min"
    if m < 30:
        return "10_to_30min"
    if m < 60:
        return "30_to_60min"
    if m < 180:
        return "60_to_180min"
    return "over_180min"


def get_text_series(df: pd.DataFrame, text_source: str = "both") -> pd.Series:
    """Return text Series according to requested source."""
    text_source = text_source.lower()
    titles = df["title"].fillna("")
    descriptions = df["description"].fillna("")

    if text_source == "title":
        return titles
    if text_source == "description":
        return descriptions
    if text_source == "both":
        return titles + " " + descriptions
    raise ValueError(f"Unsupported text_source: {text_source!r}")


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
    df["title_has_no_talking_tag"] = titles.str.contains(
        r"no[-\s]?talk(?:ing)?", case=False, regex=True
    )
    return df


def add_theme_flags(df: pd.DataFrame, model_name: str = "en_core_web_sm", text_source: str = "both") -> pd.DataFrame:
    """
    Add boolean columns for content themes using spaCy.
    """
    nlp = get_spacy_nlp(model_name)
    theme_cols = [
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

    for col in theme_cols:
        if col not in df.columns:
            df[col] = False

    if nlp is None:
        logger.warning("spaCy not available; theme flags remain False.")
        return df

    texts = get_text_series(df, text_source=text_source).tolist()

    logger.info(
        f"Running spaCy theme detection on {len(df)} videos (text_source={text_source})"
    )

    WHISPER_LEMMAS = {"whisper"}
    SLEEP_LEMMAS = {"sleep", "insomnia"}
    ROLEPLAY_LEMMAS = {"roleplay", "exam", "checkup", "check-up", "haircut", "barber"}
    EAR_LEMMAS = {"ear", "otoscope"}
    MUKBANG_LEMMAS = {"mukbang"}
    KEYBOARD_LEMMAS = {"keyboard", "type"}
    VISUAL_LEMMAS = {"visual", "movement", "trigger"}
    DRIVE_LEMMAS = {"drive"}  # lemma-based driving theme

    for idx, doc in zip(df.index, nlp.pipe(texts, batch_size=256)):
        lower_text = doc.text.lower()

        has_whisper = any(tok.lemma_.lower() in WHISPER_LEMMAS for tok in doc)

        has_no_talking = False
        if (
            "no talking" in lower_text
            or "no-talk" in lower_text
            or "no talk" in lower_text
            or "without talking" in lower_text
        ):
            has_no_talking = True
        else:
            for i, tok in enumerate(doc):
                if tok.lemma_.lower() in {"talk", "speak"} and i > 0:
                    prev = doc[i - 1]
                    if prev.lemma_.lower() in {"no", "without"}:
                        has_no_talking = True
                        break

        has_sleep = any(tok.lemma_.lower() in SLEEP_LEMMAS for tok in doc) or "for sleep" in lower_text

        has_binaural = any(
            kw in lower_text
            for kw in ["binaural", "3dio", "3d audio", "3d sound", "8d audio", "8d sound"]
        )

        has_roleplay = False
        if "roleplay" in lower_text or "rp " in lower_text or " rp" in lower_text:
            has_roleplay = True
        else:
            for tok in doc:
                if tok.lemma_.lower() in ROLEPLAY_LEMMAS:
                    has_roleplay = True
                    break

        has_ear_cleaning = False
        if (
            "ear cleaning" in lower_text
            or "ear massage" in lower_text
            or "ear exam" in lower_text
            or "ear attention" in lower_text
            or "ear brushing" in lower_text
        ):
            has_ear_cleaning = True
        else:
            for i, tok in enumerate(doc):
                if tok.lemma_.lower() in EAR_LEMMAS:
                    window = doc[max(0, i - 3): i + 4]
                    for w in window:
                        if w.lemma_.lower() in {"clean", "brush", "massage", "attention"}:
                            has_ear_cleaning = True
                            break
                if has_ear_cleaning:
                    break

        has_mukbang = (
            "mukbang" in lower_text
            or "eating asmr" in lower_text
            or "eating sounds" in lower_text
            or "eating show" in lower_text
        )
        if not has_mukbang:
            for tok in doc:
                if tok.lemma_.lower() in MUKBANG_LEMMAS:
                    has_mukbang = True
                    break

        has_keyboard = "keyboard" in lower_text
        if not has_keyboard:
            for tok in doc:
                if tok.lemma_.lower() in KEYBOARD_LEMMAS:
                    has_keyboard = True
                    break

        has_visual = any(
            phrase in lower_text
            for phrase in [
                "visual triggers",
                "hand movements",
                "visuals",
                "slow movements",
                "trigger assortment",
            ]
        )
        if not has_visual:
            for tok in doc:
                if tok.lemma_.lower() in VISUAL_LEMMAS:
                    has_visual = True
                    break

        # --- has_drive ---
        has_drive = False

        # lemma-based detection
        if any(tok.lemma_.lower() in DRIVE_LEMMAS for tok in doc):
            has_drive = True

        # optional extra phrase-based heuristics
        if not has_drive and any(
            phrase in lower_text
            for phrase in [
                "driving asmr",
                "drive with me",
                "car asmr",
                "road trip asmr",
            ]
        ):
            has_drive = True

        df.at[idx, "has_whisper"] = has_whisper
        df.at[idx, "has_no_talking"] = has_no_talking
        df.at[idx, "has_sleep"] = has_sleep
        df.at[idx, "has_binaural"] = has_binaural
        df.at[idx, "has_roleplay"] = has_roleplay
        df.at[idx, "has_ear_cleaning"] = has_ear_cleaning
        df.at[idx, "has_mukbang"] = has_mukbang
        df.at[idx, "has_keyboard"] = has_keyboard
        df.at[idx, "has_visual"] = has_visual
        df.at[idx, "has_drive"] = has_drive

    theme_counts = {col: int(df[col].sum()) for col in theme_cols}
    logger.info(f"Theme flag counts (number of videos with flag=True): {theme_counts}")

    return df


def add_growth_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'growth_category' based on fixed views_per_day thresholds
    (no quantiles):

        views_per_day < 100        -> 'slow'
        100 <= views_per_day < 5000 -> 'medium'
        views_per_day >= 5000      -> 'fast'
        missing / <= 0             -> 'unknown'
    """
    vpd = pd.to_numeric(df["views_per_day"], errors="coerce")

    def _cat(x: float) -> str:
        if pd.isna(x) or x <= 0:
            return "unknown"
        if x < 100:
            return "slow"
        if x < 5000:
            return "medium"
        return "fast"

    df["growth_category"] = vpd.apply(_cat)
    counts = df["growth_category"].value_counts()
    logger.info(
        "Growth category distribution based on fixed views_per_day thresholds:\n"
        f"{counts.to_string()}"
    )
    return df


def json_to_dataframe(data: Dict[str, Any], reference_date: Optional[datetime] = None,
                      text_source: str = "both") -> pd.DataFrame:
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
        raw_language = info.get("language")
        language = normalize_language_code(raw_language)
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
    logger.info(f"Constructed initial DataFrame with shape {df.shape}")

    df["upload_datetime"] = pd.to_datetime(
        df["upload_datetime"], errors="coerce", utc=True
    )

    df["views"] = pd.to_numeric(df["views"], errors="coerce")
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce")
    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")
    df["channel_average_views"] = pd.to_numeric(
        df["channel_average_views"], errors="coerce"
    )

    df["duration_minutes"] = df["duration_seconds"] / 60.0
    df["duration_bucket"] = df["duration_minutes"].apply(_duration_bucket)

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
        (df["channel_average_views"] > 0)
        & df["channel_average_views"].notna(),
        df["views"] / df["channel_average_views"],
        np.nan,
    )

    df["upload_year"] = df["upload_datetime"].dt.year  # type: ignore
    df["upload_month"] = df["upload_datetime"].dt.month  # type: ignore
    df["upload_day"] = df["upload_datetime"].dt.day  # type: ignore
    df["upload_date"] = df["upload_datetime"].dt.date  # type: ignore
    df["upload_season"] = df["upload_month"].apply(_month_to_season)

    df = add_title_style_features(df)
    df = add_theme_flags(df, text_source=text_source)
    df = add_growth_category(df)

    logger.info(
        "Finished enriching DataFrame with derived columns; "
        f"final shape is {df.shape}"
    )
    return df


def summarize_by_duration_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Duration bucket table (using fixed duration buckets)."""
    df_copy = df.copy()

    bucket_order = [
        "under_10min",
        "10_to_30min",
        "30_to_60min",
        "60_to_180min",
        "over_180min",
        "unknown",
    ]

    df_copy["duration_bucket"] = pd.Categorical(
        df_copy["duration_bucket"],
        categories=bucket_order,
        ordered=True,
    )

    agg = (
        df_copy.groupby("duration_bucket")
        .agg(
            video_count=("video_id", "count"),
            mean_views=("views", "mean"),
            median_views=("views", "median"),
            mean_views_per_day=("views_per_day", "mean"),
            mean_engagement_rate=("engagement_rate", "mean"),
        )
        .reset_index()
    )
    logger.info(
        "Duration bucket summary table:\n"
        f"{agg.to_string(index=False)}"
    )
    return agg


def summarize_by_language(df: pd.DataFrame) -> pd.DataFrame:
    """Language-level engagement / growth."""
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
    logger.info(
        "Language-level summary (all languages):\n"
        f"{agg.sort_values('video_count', ascending=False).to_string(index=False)}"
    )
    top_langs = (
        agg.sort_values("video_count", ascending=False)
        .head(10)[["language", "video_count"]]
    )
    logger.info(
        "Top languages by video count (first 10):\n"
        f"{top_langs.to_string(index=False)}"
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
    logger.info(
        "Title length bucket summary:\n"
        f"{agg.to_string(index=False)}"
    )
    return agg


def summarize_theme_vs_growth(df: pd.DataFrame, theme_col: str) -> pd.DataFrame:
    """Compare views_per_day 
::contentReference[oaicite:0]{index=0}
for videos with vs without a given theme flag."""
    if theme_col not in df.columns:
        raise ValueError(f"Unknown theme column: {theme_col}")

    agg = (
        df.groupby(theme_col)["views_per_day"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .reset_index()
    )
    logger.info(
        f"views_per_day summary by {theme_col} flag:\n"
        f"{agg.to_string(index=False)}"
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

    if not monthly.empty:
        start = monthly["year_month"].min()
        end = monthly["year_month"].max()
        total = int(monthly["video_count"].sum())
        logger.info(
            "Monthly video counts cover "
            f"{len(monthly)} months from {start.date()} to {end.date()}, "
            f"total videos across all months={total}"
        )

    return monthly


def compute_language_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Growth of ASMR per language (videos per year)."""
    growth = (
        df.groupby(["upload_year", "language"])
        .size()
        .rename("video_count")
        .reset_index()
    )
    if not growth.empty:
        years = sorted(growth["upload_year"].dropna().unique())
        logger.info(
            "Language growth table: "
            f"{len(growth)} rows, {growth['language'].nunique()} languages, "
            f"years {int(years[0])}–{int(years[-1])}"
        )
    return growth


def compute_theme_trend_over_time(df: pd.DataFrame, theme_col: str, by_language: bool = False) -> pd.DataFrame:
    """
    Number of videos with a given theme per year (optionally per language).
    """
    if theme_col not in df.columns:
        raise ValueError(f"Unknown theme column: {theme_col}")

    df_tmp = df.dropna(subset=["upload_year"]).copy()
    df_tmp["upload_year"] = df_tmp["upload_year"].astype(int)

    group_keys = ["upload_year"]
    if by_language:
        group_keys.append("language")

    grouped = df_tmp.groupby(group_keys)[theme_col]

    trend = (
        grouped.agg(
            theme_count=lambda s: int(s.sum()),
            total_videos="count",
        )
        .reset_index()
    )

    if not trend.empty:
        logger.info(
            f"Trend for {theme_col} (by_language={by_language}): "
            f"{len(trend)} rows, total themed videos="
            f"{int(trend['theme_count'].sum())}"
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

    logger.info(
        "Seasonal sleep-share pattern:\n"
        f"{agg.to_string(index=False)}"
    )
    return agg


def compute_lemma_trend_over_time(df: pd.DataFrame, lemma_name: str, lemma_targets: Set[str],
                                  text_source: str = "both", model_name: str = "en_core_web_sm") -> pd.DataFrame:
    """
    Number of videos per year containing any of the given lemmas.
    """
    nlp = get_spacy_nlp(model_name)
    if nlp is None:
        logger.warning("spaCy not available; lemma trend not computed.")
        return pd.DataFrame(columns=["upload_year", "theme_count", "total_videos"])

    df_tmp = df.dropna(subset=["upload_year"]).copy()
    df_tmp["upload_year"] = df_tmp["upload_year"].astype(int)

    texts = get_text_series(df_tmp, text_source=text_source).tolist()
    years = df_tmp["upload_year"].tolist()

    lemma_targets = {le.lower() for le in lemma_targets}

    records: list[tuple[int, bool]] = []

    for year, doc in zip(years, nlp.pipe(texts, batch_size=256)):
        lemma_set = {
            tok.lemma_.lower()
            for tok in doc
            if tok.is_alpha and not tok.is_stop
        }
        has_lemma = bool(lemma_set & lemma_targets)
        records.append((year, has_lemma))

    if not records:
        return pd.DataFrame(columns=["upload_year", "theme_count", "total_videos"])

    tmp = pd.DataFrame(records, columns=["upload_year", "has_lemma"])

    grouped = tmp.groupby("upload_year")["has_lemma"]
    trend = (
        grouped.agg(
            theme_count=lambda s: int(s.sum()),
            total_videos="count",
        )
        .reset_index()
    )

    if not trend.empty:
        logger.info(
            f"Lemma trend for '{lemma_name}': {len(trend)} years, "
            f"total videos with lemma={int(trend['theme_count'].sum())}"
        )
        logger.info(
            "Lemma trend table:\n"
            f"{trend.to_string(index=False)}"
        )

    return trend


def cluster_videos(df: pd.DataFrame, n_clusters: int = 10, random_state: int = 42,
                   text_source: str = "both") -> Tuple[pd.DataFrame, Optional[Pipeline], Optional[pd.DataFrame]]:
    """Cluster videos using title/description text, duration, engagement, and language."""
    df_copy = df.copy()
    df_copy["text_all"] = get_text_series(df_copy, text_source=text_source)

    feature_cols = [
        "text_all",
        "duration_minutes",
        "engagement_rate",
        "views_per_day",
        "language",
    ]

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
    logger.info(
        f"Fitting clustering model on {len(df_copy)} videos (text_source={text_source}, "
        f"n_clusters={n_clusters})"
    )
    pipeline.fit(X)

    logger.info("Assigning cluster labels...")
    df_copy["cluster"] = pipeline.predict(X)

    pca_info: Optional[pd.DataFrame] = None

    try:
        logger.info("Computing 2D PCA embedding for cluster visualization...")
        features = pipeline.named_steps["preprocess"].transform(X)

        reducer = PCA(n_components=2, random_state=random_state)
        embedding_2d = reducer.fit_transform(features)

        df_copy["embedding_x"] = embedding_2d[:, 0]
        df_copy["embedding_y"] = embedding_2d[:, 1]

        pca_info = pd.DataFrame(
            {
                "component": [1, 2],
                "explained_variance_ratio": reducer.explained_variance_ratio_,
            }
        )
        logger.info(
            "PCA explained variance ratios for first 2 components: "
            f"{list(reducer.explained_variance_ratio_)}"
        )
    except Exception as exc:
        logger.warning(f"Could not compute 2D embedding for clusters: {exc}")
        df_copy["embedding_x"] = np.nan
        df_copy["embedding_y"] = np.nan

    if "cluster" in df_copy.columns:
        cluster_counts = df_copy["cluster"].value_counts().sort_index()
        logger.info(
            "Cluster sizes (PCA-based clustering):\n"
            f"{cluster_counts.to_string()}"
        )
        mean_vpd = (
            df_copy.groupby("cluster")["views_per_day"]
            .mean()
            .round(2)
        )
        logger.info(
            "Mean views_per_day by cluster:\n"
            f"{mean_vpd.to_string()}"
        )

    return df_copy, pipeline, pca_info


# ---------------------------------------------------------------------------
# Plotly figure helpers for analytics
# ---------------------------------------------------------------------------

def plot_duration_vs_views(df: pd.DataFrame) -> None:
    """Single plot: log-log duration (seconds) vs views for all videos."""
    df_plot = df[["duration_seconds", "views"]].dropna()
    df_plot = df_plot[
        (df_plot["duration_seconds"] > 0) &
        (df_plot["views"] > 0)
    ]

    if df_plot.empty:
        logger.warning("No data for duration vs views plot.")
        return

    logger.info(
        f"Duration vs views plot uses {len(df_plot)} videos with positive "
        "duration and views."
    )
    logger.info(
        "Duration (seconds) summary for plotted videos:\n"
        f"{df_plot['duration_seconds'].describe().to_string()}"
    )
    logger.info(
        "Views summary for plotted videos:\n"
        f"{df_plot['views'].describe().to_string()}"
    )

    fig = px.scatter(
        df_plot,
        x="duration_seconds",
        y="views",
        title="",
        labels={
            "duration_seconds": "Duration (seconds)",
            "views": "Views",
        },
        opacity=0.6,
    )

    fig.update_traces(marker=dict(color="red"))

    fig.update_xaxes(
        type="log",
        tick0=0,
        dtick=1,
    )
    fig.update_yaxes(
        type="log",
        tick0=0,
        dtick=1,
    )

    save_plotly_figure(
        fig,
        "duration_vs_views",
        width=1600,
        height=900,
        scale=SCALE,
    )


def analyze_log_views_normality(df: pd.DataFrame) -> None:
    """
    Check whether log10(views) is approximately normally distributed.
    """
    views = pd.to_numeric(df["views"], errors="coerce")
    views = views[views > 0].dropna()

    if views.empty:
        logger.warning("No positive views available; cannot analyze normality.")
        return

    log_views = np.log10(views)

    logger.info("===== LOG10(VIEWS) DISTRIBUTION ANALYSIS =====")
    logger.info(f"N = {len(log_views)}")
    logger.info(f"Mean(log10(views)) = {log_views.mean():.3f}")
    logger.info(f"Std(log10(views))  = {log_views.std():.3f}")

    k2, p_normaltest = stats.normaltest(log_views)
    logger.info(
        f"normaltest: statistic = {k2:.3f}, p-value = {p_normaltest:.3g} "
        "(H0: data come from a normal distribution)"
    )

    sample = log_views
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
    )

    try:
        (osm, osr), (slope, intercept, r) = stats.probplot(
            log_views, dist="norm", plot=None
        )

        qq_df = pd.DataFrame(
            {
                "theoretical_quantiles": osm,
                "ordered_log_views": osr,
            }
        )

        fig = px.scatter(
            qq_df,
            x="theoretical_quantiles",
            y="ordered_log_views",
            title="",
            labels={
                "theoretical_quantiles": "Theoretical quantiles (Normal)",
                "ordered_log_views": "Ordered log10(views)",
            },
            opacity=0.7,
        )

        x_min = qq_df["theoretical_quantiles"].min()
        x_max = qq_df["theoretical_quantiles"].max()
        line_x = np.array([x_min, x_max])
        line_y = slope * line_x + intercept  # type: ignore

        fig.add_scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(dash="dash"),
            showlegend=False,
        )

        save_plotly_figure(
            fig,
            filename="log_views_qq_plot",
            width=1600,
            height=900,
            scale=SCALE,
        )

        logger.info(
            "Q–Q plot for log10(views) saved as 'log_views_qq_plot.*'. "
            "Points close to the dashed line indicate approximate normality."
        )
    except Exception as exc:
        logger.warning(f"Could not create Q–Q plot for log10(views): {exc}")

    logger.info("===== END LOG10(VIEWS) DISTRIBUTION ANALYSIS =====")


def plot_language_stats(lang_stats: pd.DataFrame, min_videos: int = 20) -> None:
    """Plot language-level engagement / growth statistics."""
    if lang_stats.empty:
        logger.warning("Language stats DataFrame is empty; skipping plots.")
        return

    df_plot = lang_stats.copy()
    df_plot = df_plot[df_plot["video_count"] >= min_videos]
    if df_plot.empty:
        logger.warning(
            f"No languages with at least {min_videos} videos; skipping language plots."
        )
        return

    logger.info(
        f"Language stats include {len(df_plot)} languages with at least "
        f"{min_videos} videos each."
    )

    df_views = df_plot.sort_values("mean_views_per_day", ascending=False)
    top_vpd = df_views.iloc[0]
    logger.info(
        "Language with highest mean views per day: "
        f"{top_vpd['language']} "
        f"(videos={int(top_vpd['video_count'])}, "
        f"mean_views_per_day={top_vpd['mean_views_per_day']:.2f})"
    )

    df_eng = df_plot.sort_values("mean_engagement_rate", ascending=False)
    top_eng = df_eng.iloc[0]
    logger.info(
        "Language with highest mean engagement rate: "
        f"{top_eng['language']} "
        f"(videos={int(top_eng['video_count'])}, "
        f"mean_engagement_rate={top_eng['mean_engagement_rate']:.4f})"
    )

    fig = px.bar(
        df_views,
        x="language",
        y="mean_views_per_day",
        title="",
        labels={"language": "Language", "mean_views_per_day": "Mean views per day"},
    )
    fig.update_xaxes(tickangle=45)
    save_plotly_figure(
        fig,
        "language_mean_views_per_day",
        width=1600,
        height=900,
        scale=SCALE,
    )

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
    fig.update_xaxes(tickangle=45)
    save_plotly_figure(
        fig,
        "language_mean_engagement_rate",
        width=1600,
        height=900,
        scale=SCALE,
    )


def plot_title_style_stats(title_stats: pd.DataFrame) -> None:
    """Plot engagement vs title length buckets."""
    if title_stats.empty:
        logger.warning("Title stats DataFrame is empty; skipping plots.")
        return

    df_plot = title_stats.copy()
    order = ["<=5 words", "6–10 words", "11–20 words", ">20 words"]
    df_plot["title_length_bucket"] = pd.Categorical(
        df_plot["title_length_bucket"], categories=order, ordered=True
    )
    df_plot = df_plot.sort_values("title_length_bucket")

    logger.info(
        "Title length statistics (mean engagement rate and views):\n"
        f"{df_plot.to_string(index=False)}"
    )

    fig = px.bar(
        df_plot,
        x="title_length_bucket",
        y="mean_engagement_rate",
        title="",
        labels={
            "title_length_bucket": "Title length",
            "mean_engagement_rate": "Mean engagement rate",
        },
    )
    save_plotly_figure(
        fig,
        "title_length_mean_engagement_rate",
        width=1600,
        height=900,
        scale=SCALE,
    )

    fig = px.bar(
        df_plot,
        x="title_length_bucket",
        y="mean_views",
        title="",
        labels={
            "title_length_bucket": "Title length",
            "mean_views": "Mean views",
        },
    )
    save_plotly_figure(
        fig,
        "title_length_mean_views",
        width=1600,
        height=900,
        scale=SCALE,
    )


def plot_theme_growth_box(df: pd.DataFrame, theme_col: str) -> None:
    """Boxplot of views_per_day for videos with/without a given theme."""
    if theme_col not in df.columns:
        logger.warning(f"Theme column {theme_col} not present; skipping boxplot.")
        return

    df_plot = df[["views_per_day", theme_col]].dropna(subset=["views_per_day"])
    if df_plot.empty:
        logger.warning(
            f"No non-missing views_per_day data for theme {theme_col}; skipping boxplot."
        )
        return

    logger.info(
        f"Theme growth boxplot for {theme_col} with {len(df_plot)} videos "
        "having non-missing views_per_day."
    )
    counts = df_plot[theme_col].value_counts()
    logger.info(
        f"Counts by {theme_col} flag (False/True):\n{counts.to_string()}"
    )

    fig = px.box(
        df_plot,
        x=theme_col,
        y="views_per_day",
        title="",
        labels={
            theme_col: f"{theme_col} (False / True)",
            "views_per_day": "Views per day",
        },
    )
    filename = f"{theme_col}_views_per_day_boxplot"
    save_plotly_figure(fig, filename, width=1600, height=900, scale=SCALE)


def plot_language_growth(lang_growth: pd.DataFrame, min_total_videos: int = 50) -> None:
    """Line chart for growth of ASMR per language (videos per year)."""
    if lang_growth.empty:
        logger.warning("Language growth DataFrame is empty; skipping plot.")
        return

    df_plot = lang_growth.copy()
    totals = df_plot.groupby("language")["video_count"].sum().reset_index()
    keep_langs = totals[totals["video_count"] >= min_total_videos]["language"]
    df_plot = df_plot[df_plot["language"].isin(keep_langs)]
    if df_plot.empty:
        logger.warning(
            f"No languages with at least {min_total_videos} total videos; "
            "skipping language growth plot."
        )
        return

    df_plot = df_plot.sort_values(["language", "upload_year"])

    logger.info(
        "Language growth includes "
        f"{len(df_plot)} rows for {df_plot['language'].nunique()} languages "
        f"over years {int(df_plot['upload_year'].min())}"
        f"–{int(df_plot['upload_year'].max())}."
    )

    fig = px.line(
        df_plot,
        x="upload_year",
        y="video_count",
        color="language",
        markers=True,
        title="",
        labels={
            "upload_year": "Year",
            "video_count": "Number of videos",
            "language": "Language",
        },
    )
    save_plotly_figure(
        fig,
        "language_growth_over_years",
        width=1600,
        height=900,
        scale=SCALE,
    )


def plot_theme_trend_overall(trend_df: pd.DataFrame, theme_col: str) -> None:
    """Trend of number of themed videos over years (all languages combined)."""
    if trend_df.empty:
        logger.warning(
            f"Trend DataFrame for {theme_col} is empty; skipping overall trend plot."
        )
        return

    df_plot = trend_df.copy()
    if "theme_count" not in df_plot.columns:
        logger.warning(
            f"'theme_count' column not found in trend_df for {theme_col}; "
            "skipping overall theme trend plot."
        )
        return

    logger.info(
        f"Overall theme trend for {theme_col}: "
        f"{len(df_plot)} years, total themed videos="
        f"{int(df_plot['theme_count'].sum())}."
    )

    fig = px.line(
        df_plot,
        x="upload_year",
        y="theme_count",
        title="",
        labels={
            "upload_year": "Year",
            "theme_count": f"Number of videos with {theme_col}",
        },
        markers=True,
    )
    filename = f"{theme_col}_trend_overall_fig"
    save_plotly_figure(fig, filename, width=1600, height=900, scale=SCALE)


def plot_theme_trend_by_language(trend_df: pd.DataFrame, theme_col: str, min_videos: int = 30) -> None:
    """Trend of number of themed videos over years by language."""
    if trend_df.empty:
        logger.warning(
            f"Trend DataFrame for {theme_col} by language is empty; skipping plot."
        )
        return

    df_plot = trend_df.copy()
    if "theme_count" not in df_plot.columns:
        logger.warning(
            f"'theme_count' column not found in trend_df for {theme_col}; "
            "skipping theme trend by language plot."
        )
        return

    counts = (
        df_plot.groupby("language")["theme_count"]
        .count()
        .reset_index(name="n")
    )
    keep_langs = counts[counts["n"] >= min_videos]["language"]
    df_plot = df_plot[df_plot["language"].isin(keep_langs)]
    if df_plot.empty:
        logger.warning(
            f"No languages with at least {min_videos} year-groups for theme {theme_col}; "
            "skipping by-language trend plot."
        )
        return

    logger.info(
        f"Theme trend by language for {theme_col}: "
        f"{len(df_plot)} rows across {df_plot['language'].nunique()} languages."
    )

    fig = px.line(
        df_plot,
        x="upload_year",
        y="theme_count",
        color="language",
        markers=True,
        title="",
        labels={
            "upload_year": "Year",
            "theme_count": f"Number of videos with {theme_col}",
            "language": "Language",
        },
    )
    filename = f"{theme_col}_trend_by_language_fig"
    save_plotly_figure(fig, filename, width=1600, height=900, scale=SCALE)


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
    save_plotly_figure(
        fig,
        "monthly_video_counts",
        width=1600,
        height=900,
        scale=SCALE,
    )


def plot_cluster_distribution(clustered_df: pd.DataFrame, name_suffix: str = "") -> None:
    """Visualize clusters: bar charts + 2D scatter with circles around clusters."""
    if "cluster" not in clustered_df.columns:
        logger.warning("No 'cluster' column in clustered_df; skipping cluster plots.")
        return

    df_plot = clustered_df.copy()
    suffix = f"_{name_suffix}" if name_suffix else ""

    agg = (
        df_plot.groupby("cluster")
        .agg(
            video_count=("video_id", "count"),
            mean_views=("views", "mean"),
            mean_views_per_day=("views_per_day", "mean"),
        )
        .reset_index()
    )

    logger.info(
        f"Cluster distribution summary (suffix='{name_suffix}'):\n"
        f"{agg.to_string(index=False)}"
    )

    fig = px.bar(
        agg,
        x="cluster",
        y="video_count",
        title="",
        labels={"cluster": "Cluster", "video_count": "Number of videos"},
    )
    save_plotly_figure(
        fig,
        f"cluster_sizes{suffix}",
        width=1600,
        height=900,
        scale=SCALE,
    )

    fig = px.bar(
        agg,
        x="cluster",
        y="mean_views_per_day",
        title="",
        labels={
            "cluster": "Cluster",
            "mean_views_per_day": "Mean views per day",
        },
    )
    save_plotly_figure(
        fig,
        f"cluster_mean_views_per_day{suffix}",
        width=1600,
        height=900,
        scale=SCALE,
    )

    if "embedding_x" not in df_plot.columns or "embedding_y" not in df_plot.columns:
        logger.warning(
            "No embedding_x / embedding_y columns found; skipping cluster scatter plot."
        )
        return

    df_emb = df_plot.dropna(subset=["embedding_x", "embedding_y"]).copy()
    if df_emb.empty:
        logger.warning(
            "Embedding columns are empty; skipping cluster scatter plot."
        )
        return

    fig = px.scatter(
        df_emb,
        x="embedding_x",
        y="embedding_y",
        color="cluster",
        hover_data=["video_id", "title", "language", "views", "duration_minutes"],
        title="",
        labels={
            "embedding_x": "",
            "embedding_y": "",
            "cluster": "",
        },
    )

    shapes = []
    for cluster_id, group in df_emb.groupby("cluster"):
        if len(group) < 2:
            continue

        cx = group["embedding_x"].mean()
        cy = group["embedding_y"].mean()
        distances = np.sqrt(
            (group["embedding_x"] - cx) ** 2 + (group["embedding_y"] - cy) ** 2
        )

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

    save_plotly_figure(
        fig,
        f"cluster_scatter_embedding{suffix}",
        width=1600,
        height=900,
        scale=SCALE,
    )


def cluster_videos_tsne(df: pd.DataFrame, n_clusters: int = 10, random_state: int = 42, text_source: str = "both",
                        tsne_perplexity: float = 30.0, tsne_learning_rate: float = 200.0,
                        tsne_n_iter: int = 1000) -> Tuple[pd.DataFrame, Optional[Pipeline]]:
    """
    Same idea as `cluster_videos`, but the 2D embedding used for visualization
    is computed with t-SNE (after an SVD pre-step for speed).
    """
    df_copy = df.copy()
    df_copy["text_all"] = get_text_series(df_copy, text_source=text_source)

    feature_cols = [
        "text_all",
        "duration_minutes",
        "engagement_rate",
        "views_per_day",
        "language",
    ]

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
    logger.info(
        f"Fitting clustering model with t-SNE embedding on {len(df_copy)} videos "
        f"(text_source={text_source}, n_clusters={n_clusters})"
    )
    pipeline.fit(X)

    logger.info("Assigning cluster labels (t-SNE version)...")
    df_copy["cluster"] = pipeline.predict(X)

    try:
        logger.info("Computing 2D t-SNE embedding for cluster visualization...")
        features = pipeline.named_steps["preprocess"].transform(X)

        n_features = features.shape[1]
        n_samples = features.shape[0]
        max_components = min(n_features, n_samples - 1)

        if max_components >= 2:
            svd = TruncatedSVD(
                n_components=max_components,
                random_state=random_state,
            )
            features_reduced = svd.fit_transform(features)
            logger.info(
                f"Reduced feature space to {max_components} dimensions via TruncatedSVD "
                f"before t-SNE."
            )
        else:
            features_reduced = (
                features.toarray() if hasattr(features, "toarray") else np.array(features)
            )
            logger.info(
                "Skipped SVD reduction before t-SNE because dataset is very small."
            )

        tsne = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            learning_rate=tsne_learning_rate,
            random_state=random_state,
            init="random",
        )
        embedding_2d = tsne.fit_transform(features_reduced)

        df_copy["embedding_x"] = embedding_2d[:, 0]
        df_copy["embedding_y"] = embedding_2d[:, 1]

        logger.info(
            "Completed t-SNE embedding with "
            f"{embedding_2d.shape[0]} points."
        )

        if "cluster" in df_copy.columns:
            cluster_counts = df_copy["cluster"].value_counts().sort_index()
            logger.info(
                "Cluster sizes (t-SNE embedding version):\n"
                f"{cluster_counts.to_string()}"
            )

    except Exception as exc:
        logger.warning(f"Could not compute 2D t-SNE embedding for clusters: {exc}")
        df_copy["embedding_x"] = np.nan
        df_copy["embedding_y"] = np.nan

    return df_copy, pipeline


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

    logger.info("===== END DATASET SUMMARY =====")


# ---------------------------------------------------------------------------
# Analytics pipeline: CSVs + figures
# ---------------------------------------------------------------------------

def run_analytics_pipeline(data: Dict[str, Any], text_source: str = "both") -> None:
    """Run all analytics, write CSVs into output/analysis, and create Plotly figures."""
    analysis_dir = os.path.join(common.output_dir, "analysis")

    enriched_pickle = os.path.join(
        analysis_dir,
        f"asmr_videos_enriched_{text_source}.pkl",
    )

    if os.path.isfile(enriched_pickle):
        logger.info(f"Loading enriched dataset from pickle {enriched_pickle}")
        df = pd.read_pickle(enriched_pickle)
    else:
        logger.info("No enriched pickle found; building DataFrame from JSON...")
        df = json_to_dataframe(data, text_source=text_source)
        logger.info(f"Saving enriched dataset to pickle {enriched_pickle}")
        df.to_pickle(enriched_pickle)

    logger.info(
        f"Enriched DataFrame ready with {len(df)} rows and {len(df.columns)} columns."
    )

    print_dataset_summary(df)

    enriched_csv = os.path.join(analysis_dir, "asmr_videos_enriched.csv")
    logger.info(f"Saving enriched dataset CSV to {enriched_csv}")
    df.to_csv(enriched_csv, index=False)

    duration_stats = summarize_by_duration_bucket(df)
    duration_stats.to_csv(
        os.path.join(analysis_dir, "duration_stats.csv"),
        index=False,
    )

    plot_duration_vs_views(df)

    analyze_log_views_normality(df)

    lang_stats = summarize_by_language(df)
    lang_stats.to_csv(
        os.path.join(analysis_dir, "language_stats.csv"),
        index=False,
    )
    plot_language_stats(lang_stats)

    title_stats = summarize_title_styles(df)
    title_stats.to_csv(
        os.path.join(analysis_dir, "title_style_stats.csv"),
        index=False,
    )
    plot_title_style_stats(title_stats)

    for theme in ["has_whisper", "has_no_talking", "has_sleep", "has_binaural", "has_drive"]:
        if theme in df.columns:
            theme_stats = summarize_theme_vs_growth(df, theme)
            theme_stats.to_csv(
                os.path.join(analysis_dir, f"{theme}_growth_stats.csv"),
                index=False,
            )
            plot_theme_growth_box(df, theme)

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

    for theme in ["has_no_talking", "has_binaural"]:
        if theme in df.columns:
            trend_all = compute_theme_trend_over_time(
                df, theme_col=theme, by_language=False
            )
            trend_lang = compute_theme_trend_over_time(
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

            plot_theme_trend_overall(trend_all, theme)
            plot_theme_trend_by_language(trend_lang, theme)

    drive_trend = compute_lemma_trend_over_time(
        df,
        lemma_name="drive",
        lemma_targets={"drive"},
        text_source=text_source,
        model_name="en_core_web_sm",
    )
    if not drive_trend.empty:
        drive_trend.to_csv(
            os.path.join(analysis_dir, "drive_trend_overall.csv"),
            index=False,
        )
        plot_theme_trend_overall(drive_trend, theme_col="drive")

    clustered_df, pipeline, pca_info = cluster_videos(
        df, n_clusters=12, text_source=text_source
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
                median_views=("views", "median"),
                mean_views_per_day=("views_per_day", "mean"),
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

        plot_cluster_distribution(clustered_df, name_suffix="pca")

        clustered_tsne_df, tsne_pipeline = cluster_videos_tsne(
            df, n_clusters=12, text_source=TEXT_SOURCE
        )

        tsne_embedding_csv = os.path.join(analysis_dir, "cluster_embedding_2d_tsne.csv")
        if {"embedding_x", "embedding_y"}.issubset(clustered_tsne_df.columns):
            clustered_tsne_df[emb_cols].to_csv(tsne_embedding_csv, index=False)
            logger.info(f"2D t-SNE embedding for clusters saved to {tsne_embedding_csv}")

        plot_cluster_distribution(clustered_tsne_df, name_suffix="tsne")

    logger.info(
        f"Analytics pipeline complete. CSVs and figures written to {analysis_dir}"
    )


# ============================================================================
# MAIN — run wordclouds + analytics + spaCy keyword bar plot
# ============================================================================

def main() -> None:
    json_path = os.path.join(common.get_configs("data"), "asmr_results.json")
    logger.info(f"Loading ASMR data from {json_path}")
    data = load_asmr_data(json_path)

    analysis_dir = os.path.join(common.output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    run_wordcloud_pipeline(data, text_source=TEXT_SOURCE)

    keyword_pickle = os.path.join(
        analysis_dir,
        f"spacy_keywords_{TEXT_SOURCE}.pkl",
    )

    if os.path.isfile(keyword_pickle):
        logger.info(f"Loading spaCy keyword counts from pickle {keyword_pickle}")
        keyword_df = pd.read_pickle(keyword_pickle)
    else:
        logger.info("No spaCy keyword pickle found; computing keyword counts...")
        keyword_df = compute_spacy_keyword_counts(
            data,
            target_lemmas=None,
            text_source=TEXT_SOURCE,
            model_name="en_core_web_sm",
            top_k=30,
            extra_stopwords=get_custom_stopwords(),
        )
        logger.info(f"Saving spaCy keyword counts to pickle {keyword_pickle}")
        keyword_df.to_pickle(keyword_pickle)

    plot_spacy_keyword_bar(
        keyword_df,
        filename=f"spacy_keywords_{TEXT_SOURCE}",
    )

    run_analytics_pipeline(data, text_source=TEXT_SOURCE)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()