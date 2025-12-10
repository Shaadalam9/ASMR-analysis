import json
import pandas as pd
import numpy as np
from custom_logger import CustomLogger
from typing import Any, Dict, Optional
from datetime import datetime, timezone
import spacy

from utils.tool import Tools

logger = CustomLogger(__name__)

_NLP_CACHE: Optional["spacy.language.Language"] = None  # type: ignore[valid-type]

tool_class = Tools()


class Preprocessing():
    def __init__(self) -> None:
        pass

    def load_asmr_data(self, json_path: str) -> Dict[str, Any]:
        """Load ASMR results from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded JSON with {len(data)} video entries from {json_path}")
        return data

    def _parse_upload_datetime(self, upload_date_str: Optional[str]) -> Optional[datetime]:
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

    def normalize_language_code(self, lang: Any) -> str:
        """Map short codes (en, jp, tl, ...) to human-readable language names."""
        if not isinstance(lang, str):
            code = "unknown"
        else:
            code = lang.strip().lower() or "unknown"

        label = tool_class.get_language_name(code)
        if label is not None:
            return label

        # Fallback: title-case whatever is left
        return code.title()

    def json_to_dataframe(self, data: Dict[str, Any], reference_date: Optional[datetime] = None,
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
            language = self.normalize_language_code(raw_language)
            upload_date_str = info.get("uploadDate")
            channel_avg_views = info.get("channel_average_views")

            upload_dt = self._parse_upload_datetime(upload_date_str)
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
        df["duration_bucket"] = df["duration_minutes"].apply(tool_class._duration_bucket)

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
        df["likes_per_day"] = np.where(
            df["days_since_upload"] > 0,
            df["likes"] / df["days_since_upload"],
            np.nan,
        )
        df["rel_views_vs_channel_avg"] = np.where(
            (df["channel_average_views"] > 0)
            & df["channel_average_views"].notna(),
            df["views"] / df["channel_average_views"],
            np.nan,
        )

        # ---- NEW: log10-transformed metrics for use in tables ----
        df["log10_views"] = np.where(df["views"] > 0, np.log10(df["views"]), np.nan)
        df["log10_views_per_day"] = np.where(
            df["views_per_day"] > 0, np.log10(df["views_per_day"]), np.nan
        )
        df["log10_likes"] = np.where(df["likes"] > 0, np.log10(df["likes"]), np.nan)
        df["log10_likes_per_day"] = np.where(
            df["likes_per_day"] > 0, np.log10(df["likes_per_day"]), np.nan
        )
        df["log10_engagement_rate"] = np.where(
            df["engagement_rate"] > 0, np.log10(df["engagement_rate"]), np.nan
        )

        df["upload_year"] = df["upload_datetime"].dt.year  # type: ignore
        df["upload_month"] = df["upload_datetime"].dt.month  # type: ignore
        df["upload_day"] = df["upload_datetime"].dt.day  # type: ignore
        df["upload_date"] = df["upload_datetime"].dt.date  # type: ignore
        df["upload_season"] = df["upload_month"].apply(tool_class._month_to_season)

        df = self.add_title_style_features(df)
        df = self.add_theme_flags(df, text_source=text_source)
        df = self.add_growth_category(df)

        logger.info(
            "Finished enriching DataFrame with derived columns; "
            f"final shape is {df.shape}"
        )
        return df

    def get_text_series(self, df: pd.DataFrame, text_source: str = "both") -> pd.Series:
        """Return text Series according to requested source."""
        text_source = text_source.lower()
        titles = df["title"].fillna("")
        descriptions = df["description"].fillna("")

        if text_source == "title":
            return titles
        if text_source == "description":
            return descriptions
        if text_source == "both":
            return titles + " " + descriptions  # type: ignore
        raise ValueError(f"Unsupported text_source: {text_source!r}")

    def add_title_style_features(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def add_theme_flags(self, df: pd.DataFrame, model_name: str = "en_core_web_sm",
                        text_source: str = "both") -> pd.DataFrame:
        """
        Add boolean columns for content themes using spaCy.
        """
        nlp = self.get_spacy_nlp(model_name)
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

        texts = self.get_text_series(df, text_source=text_source).tolist()

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
                    "driving",
                    "drive with me",
                    "car",
                    "road trip",
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

    def add_growth_category(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def get_spacy_nlp(self, model_name: str = "en_core_web_sm"):
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

    def build_corpus(self, data: Dict[str, Any], source: str) -> str:
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
