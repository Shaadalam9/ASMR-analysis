import pandas as pd
from wordcloud import STOPWORDS
from typing import Optional, Set, Dict
import logging
import re


logger = logging.getLogger(__name__)


class Tools():

    def __init__(self) -> None:
        pass

    def _month_to_season(self, month: Optional[int]) -> str:
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

    def get_custom_stopwords(self) -> Set[str]:
        """Create a combined stopword set for word cloud and spaCy lemma filtering."""
        stopwords = set(STOPWORDS)

        custom_stopwords = {
            # Domain-specific.
            "asmr", "ASMR", "gmail", "comment", "twitter", "facebook", "patreon", "help",
            "new", "youtube", "enjoy", "let", "spotify", "email", "nancy",

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
            "dm", "instagram", "tiktok", "channel", "paypal", "support",

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

    def clean_text(self, text: str) -> str:
        """Simple cleaning for wordcloud / spaCy text."""
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"[\r\n]+", " ", text)
        return text

    def normalize_lemma_form(self, lemma: str) -> str:
        """
        Post-process spaCy lemmas to merge obvious morphological variants into a
        single canonical form for keyword counting and plotting.

        Extend this mapping as needed.
        """
        lemma = lemma.lower()

        LEMMA_CANON = {
            # whisper family
            "whispering": "whisper",
            "whispers": "whisper",
            "whispered": "whisper",

            # relax family
            "relaxation": "relax",
            "relaxing": "relax",
            "relaxed": "relax",
            "relaxation": "relax"
        }

        return LEMMA_CANON.get(lemma, lemma)

    def _duration_bucket(self, minutes: float) -> str:
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

    def get_language_name(self, code: str) -> str:
        """
        Return the human-readable language name for a given language code.

        Normalizes the input by:
        - Stripping leading/trailing whitespace
        - Lowercasing
        - Replacing underscores with hyphens

        Falls back to "Unknown" if the code is not in the map.
        """

        language_map: Dict[str, str] = {
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
            "ca": "Catalan",
            "et": "Estonian",

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
            "af": "Afrikaans",
            "sw": "Swahili",

            # Tagalog / Filipino
            "tl": "Filipino",

            # Fallback
            "unknown": "Unknown",
        }

        if not code:
            return language_map["unknown"]

        normalized = code.strip().lower().replace("_", "-")
        return language_map.get(normalized, language_map["unknown"])
