import logging
from wordcloud import WordCloud
from typing import Set, Dict


logger = logging.getLogger(__name__)


class Plots():
    def __init__(self) -> None:
        pass

    def generate_wordcloud_image(self, text: str, stopwords: Set[str]):
        """Generate a word cloud image array from raw text."""
        wordcloud = WordCloud(
            width=1000,
            height=600,
            background_color="white",
            stopwords=stopwords,
            collocations=False,
        ).generate(text)
        img = wordcloud.to_array()
        logger.info(
            f"Generated word cloud with {len(wordcloud.words_)} unique words."
        )
        return img

    def generate_wordcloud_from_frequencies(
        self,
        frequencies: Dict[str, int],
        stopwords: Set[str],
    ):
        """
        Generate a word cloud image array from a lemma frequency dict.

        Keys are lemmas (already normalised), values are video counts.
        """
        wordcloud = WordCloud(
            width=1000,
            height=600,
            background_color="white",
            stopwords=stopwords,
            collocations=False,
        ).generate_from_frequencies(frequencies)
        img = wordcloud.to_array()
        logger.info(
            f"Generated word cloud from frequencies with "
            f"{len(wordcloud.words_)} unique lemmas."
        )
        return img
