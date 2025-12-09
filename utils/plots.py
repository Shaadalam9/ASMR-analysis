import logging
from wordcloud import WordCloud
from typing import Set


logger = logging.getLogger(__name__)


class Plots():
    def __init__(self) -> None:
        pass

    def generate_wordcloud_image(self, text: str, stopwords: Set[str]):
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
