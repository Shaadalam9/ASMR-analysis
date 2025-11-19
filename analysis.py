"""ASMR word cloud analysis.

This script loads ASMR-related YouTube metadata from a JSON file,
builds separate text corpora from video titles, descriptions, and
the combination of both, removes common stopwords, generates three
word clouds (one per corpus), visualizes them with Plotly, and saves
the figures in multiple formats.
"""

import json
import logging
import os
import re
import shutil
import warnings
from typing import Any, Dict, Set

import common
import plotly as py
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS

# Default scaling factor for saved PNG images.
SCALE = 3

logger = logging.getLogger(__name__)


def load_asmr_data(json_path: str) -> Dict[str, Any]:
    """Load ASMR results from a JSON file.

    Args:
        json_path: Absolute or relative path to the JSON file
            containing ASMR video metadata.

    Returns:
        A dictionary mapping video IDs to their metadata.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_corpus(data: Dict[str, Any], source: str) -> str:
    """Build a text corpus from titles, descriptions, or both.

    Args:
        data: Dictionary with video metadata where each value
            is expected to contain "title" and/or "description"
            fields.
        source: Which fields to use when building the corpus.
            Supported values:
            - "title": use only titles
            - "description": use only descriptions
            - "both": concatenate title and description

    Returns:
        A single string containing the concatenated corpus,
        separated by spaces.

    Raises:
        ValueError: If an unsupported source value is provided.
    """
    texts = []

    for _, info in data.items():
        title = info.get("title", "")
        description = info.get("description", "")

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
    """Clean raw text by removing URLs and normalizing whitespace.

    Args:
        text: Input text string to be cleaned.

    Returns:
        Cleaned text with URLs removed and newlines replaced by spaces.
    """
    # Remove URLs.
    text = re.sub(r"http\\S+", " ", text)

    # Replace line breaks with spaces.
    text = re.sub(r"[\\r\\n]+", " ", text)

    return text


def get_custom_stopwords() -> Set[str]:
    """Create a combined stopword set for word cloud generation.

    This includes the default WordCloud stopwords, common English
    function words, social media filler terms, and some French
    function words observed in the dataset.

    Returns:
        A set of stopwords to exclude from the word clouds.
    """
    # Start with the default WordCloud stopwords.
    stopwords = set(STOPWORDS)

    # Extend with custom stopwords.
    custom_stopwords = {
        # Domain-specific.
        "asmr", "ASMR",

        # Basic English stopwords.
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as",
        "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could",
        "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few",
        "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd",
        "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
        "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
        "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
        "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their",
        "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
        "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
        "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
        "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",

        # Social media fillers.
        "thanks", "thank", "thankyou", "thanksgiving", "subscribe", "sub", "follow", "like", "likes", "watch",
        "watching", "video", "videos", "link", "please", "dm", "instagram", "tiktok", "channel",

        # French fillers (observed in the dataset).
        "le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "dans", "ce",
        "ces", "je", "tu", "que", "qui", "au", "aux", "pour", "mais",
    }
    stopwords.update(custom_stopwords)

    # --- Remove single-letter words: a–z, A–Z ---
    single_letters = {chr(i) for i in range(ord("a"), ord("z") + 1)}
    single_letters |= {chr(i) for i in range(ord("A"), ord("Z") + 1)}
    stopwords.update(single_letters)

    # --- Remove punctuation tokens ---
    punctuation_tokens = {
        ".", ",", "!", "?", ":", ";",
        "-", "_", "(", ")", "[", "]", "{", "}",
        "'", '"', "/", "\\", "|", "&", "*", "#", "@",
        "...", ".."
    }
    stopwords.update(punctuation_tokens)

    # --- Optional: remove digits-only tokens ---
    digits = {str(i) for i in range(10)}
    stopwords.update(digits)

    return stopwords


def generate_wordcloud_image(text: str, stopwords: Set[str]):
    """Generate a word cloud image array from text.

    Args:
        text: Cleaned text corpus used to generate the word cloud.
        stopwords: Set of stopwords to exclude from the word cloud.

    Returns:
        A NumPy array representing the word cloud image.
    """
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
    """Create a Plotly figure to display the word cloud image.

    Args:
        img: Word cloud image as a NumPy array.
        title: Title to display above the figure.

    Returns:
        A Plotly figure object with the word cloud rendered.
    """
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
    """Save a Plotly figure as HTML, PNG, and EPS formats.

    This function saves the figure into two locations:
    - ``common.output_dir`` (working directory)
    - a ``figures`` directory inside ``common.root_dir`` (for final figures)

    The EPS export relies on Kaleido v0.x (e.g., 0.2.1). Deprecation
    warnings emitted by Plotly for Kaleido versions less than 1.0.0
    are suppressed locally around the image export calls.

    Args:
        fig: Plotly figure object to be saved.
        filename: Base file name (without extension) to use for saving.
        width: Width of PNG and EPS images in pixels. Defaults to 1600.
        height: Height of PNG and EPS images in pixels. Defaults to 900.
        scale: Scaling factor for the PNG image. Defaults to ``SCALE``.
        save_final: Whether to also save the figure into the ``figures``
            directory (considered the final or best version).
        save_png: Whether to save the figure as a PNG file.
        save_eps: Whether to save the figure as an EPS file.

    Raises:
        ValueError: If Plotly encounters an error while writing image files.
    """
    # Ensure output directories exist.
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
        # Save as PNG, suppressing Kaleido v0 deprecation warnings.
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

        # Save as EPS, suppressing Kaleido v0 deprecation warnings.
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


def main() -> None:
    """Run the full ASMR word cloud generation and saving pipeline.

    The pipeline:
    1. Construct the JSON path using project configuration.
    2. Load ASMR video metadata from JSON.
    3. Build and clean three corpora:
       - titles only
       - descriptions only
       - titles + descriptions
    4. Generate three word clouds.
    5. Display the results using Plotly.
    6. Save the figures in multiple formats.
    """
    # Build path to the JSON file from project configuration.
    json_path = os.path.join(common.get_configs("data"), "asmr_results.json")

    # Load ASMR video metadata.
    data = load_asmr_data(json_path)

    # Build stopword set once.
    stopwords = get_custom_stopwords()

    # Define the three corpus configurations.
    configs = [
        {
            "source": "title",
            "filename": "wordcloud_titles",
        },
        {
            "source": "description",
            "filename": "wordcloud_descriptions",
        },
        {
            "source": "both",
            "filename": "wordcloud_titles_descriptions",
        },
    ]

    for cfg in configs:
        logger.info("Generating word cloud for source='%s'.", cfg["source"])

        # Build and clean the corresponding corpus.
        raw_text = build_corpus(data, source=cfg["source"])
        cleaned_text = clean_text(raw_text)

        # Generate the word cloud image.
        img = generate_wordcloud_image(cleaned_text, stopwords)

        # Create the Plotly figure.
        fig = create_plotly_figure(img)

        # Save the figure in multiple formats.
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


if __name__ == "__main__":
    main()
