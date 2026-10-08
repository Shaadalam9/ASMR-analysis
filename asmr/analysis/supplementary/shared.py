from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go

import common
from asmr.processing.preprocessing import Preprocessing
from custom_logger import CustomLogger

logger = CustomLogger(__name__)

SEED = int(common.get_configs("random_seed"))
TEXT_SOURCE = common.get_configs("analysis_text_source")
ANALYSIS_DIR = os.path.join(common.output_dir, "analysis")
OUT = os.path.join(ANALYSIS_DIR, "extra")
FIG = os.path.join(common.root_dir, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

THEMES = [
    "has_whisper", "has_no_talking", "has_sleep", "has_binaural", "has_roleplay",
    "has_ear_cleaning", "has_mukbang", "has_keyboard", "has_visual", "has_drive",
]
THEME_NAMES = {
    "has_whisper": "Whisper", "has_no_talking": "No talking", "has_sleep": "Sleep related",
    "has_binaural": "Binaural / 3D audio", "has_roleplay": "Role play",
    "has_ear_cleaning": "Ear cleaning / ear focus", "has_mukbang": "Mukbang / eating",
    "has_keyboard": "Keyboard / typing", "has_visual": "Visual triggers", "has_drive": "Driving",
}
DURATION_ORDER = ["under_10min", "10_to_30min", "30_to_60min", "60_to_180min", "over_180min"]

pre = Preprocessing()


def load_df() -> pd.DataFrame:
    path = os.path.join(ANALYSIS_DIR, f"asmr_videos_enriched_{TEXT_SOURCE}.pkl")
    df = pd.read_pickle(path)
    for col in THEMES:
        df[col] = df[col].astype(bool)
    logger.info(f"Loaded {len(df)} videos from {path}")
    return df


def save_table(df: pd.DataFrame, name: str) -> None:
    """Write a CSV table to the supplementary output folder."""
    df.to_csv(os.path.join(OUT, name), index=False)
    logger.info(f"Wrote {name} ({len(df)} rows)")


def save_figure(fig: go.Figure, name: str, width: int = 1200, height: int = 800, scale: int = 3) -> None:
    """Save a Plotly figure as interactive HTML (Plotly.js from the CDN), PNG and EPS in ``figures/``."""
    fig.update_layout(
        template=common.get_configs("plotly_template"), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family=common.get_configs("font_family"), size=14), width=width, height=height,
    )
    fig.write_html(os.path.join(FIG, f"{name}.html"), include_plotlyjs="cdn", full_html=True)
    try:
        fig.write_image(os.path.join(FIG, f"{name}.png"), width=width, height=height, scale=scale)
        fig.write_image(os.path.join(FIG, f"{name}.eps"), width=width, height=height)
    except Exception as exc:  # noqa: BLE001 - static export needs kaleido; the interactive file is still written
        logger.warning(f"Static export failed for {name}: {exc}")
    logger.info(f"Wrote figure {name} (html, png, eps)")


def theme_flags_for_text(df: pd.DataFrame, texts: Iterable[str]) -> pd.DataFrame:
    """Apply the rule-based theme patterns to arbitrary text, one string per row of ``df``."""
    work = pd.DataFrame(index=df.index)
    work = pre._add_theme_flags_rule_based(work, list(texts), THEMES)
    return work[THEMES].astype(bool)
