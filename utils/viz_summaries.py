import plotly.express as px
import pandas as pd
from custom_logger import CustomLogger

from utils.viz_core import Plots

logger = CustomLogger(__name__)

plot_class = Plots()

# Default scaling factor for saved PNG images.
SCALE = 3


class Viz_summaries():
    def __init__(self) -> None:
        pass

    def plot_language_stats(self, lang_stats: pd.DataFrame, min_videos: int = 20) -> None:
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
        plot_class.save_plotly_figure(
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
        plot_class.save_plotly_figure(
            fig,
            "language_mean_engagement_rate",
            width=1600,
            height=900,
            scale=SCALE,
        )

    def plot_title_style_stats(self, title_stats: pd.DataFrame) -> None:
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
        plot_class.save_plotly_figure(
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
        plot_class.save_plotly_figure(
            fig,
            "title_length_mean_views",
            width=1600,
            height=900,
            scale=SCALE,
        )

    def plot_theme_growth_box(self, df: pd.DataFrame, theme_col: str) -> None:
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
        plot_class.save_plotly_figure(fig, filename, width=1600, height=900, scale=SCALE)

    def plot_language_growth(self, lang_growth: pd.DataFrame, min_total_videos: int = 50) -> None:
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

        # Make the dots bigger
        fig.update_traces(
            mode="lines+markers",
            marker=dict(
                size=10,        # increase this number for larger dots (e.g. 8, 10, 12)
                line=dict(width=1),
            ),
        )

        plot_class.save_plotly_figure(
            fig,
            "language_growth_over_years",
            width=1600,
            height=900,
            scale=SCALE,
        )

    def plot_theme_trend_overall(self, trend_df: pd.DataFrame, theme_col: str) -> None:
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

        df_plot["upload_year"] = pd.to_numeric(df_plot["upload_year"], errors="coerce")
        df_plot["theme_count"] = pd.to_numeric(df_plot["theme_count"], errors="coerce").fillna(0)
        df_plot = df_plot.dropna(subset=["upload_year"]).sort_values("upload_year")

        total_themed = int(df_plot["theme_count"].sum())
        logger.info(
            f"Overall theme trend for {theme_col}: "
            f"{len(df_plot)} years, total themed videos={total_themed}."
        )
        if total_themed == 0:
            logger.warning(
                f"Overall theme trend for {theme_col} is all zero; the plot is being saved "
                "for diagnostics, but the theme flags should be checked."
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

        fig.update_traces(
            mode="lines+markers",
            marker=dict(
                size=10,        # increase this number for larger dots (e.g. 8, 10, 12)
                line=dict(width=1),
            ),
        )

        filename = f"{theme_col}_trend_overall_fig"
        plot_class.save_plotly_figure(fig, filename, width=1600, height=900, scale=SCALE)

    def plot_theme_trend_by_language(self, trend_df: pd.DataFrame, theme_col: str, min_videos: int = 30) -> None:
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

        df_plot["upload_year"] = pd.to_numeric(df_plot["upload_year"], errors="coerce")
        df_plot["theme_count"] = pd.to_numeric(df_plot["theme_count"], errors="coerce").fillna(0)
        df_plot = df_plot.dropna(subset=["upload_year"]).sort_values(["language", "upload_year"])

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
        plot_class.save_plotly_figure(fig, filename, width=1600, height=900, scale=SCALE)

    def plot_monthly_counts(self, monthly_counts: pd.DataFrame) -> None:
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
        plot_class.save_plotly_figure(
            fig,
            "monthly_video_counts",
            width=1600,
            height=900,
            scale=SCALE,
        )
