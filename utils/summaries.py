import pandas as pd
from custom_logger import CustomLogger


logger = CustomLogger(__name__)

# Default scaling factor for saved PNG images.
SCALE = 3


class Summaries():
    def __init__(self) -> None:
        pass

    def summarize_by_duration_bucket(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Duration-bucket level stats: mean(SD) for views, likes,
        views/day, likes/day, engagement rate + same in log10.
        """
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

                mean_duration_minutes=("duration_minutes", "mean"),
                sd_duration_minutes=("duration_minutes", "std"),

                mean_views=("views", "mean"),
                sd_views=("views", "std"),
                median_views=("views", "median"),

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

        logger.info(
            "Duration bucket summary table (raw + log10):\n"
            f"{agg.to_string(index=False)}"
        )
        return agg

    def summarize_by_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """Language-level engagement / growth."""
        agg = (
            df.groupby("language")
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

    def summarize_title_styles(self, df: pd.DataFrame) -> pd.DataFrame:
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
                sd_views=("views", "std"),
                mean_views_per_day=("views_per_day", "mean"),
                sd_views_per_day=("views_per_day", "std"),
                mean_likes=("likes", "mean"),
                sd_likes=("likes", "std"),
                mean_likes_per_day=("likes_per_day", "mean"),
                sd_likes_per_day=("likes_per_day", "std"),
                mean_engagement_rate=("engagement_rate", "mean"),
            )
            .reset_index()
        )

        logger.info(
            "Title length bucket summary:\n"
            f"{agg.to_string(index=False)}"
        )
        return agg

    def summarize_theme_vs_growth(self, df: pd.DataFrame, theme_col: str) -> pd.DataFrame:
        """Compare views_per_day for videos with vs without a given theme flag."""
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

    def compute_monthly_video_counts(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def compute_language_growth(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def compute_theme_trend_over_time(self, df: pd.DataFrame, theme_col: str,
                                      by_language: bool = False) -> pd.DataFrame:
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

    def compute_seasonal_sleep_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
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
