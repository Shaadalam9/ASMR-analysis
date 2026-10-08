"""Diversification of supply and concentration of attention over time."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from asmr.analysis.supplementary.shared import (
    THEMES,
    save_figure,
    save_table,
)
from asmr.analysis.supplementary.statistics import gini, hhi, shannon, share_top

PERIODS = [(2008, 2011), (2012, 2015), (2016, 2019), (2020, 2023), (2024, 2026)]


def _period_label(year: float) -> str:
    for a, b in PERIODS:
        if a <= year <= b:
            return f"{a}-{b}"
    return "other"


def ecosystem_structure(df: pd.DataFrame) -> None:
    d = df[df["upload_year"].notna()].copy()
    d["year"] = d["upload_year"].astype(int)
    d["period"] = d["year"].map(_period_label)
    d["any_theme"] = d[THEMES].any(axis=1)
    d["n_themes"] = d[THEMES].sum(axis=1)
    d["is_english"] = d["language"].eq("English")

    def summarise(g: pd.DataFrame) -> dict:
        ch = g["channel_id"].fillna("unknown_" + g["video_id"])
        per_channel = ch.value_counts()
        lang_counts = g["language"].value_counts().to_numpy()
        theme_counts = g[THEMES].sum().to_numpy()
        views = g["views"].dropna().to_numpy()
        ch_views = g.assign(_ch=ch).groupby("_ch")["views"].sum().to_numpy()
        return {
            "n_videos": len(g),
            "n_channels": int(per_channel.size),
            "videos_per_channel_median": float(per_channel.median()),
            "top1pct_channels_share_of_uploads": share_top(per_channel.to_numpy(), 0.01),
            "top10pct_channels_share_of_uploads": share_top(per_channel.to_numpy(), 0.10),
            "pct_non_english": 100 * (1 - g["is_english"].mean()),
            "n_language_categories": int(g["language"].nunique()),
            "language_entropy_norm": shannon(lang_counts, normalise=False) / np.log(max(len(lang_counts), 2)),
            "language_hhi": hhi(lang_counts),
            "pct_any_theme": 100 * g["any_theme"].mean(),
            "mean_theme_labels": g["n_themes"].mean(),
            "theme_entropy_norm": shannon(theme_counts, normalise=True),
            "pct_any_theme_english_only": 100 * g.loc[g["is_english"], "any_theme"].mean(),
            "mean_theme_labels_english_only": g.loc[g["is_english"], "n_themes"].mean(),
            "theme_entropy_norm_english_only": shannon(g.loc[g["is_english"], THEMES].sum().to_numpy(),
                                                       normalise=True),
            "views_gini_videos": gini(views),
            "top1pct_videos_share_of_views": share_top(views, 0.01),
            "top10pct_videos_share_of_views": share_top(views, 0.10),
            "views_gini_channels": gini(ch_views),
            "top1pct_channels_share_of_views": share_top(ch_views, 0.01),
            "median_duration_min": g["duration_minutes"].median(),
            "pct_over_60min": 100 * (g["duration_minutes"] > 60).sum() / max(g["duration_minutes"].notna().sum(), 1),
        }

    by_year = pd.DataFrame([{"year": y, **summarise(g)} for y, g in d.groupby("year")])
    by_period = pd.DataFrame([{"period": p, **summarise(g)} for p, g in d.groupby("period")])
    overall = pd.DataFrame([{"scope": "all", **summarise(d)}])
    save_table(by_year, "ecosystem_by_year.csv")
    save_table(by_period, "ecosystem_by_period.csv")
    save_table(overall, "ecosystem_overall.csv")

    # New-entrant channels: first upload year of each channel
    first = d.assign(_ch=d["channel_id"]).dropna(subset=["_ch"]).groupby("_ch")["year"].min()
    entrants = first.value_counts().sort_index().rename("new_channels").reset_index()
    entrants.columns = ["year", "new_channels"]
    save_table(entrants, "channel_entrants_by_year.csv")

    plot_ecosystem(by_year, d)


def plot_ecosystem(by_year: pd.DataFrame, d: pd.DataFrame) -> None:
    """Four-panel figure: uploads and channels, language mix, theme diversity and Lorenz curves."""
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {"secondary_y": True}], [{"secondary_y": True}, {}]],
        subplot_titles=("(a) Uploads and active channels", "(b) Language diversification",
                        "(c) Theme-label diversity", "(d) Concentration of views (Lorenz curve)"),
        horizontal_spacing=0.12, vertical_spacing=0.16,
    )
    yr = by_year["year"]
    fig.add_trace(go.Scatter(x=yr, y=by_year["n_videos"], mode="lines+markers", name="Videos"), row=1, col=1)
    fig.add_trace(go.Scatter(x=yr, y=by_year["n_channels"], mode="lines+markers", name="Active channels",
                             line=dict(dash="dash")), row=1, col=1)
    fig.update_yaxes(type="log", title_text="Count (log scale)", row=1, col=1)
    fig.add_trace(go.Scatter(x=yr, y=by_year["pct_non_english"], mode="lines+markers",
                             name="Non-English uploads (%)"), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=yr, y=by_year["n_language_categories"], mode="lines+markers",
                             name="Language categories", line=dict(dash="dash")), row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Non-English uploads (%)", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Language categories", row=1, col=2, secondary_y=True)
    fig.add_trace(go.Scatter(x=yr, y=by_year["theme_entropy_norm"], mode="lines+markers",
                             name="Normalised theme entropy"), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=yr, y=by_year["mean_theme_labels"], mode="lines+markers",
                             name="Mean theme labels per video", line=dict(dash="dash")), row=2, col=1,
                  secondary_y=True)
    fig.update_yaxes(title_text="Normalised theme entropy", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Mean labels per video", row=2, col=1, secondary_y=True)
    channel_views = d.dropna(subset=["channel_id"]).groupby("channel_id")["views"].sum().to_numpy()
    for label, vals in (("Videos", d["views"].dropna().to_numpy()), ("Channels", channel_views)):
        x = np.sort(vals)
        cum = np.insert(np.cumsum(x) / x.sum(), 0, 0)
        pos = np.unique(np.linspace(0, cum.size - 1, 1500).astype(int))
        fig.add_trace(go.Scatter(x=np.linspace(0, 1, cum.size)[pos], y=cum[pos], mode="lines",
                                 name=f"{label} (Gini {gini(vals):.2f})"), row=2, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Equality", line=dict(dash="dot")),
                  row=2, col=2)
    fig.update_xaxes(title_text="Cumulative share of units", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative share of views", row=2, col=2)
    for r, c in ((1, 1), (1, 2), (2, 1)):
        fig.update_xaxes(title_text="Upload year", dtick=2, row=r, col=c)
    fig.update_layout(height=800, width=1200, legend=dict(orientation="h", y=-0.12))
    save_figure(fig, "ecosystem_diversification")
