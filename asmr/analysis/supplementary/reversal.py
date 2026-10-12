"""Means versus medians and pooled versus within-channel duration associations (central figure)."""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from asmr.analysis.supplementary.shared import DURATION_ORDER, OUT, save_figure, save_table

BUCKET_LABELS = {"under_10min": "Under 10 min", "10_to_30min": "10 to 30 min", "30_to_60min": "30 to 60 min",
                 "60_to_180min": "60 to 180 min", "over_180min": "Over 180 min"}
MODELS = [
    ("pooled (year fixed effects)", "Pooled, all videos", "#999999"),
    ("pooled, same sample as within channel", "Pooled, same videos as within-channel model", "#4c78a8"),
    ("within channel (channel + year fixed effects)", "Within channel", "#e45756"),
]


def duration_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["duration_bucket"].isin(DURATION_ORDER) & df["views_per_day"].notna()]
    g = d.groupby("duration_bucket")["views_per_day"]
    out = pd.DataFrame({"n": g.size(), "mean_views_per_day": g.mean(), "median_views_per_day": g.median()})
    out["mean_to_median_ratio"] = out["mean_views_per_day"] / out["median_views_per_day"]
    return out.reindex(DURATION_ORDER).reset_index()


def reversal_figure(df: pd.DataFrame) -> None:
    summary = duration_summary(df)
    save_table(summary, "duration_mean_vs_median.csv")
    coefs = pd.read_csv(os.path.join(OUT, "model_coefficients.csv"))
    coefs = coefs[(coefs["outcome"] == "log10 views per day") & coefs["term"].str.startswith("dur_")]
    coefs = coefs.assign(bucket=coefs["term"].str.replace("dur_", "", regex=False))
    labels = [BUCKET_LABELS[b] for b in DURATION_ORDER]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58], horizontal_spacing=0.12, subplot_titles=(
        "(a) Mean and median views per day", "(b) Difference from 10 to 30 min videos"))
    fig.add_trace(go.Bar(x=labels, y=summary["mean_views_per_day"], name="Mean", marker_color="#f58518"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=summary["median_views_per_day"], name="Median", marker_color="#54a24b"),
                  row=1, col=1)
    fig.update_yaxes(type="log", title_text="Views per day (log scale)", row=1, col=1)
    fig.update_layout(barmode="group")

    shown = [b for b in DURATION_ORDER if b != "10_to_30min"]
    for i, (key, name, colour) in enumerate(MODELS):
        sub = coefs[coefs["model"] == key].set_index("bucket")
        sub = sub.reindex(shown)
        x = [j + (i - 1) * 0.18 for j in range(len(shown))]
        ratio = 1 + sub["pct_difference"] / 100
        lo = 1 + sub["pct_ci_low"] / 100
        hi = 1 + sub["pct_ci_high"] / 100
        fig.add_trace(go.Scatter(
            x=x, y=ratio, mode="markers", name=name, marker=dict(color=colour, size=10),
            error_y=dict(type="data", symmetric=False, array=(hi - ratio).clip(lower=0),
                         arrayminus=(ratio - lo).clip(lower=0), color=colour),
            hovertemplate="ratio %{y:.2f}<extra>" + name + "</extra>"), row=1, col=2)
    fig.add_hline(y=1, line_dash="dot", line_color="gray", row=1, col=2)
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(shown))),
                     ticktext=[BUCKET_LABELS[b] for b in shown], row=1, col=2)
    fig.update_yaxes(type="log", title_text="Ratio of views per day (log scale)", row=1, col=2)
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    save_figure(fig, "duration_reversal", width=1300, height=520)
