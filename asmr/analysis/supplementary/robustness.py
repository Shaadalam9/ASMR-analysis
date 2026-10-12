"""Stability of the pooled versus within-channel duration associations across periods, languages and channels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from asmr.analysis.supplementary.models import _design, _ols_cluster, n_groups_fe_count
from asmr.analysis.supplementary.reversal import BUCKET_LABELS
from asmr.analysis.supplementary.shared import DURATION_ORDER, save_figure, save_table

OUTCOME = "log10_views_per_day"
DURATION_TERMS = [f"dur_{b}" for b in DURATION_ORDER if b != "10_to_30min"]


def _top_channels(df: pd.DataFrame, by: str, share: float = 0.01) -> set:
    ch = df["channel_id"].fillna("unknown_" + df["video_id"])
    totals = df.assign(_ch=ch).groupby("_ch")["views"].sum() if by == "views" else ch.value_counts()
    k = max(int(round(len(totals) * share)), 1)
    return set(totals.nlargest(k).index)


def _subsets(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    ch = df["channel_id"].fillna("unknown_" + df["video_id"])
    year = df["upload_year"]
    return [
        ("All videos", df),
        ("Uploaded 2008-2015", df[year <= 2015]),
        ("Uploaded 2016-2019", df[year.between(2016, 2019)]),
        ("Uploaded 2020-2026", df[year >= 2020]),
        ("English only", df[df["language"].eq("English")]),
        ("Non-English only", df[~df["language"].eq("English")]),
        ("Without top 1% of channels by views", df[~ch.isin(_top_channels(df, "views"))]),
        ("Without top 1% of channels by uploads", df[~ch.isin(_top_channels(df, "uploads"))]),
    ]


def _fit(df: pd.DataFrame, label: str) -> list[dict]:
    d, terms, _ = _design(df)
    sub = d[d[OUTCOME].replace([np.inf, -np.inf], np.nan).notna()]
    y = sub[OUTCOME].to_numpy(float)
    groups = sub["channel_id"].fillna("unknown_" + sub["video_id"]).to_numpy()
    pooled, _, n_ch = _ols_cluster(y, sub[terms].to_numpy(float), groups, terms)

    cnt = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    keep = cnt >= 2
    sw = sub[keep]
    g = pd.Series(groups[keep], index=sw.index)
    yw = sw[OUTCOME] - sw[OUTCOME].groupby(g).transform("mean")
    Xw = sw[terms] - sw[terms].groupby(g).transform("mean")
    varying = [t for t in terms if Xw[t].abs().sum() > 1e-9]
    within, _, n_ch_w = _ols_cluster(yw.to_numpy(float), Xw[varying].to_numpy(float), g.to_numpy(), varying,
                                     intercept=False, dof_extra=n_groups_fe_count(g))
    rows = []
    fits = (("pooled", pooled, len(sub), n_ch), ("within channel", within, int(keep.sum()), n_ch_w))
    for model, res, n, n_c in fits:
        res = res.set_index("term")
        for t in DURATION_TERMS:
            if t not in res.index:
                continue
            r = res.loc[t]
            rows.append({"subset": label, "model": model, "term": t, "n": n, "n_channels": n_c,
                         "pct_difference": (10 ** r["coef"] - 1) * 100,
                         "pct_ci_low": (10 ** r["ci_low"] - 1) * 100,
                         "pct_ci_high": (10 ** r["ci_high"] - 1) * 100, "p": r["p"]})
    return rows


def duration_robustness(df: pd.DataFrame) -> None:
    rows = []
    for label, sub in _subsets(df):
        rows += _fit(sub, label)
    res = pd.DataFrame(rows)
    save_table(res, "duration_robustness.csv")
    plot_duration_robustness(res)


def plot_duration_robustness(res: pd.DataFrame) -> None:
    labels = list(dict.fromkeys(res["subset"]))
    terms = DURATION_TERMS
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=len(terms), shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=[
        f"{BUCKET_LABELS[t.replace('dur_', '')]}" for t in terms])
    for c, term in enumerate(terms, 1):
        for model, colour, offset in (("pooled", "#999999", 0.14), ("within channel", "#e45756", -0.14)):
            sub = res[(res["term"] == term) & (res["model"] == model)].set_index("subset").reindex(labels)
            ratio = 1 + sub["pct_difference"] / 100
            lo, hi = 1 + sub["pct_ci_low"] / 100, 1 + sub["pct_ci_high"] / 100
            fig.add_trace(go.Scatter(
                x=ratio, y=[len(labels) - 1 - i + offset for i in range(len(labels))], mode="markers",
                name="Pooled" if model == "pooled" else "Within channel", showlegend=(c == 1),
                legendgroup=model, marker=dict(color=colour, size=8),
                error_x=dict(type="data", symmetric=False, array=(hi - ratio).clip(lower=0),
                             arrayminus=(ratio - lo).clip(lower=0), color=colour, thickness=1.2),
                hovertemplate="ratio %{x:.2f}<extra>" + model + "</extra>"), row=1, col=c)
        fig.add_vline(x=1, line_dash="dot", line_color="gray", row=1, col=c)
        fig.update_xaxes(type="log", title_text="Ratio of views per day", row=1, col=c)
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels[::-1], row=1, col=1)
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    save_figure(fig, "duration_robustness", width=1500, height=620)
