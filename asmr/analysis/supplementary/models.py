"""Multivariable models of views per day and engagement with year and channel fixed effects."""
from __future__ import annotations

import numpy as np
import pandas as pd

from asmr.analysis.supplementary.shared import (
    DURATION_ORDER,
    THEMES,
    save_table,
    theme_flags_for_text,
)


def _design(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Return the modelling frame, the ordered term names, and each term's block."""
    d = df[df["duration_bucket"].isin(DURATION_ORDER) & df["upload_year"].notna()].copy()
    block_of: dict[str, str] = {}
    new_cols: dict[str, pd.Series] = {}

    years = sorted(d["upload_year"].astype(int).unique())
    ref_year = 2018 if 2018 in years else years[len(years) // 2]
    for y in years:
        if y != ref_year:
            new_cols[f"year_{y}"] = (d["upload_year"].astype(int) == y).astype(float)
            block_of[f"year_{y}"] = "year"
    for b in DURATION_ORDER:
        if b != "10_to_30min":
            new_cols[f"dur_{b}"] = (d["duration_bucket"] == b).astype(float)
            block_of[f"dur_{b}"] = "duration"
    for t in THEMES:
        d[t] = d[t].astype(float)
        block_of[t] = "theme"
    for lang in ("Korean", "Japanese", "Spanish", "Portuguese"):
        new_cols[f"lang_{lang}"] = (d["language"] == lang).astype(float)
        block_of[f"lang_{lang}"] = "language"
    major = ["English", "Korean", "Japanese", "Spanish", "Portuguese"]
    new_cols["lang_Other_or_unknown"] = (~d["language"].isin(major)).astype(float)
    block_of["lang_Other_or_unknown"] = "language"
    new_cols["title_log_words"] = np.log(d["title_word_count"].clip(lower=1))
    block_of["title_log_words"] = "title"
    for c in ("title_has_brackets", "title_has_all_caps_word", "title_has_exclamation",
              "title_has_question", "title_has_hashtag"):
        d[c] = d[c].astype(float)
        block_of[c] = "title"
    d = pd.concat([d, pd.DataFrame(new_cols, index=d.index)], axis=1)
    order = ["year", "duration", "theme", "language", "title"]
    terms = [t for blk in order for t, bo in block_of.items() if bo == blk]
    return d, terms, block_of


def _ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray, names: list[str], intercept: bool = True,
                 dof_extra: int = 0) -> tuple[pd.DataFrame, float, int]:
    if intercept:
        X = np.column_stack([np.ones(len(y)), X])
        names = ["intercept"] + names
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    scores = X * resid[:, None]
    meat_groups = np.zeros((G, X.shape[1]))
    np.add.at(meat_groups, inv, scores)
    meat = meat_groups.T @ meat_groups
    n, k = X.shape
    adj = (G / (G - 1)) * ((n - 1) / (n - k - dof_extra))
    cov = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    from scipy import stats
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=max(G - 1, 1))
    out = pd.DataFrame({"term": names, "coef": beta, "se": se, "t": t, "p": p,
                        "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se})
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / tss
    return out, float(r2), G


def models(df: pd.DataFrame, title_only: bool = False) -> None:
    if title_only:
        df = df.copy()
        title_flags = theme_flags_for_text(df, df["title"].fillna("").astype(str))
        for col in THEMES:
            df[col] = title_flags[col]
    suffix = "_title_only_themes" if title_only else ""
    d, terms, block_of = _design(df)

    results = []
    r2_rows = []
    for outcome, label in (("log10_views_per_day", "log10 views per day"),
                           ("log10_engagement_rate", "log10 engagement rate")):
        sub = d[d[outcome].replace([np.inf, -np.inf], np.nan).notna()].copy()
        y = sub[outcome].to_numpy(float)
        groups = sub["channel_id"].fillna("unknown_" + sub["video_id"]).to_numpy()

        # cumulative block R^2 (year first, because views per day depends strongly on age)
        used: list[str] = []
        for blk in ("year", "duration", "theme", "language", "title"):
            used += [t for t in terms if block_of[t] == blk]
            _, r2, _ = _ols_cluster(y, sub[used].to_numpy(float), groups, used)
            r2_rows.append({"outcome": label, "cumulative_blocks_up_to": blk, "r2": r2, "n": len(sub)})

        coefs, r2_full, n_groups = _ols_cluster(y, sub[terms].to_numpy(float), groups, terms)
        coefs["model"] = "pooled (year fixed effects)"
        coefs["outcome"] = label
        coefs["n"] = len(sub)
        coefs["n_channels"] = n_groups
        coefs["r2"] = r2_full
        results.append(coefs)

        # within-channel: demean outcome and predictors by channel (absorbs channel fixed effects)
        cnt = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
        keep = cnt >= 2
        sw = sub[keep]
        g_ser = pd.Series(groups[keep], index=sw.index)
        yw = sw[outcome] - sw[outcome].groupby(g_ser).transform("mean")
        Xw = sw[terms] - sw[terms].groupby(g_ser).transform("mean")
        varying = [t for t in terms if Xw[t].abs().sum() > 1e-9]
        coefs_fe, r2_w, n_groups_fe = _ols_cluster(
            yw.to_numpy(float), Xw[varying].to_numpy(float), g_ser.to_numpy(), varying,
            intercept=False, dof_extra=n_groups_fe_count(g_ser),
        )
        coefs_fe["model"] = "within channel (channel + year fixed effects)"
        coefs_fe["outcome"] = label
        coefs_fe["n"] = int(keep.sum())
        coefs_fe["n_channels"] = n_groups_fe
        coefs_fe["r2"] = r2_w
        results.append(coefs_fe)

    res = pd.concat(results, ignore_index=True)
    res["pct_difference"] = (np.power(10.0, res["coef"]) - 1) * 100
    res["pct_ci_low"] = (np.power(10.0, res["ci_low"]) - 1) * 100
    res["pct_ci_high"] = (np.power(10.0, res["ci_high"]) - 1) * 100
    res["block"] = res["term"].map(block_of).fillna("intercept")
    save_table(res, f"model_coefficients{suffix}.csv")
    save_table(pd.DataFrame(r2_rows), f"model_block_r2{suffix}.csv")


def n_groups_fe_count(g: pd.Series) -> int:
    """Number of absorbed channel fixed effects (for the degrees-of-freedom correction)."""
    return int(g.nunique())


def models_title_only(df: pd.DataFrame) -> None:
    models(df, title_only=True)
