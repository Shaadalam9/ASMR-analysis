"""Cluster-number diagnostics and stability for k-means."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from asmr import settings
from asmr.analysis.supplementary.shared import (
    SEED,
    TEXT_SOURCE,
    logger,
    pre,
    save_figure,
    save_table,
)


def cluster_diagnostics(df: pd.DataFrame) -> None:
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
    from sklearn.preprocessing import OneHotEncoder

    from asmr.analysis.clustering import Clustering

    cu = Clustering()
    d = df.copy()
    d["text_all"] = pre.get_text_series(d, text_source=TEXT_SOURCE)
    numeric_cols = cu._prepare_numeric_features(d)
    prep = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5), "text_all"),
            ("numeric", cu._numeric_pipeline(), numeric_cols),
            ("lang", OneHotEncoder(handle_unknown="ignore"), ["language"]),
        ],
        remainder="drop",
    )
    X = prep.fit_transform(d[["text_all", *numeric_cols, "language"]])
    svd = TruncatedSVD(n_components=50, random_state=SEED)
    Z = svd.fit_transform(X)
    logger.info(f"Feature matrix {X.shape}; SVD explained variance {svd.explained_variance_ratio_.sum():.3f}")

    rng = np.random.default_rng(SEED)
    sil_idx = rng.choice(len(d), size=min(10000, len(d)), replace=False)

    rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in range(2, 21):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        lab = km.labels_
        labels_by_k[k] = lab
        sil_all = silhouette_samples(Z[sil_idx], lab[sil_idx])
        sizes = np.bincount(lab, minlength=k)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette_svd50_sample10k": float(sil_all.mean()),
            "davies_bouldin_svd50": float(davies_bouldin_score(Z, lab)),
            "calinski_harabasz_svd50": float(calinski_harabasz_score(Z, lab)),
            "smallest_cluster": int(sizes.min()),
            "largest_cluster_share": float(sizes.max() / len(lab)),
            "clusters_under_100_videos": int((sizes < 100).sum()),
        })
        logger.info(f"k={k}: silhouette={rows[-1]['silhouette_svd50_sample10k']:.3f}")
    metrics = pd.DataFrame(rows)

    # Stability: re-fit on 80% subsamples in SVD space, assign everyone, compare with the reference labelling
    stab = []
    for k in (4, 6, 8, 11, 14, 17, 20):
        ref = labels_by_k[k]
        aris = []
        for rep in range(5):
            sub = np.random.default_rng(SEED + rep).choice(len(d), size=int(0.8 * len(d)), replace=False)
            km_s = KMeans(n_clusters=k, random_state=SEED + rep, n_init=3).fit(Z[sub])
            aris.append(adjusted_rand_score(ref, km_s.predict(Z)))
        stab.append({"k": k, "mean_ari_vs_reference": float(np.mean(aris)), "sd_ari": float(np.std(aris)),
                     "min_ari": float(np.min(aris))})
    stab_df = pd.DataFrame(stab)
    metrics = metrics.merge(stab_df, on="k", how="left")
    save_table(metrics, "kmeans_k_diagnostics.csv")

    # Per-cluster silhouette for the primary solution and for the finer k=11 solution
    per_all = []
    for kk in sorted({int(settings.get_configs("clustering_n_clusters")), 11}):
        labk = labels_by_k[kk]
        sk = silhouette_samples(Z[sil_idx], labk[sil_idx])
        per = pd.DataFrame({"cluster": labk[sil_idx], "s": sk}).groupby("cluster")["s"].agg(["mean", "size"])
        per = per.rename(columns={"mean": "mean_silhouette", "size": "n_in_silhouette_sample"}).reset_index()
        per["cluster_size"] = per["cluster"].map(pd.Series(np.bincount(labk)))
        per.insert(0, "k", kk)
        per_all.append(per)
    save_table(pd.concat(per_all, ignore_index=True), "kmeans_per_cluster_silhouette.csv")

    primary = int(settings.get_configs("clustering_n_clusters"))
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "(a) Silhouette (higher is better)", "(b) Davies-Bouldin (lower is better)",
        "(c) Stability, ARI on 80% subsamples"))
    for col, ycol in enumerate(("silhouette_svd50_sample10k", "davies_bouldin_svd50", "mean_ari_vs_reference"), 1):
        sub = metrics.dropna(subset=[ycol])
        fig.add_trace(go.Scatter(x=sub["k"], y=sub[ycol], mode="lines+markers", showlegend=False,
                                   hovertemplate="k=%{x}<br>%{y:.3f}<extra></extra>"), row=1, col=col)
        fig.add_vline(x=primary, line_dash="dot", line_color="gray", row=1, col=col)
        fig.update_xaxes(title_text="Number of clusters k", dtick=2, row=1, col=col)
    fig.update_layout(height=450, width=1300)
    save_figure(fig, "kmeans_k_diagnostics", width=1300, height=450)
