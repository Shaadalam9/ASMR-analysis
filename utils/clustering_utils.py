import pandas as pd
import numpy as np
from custom_logger import CustomLogger
from typing import Any, Dict, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from utils.preprocessing import Preprocessing
from utils.viz_core import Plots


logger = CustomLogger(__name__)


# Default scaling factor for saved PNG images.
SCALE = 3

pre_process_class = Preprocessing()
plots_class = Plots()


class Clustering_utils():
    def __init__(self) -> None:
        pass

    def cluster_videos(self, df: pd.DataFrame, n_clusters: int = 11, random_state: int = 42,
                       text_source: str = "both") -> Tuple[pd.DataFrame, Optional[Pipeline], Optional[pd.DataFrame]]:
        """Cluster videos using title/description text, duration, engagement, and language."""
        df_copy = df.copy()
        df_copy["text_all"] = pre_process_class.get_text_series(df_copy, text_source=text_source)

        feature_cols = [
            "text_all",
            "duration_minutes",
            "engagement_rate",
            "views_per_day",
            "language",
        ]

        for col in ["duration_minutes", "engagement_rate", "views_per_day"]:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0.0)

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "text",
                    TfidfVectorizer(
                        max_features=5000,
                        ngram_range=(1, 2),
                        min_df=5,
                    ),
                    "text_all",
                ),
                (
                    "numeric",
                    StandardScaler(with_mean=False),
                    ["duration_minutes", "engagement_rate", "views_per_day"],
                ),
                (
                    "lang",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["language"],
                ),
            ],
            remainder="drop",
        )

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "cluster",
                    KMeans(
                        n_clusters=n_clusters,
                        random_state=random_state,
                        n_init=10,
                    ),
                ),
            ]
        )

        X = df_copy[feature_cols]
        logger.info(
            f"Fitting clustering model on {len(df_copy)} videos (text_source={text_source}, "
            f"n_clusters={n_clusters})"
        )
        pipeline.fit(X)

        logger.info("Assigning cluster labels...")
        df_copy["cluster"] = pipeline.predict(X)

        pca_info: Optional[pd.DataFrame] = None

        try:
            logger.info("Computing 2D PCA embedding for cluster visualization...")
            features = pipeline.named_steps["preprocess"].transform(X)

            reducer = PCA(n_components=2, random_state=random_state)
            embedding_2d = reducer.fit_transform(features)

            df_copy["embedding_x"] = embedding_2d[:, 0]
            df_copy["embedding_y"] = embedding_2d[:, 1]

            pca_info = pd.DataFrame(
                {
                    "component": [1, 2],
                    "explained_variance_ratio": reducer.explained_variance_ratio_,
                }
            )
            logger.info(
                "PCA explained variance ratios for first 2 components: "
                f"{list(reducer.explained_variance_ratio_)}"
            )
        except Exception as exc:
            logger.warning(f"Could not compute 2D embedding for clusters: {exc}")
            df_copy["embedding_x"] = np.nan
            df_copy["embedding_y"] = np.nan

        if "cluster" in df_copy.columns:
            cluster_counts = df_copy["cluster"].value_counts().sort_index()
            logger.info(
                "Cluster sizes (PCA-based clustering):\n"
                f"{cluster_counts.to_string()}"
            )
            mean_vpd = (
                df_copy.groupby("cluster")["views_per_day"]
                .mean()
                .round(2)
            )
            logger.info(
                "Mean views_per_day by cluster:\n"
                f"{mean_vpd.to_string()}"
            )

        return df_copy, pipeline, pca_info

    def cluster_videos_tsne(self, df: pd.DataFrame, n_clusters: int = 11, random_state: int = 42,
                            text_source: str = "both", tsne_perplexity: float = 30.0,
                            tsne_learning_rate: float = 200.0, tsne_n_iter: int = 1000
                            ) -> Tuple[pd.DataFrame, Optional[Pipeline]]:
        """
        Same idea as `cluster_videos`, but the 2D embedding used for visualization
        is computed with t-SNE (after an SVD pre-step for speed).
        """
        df_copy = df.copy()
        df_copy["text_all"] = pre_process_class.get_text_series(df_copy, text_source=text_source)

        feature_cols = [
            "text_all",
            "duration_minutes",
            "engagement_rate",
            "views_per_day",
            "language",
        ]

        for col in ["duration_minutes", "engagement_rate", "views_per_day"]:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0.0)

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "text",
                    TfidfVectorizer(
                        max_features=5000,
                        ngram_range=(1, 2),
                        min_df=5,
                    ),
                    "text_all",
                ),
                (
                    "numeric",
                    StandardScaler(with_mean=False),
                    ["duration_minutes", "engagement_rate", "views_per_day"],
                ),
                (
                    "lang",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["language"],
                ),
            ],
            remainder="drop",
        )

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "cluster",
                    KMeans(
                        n_clusters=n_clusters,
                        random_state=random_state,
                        n_init=10,
                    ),
                ),
            ]
        )

        X = df_copy[feature_cols]
        logger.info(
            f"Fitting clustering model with t-SNE embedding on {len(df_copy)} videos "
            f"(text_source={text_source}, n_clusters={n_clusters})"
        )
        pipeline.fit(X)

        logger.info("Assigning cluster labels (t-SNE version)...")
        df_copy["cluster"] = pipeline.predict(X)

        try:
            logger.info("Computing 2D t-SNE embedding for cluster visualization...")
            features = pipeline.named_steps["preprocess"].transform(X)

            n_features = features.shape[1]
            n_samples = features.shape[0]
            max_components = min(n_features, n_samples - 1)

            if max_components >= 2:
                svd = TruncatedSVD(
                    n_components=max_components,
                    random_state=random_state,
                )
                features_reduced = svd.fit_transform(features)
                logger.info(
                    f"Reduced feature space to {max_components} dimensions via TruncatedSVD "
                    f"before t-SNE."
                )
            else:
                features_reduced = (
                    features.toarray() if hasattr(features, "toarray") else np.array(features)
                )
                logger.info(
                    "Skipped SVD reduction before t-SNE because dataset is very small."
                )

            tsne = TSNE(
                n_components=2,
                perplexity=tsne_perplexity,
                learning_rate=tsne_learning_rate,
                random_state=random_state,
                init="random",
            )
            embedding_2d = tsne.fit_transform(features_reduced)

            df_copy["embedding_x"] = embedding_2d[:, 0]
            df_copy["embedding_y"] = embedding_2d[:, 1]

            logger.info(
                "Completed t-SNE embedding with "
                f"{embedding_2d.shape[0]} points."
            )

            if "cluster" in df_copy.columns:
                cluster_counts = df_copy["cluster"].value_counts().sort_index()
                logger.info(
                    "Cluster sizes (t-SNE embedding version):\n"
                    f"{cluster_counts.to_string()}"
                )

        except Exception as exc:
            logger.warning(f"Could not compute 2D t-SNE embedding for clusters: {exc}")
            df_copy["embedding_x"] = np.nan
            df_copy["embedding_y"] = np.nan

        return df_copy, pipeline

    def describe_clusters(self, clustered_df: pd.DataFrame, top_n_languages: int = 3) -> pd.DataFrame:
        """
        Produce a summary description per cluster to help interpret what it 'is'.
        """
        df = clustered_df.copy()

        theme_cols = [
            c for c in df.columns
            if c.startswith("has_") and df[c].dtype == bool
        ]

        summaries = []
        for cluster_id, group in df.groupby("cluster"):
            row: Dict[str, Any] = {
                "cluster": int(cluster_id),  # type: ignore
                "n_videos": len(group),
            }

            for col in ["views", "views_per_day", "likes", "likes_per_day", "duration_minutes", "engagement_rate"]:
                if col in group.columns:
                    row[f"mean_{col}"] = float(group[col].mean(skipna=True))

            if "language" in group.columns:
                lang_counts = group["language"].value_counts()
                top_langs = lang_counts.head(top_n_languages)
                row["top_languages"] = "; ".join(
                    f"{lang} ({cnt})" for lang, cnt in top_langs.items()
                )

            for tcol in theme_cols:
                row[f"share_{tcol}"] = float(group[tcol].mean())

            summaries.append(row)

        cluster_desc = pd.DataFrame(summaries).sort_values("cluster").reset_index(drop=True)
        logger.info(
            "Cluster interpretation summary:\n"
            f"{cluster_desc.to_string(index=False)}"
        )
        return cluster_desc

    def compute_kmeans_elbow(self, df: pd.DataFrame, k_values: range, text_source: str = "both",
                             random_state: int = 42) -> pd.DataFrame:
        """
        Compute KMeans inertia (SSE) for a range of k using the same
        feature setup as `cluster_videos`, and save an elbow plot.

        Returns a DataFrame with columns: k, inertia.
        """

        # --- Build the same feature matrix as in cluster_videos ---
        df_copy = df.copy()
        df_copy["text_all"] = pre_process_class.get_text_series(df_copy, text_source=text_source)

        feature_cols = [
            "text_all",
            "duration_minutes",
            "engagement_rate",
            "views_per_day",
            "language",
        ]

        # Ensure numeric columns are numeric
        for col in ["duration_minutes", "engagement_rate", "views_per_day"]:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0.0)

        # ColumnTransformer identical to cluster_videos
        preprocess = ColumnTransformer(
            transformers=[
                (
                    "text",
                    TfidfVectorizer(
                        max_features=5000,
                        ngram_range=(1, 2),
                        min_df=5,
                    ),
                    "text_all",
                ),
                (
                    "numeric",
                    StandardScaler(with_mean=False),
                    ["duration_minutes", "engagement_rate", "views_per_day"],
                ),
                (
                    "lang",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["language"],
                ),
            ],
            remainder="drop",
        )

        X_raw = df_copy[feature_cols]

        logger.info(
            f"Fitting preprocessing for elbow curve on {len(df_copy)} videos "
            f"(text_source={text_source})"
        )
        X_features = preprocess.fit_transform(X_raw)

        # --- Loop over k and compute inertia ---
        records = []
        for k in k_values:
            logger.info(f"[Elbow] Fitting KMeans for k={k} ...")
            km = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10,
            )
            km.fit(X_features)
            inertia = km.inertia_
            logger.info(f"[Elbow] k={k}: inertia={inertia:,.2f}")
            records.append({"k": k, "inertia": inertia})

        results = pd.DataFrame(records)

        # --- Plot inertia vs k with Plotly and your styling ---
        fig = px.line(
            results,
            x="k",
            y="inertia",
            markers=True,
            title="",
        )
        fig.update_traces(mode="lines+markers")

        plots_class.save_plotly_figure(
            fig,
            filename=f"kmeans_elbow_{text_source}",
            width=1200,
            height=700,
            scale=3,
        )

        return results

    def plot_cluster_distribution(self, clustered_df: pd.DataFrame, name_suffix: str = "",
                                  highlight_cluster: Optional[int] = None) -> None:
        """Visualize clusters: bar charts + 2D scatter with circles around clusters."""
        if "cluster" not in clustered_df.columns:
            logger.warning("No 'cluster' column in clustered_df; skipping cluster plots.")
            return

        df_plot = clustered_df.copy()
        suffix = f"_{name_suffix}" if name_suffix else ""

        agg = (
            df_plot.groupby("cluster")
            .agg(
                video_count=("video_id", "count"),
                mean_views=("views", "mean"),
                mean_views_per_day=("views_per_day", "mean"),
            )
            .reset_index()
        )

        logger.info(
            f"Cluster distribution summary (suffix='{name_suffix}'):\n"
            f"{agg.to_string(index=False)}"
        )

        fig = px.bar(
            agg,
            x="cluster",
            y="video_count",
            title="",
            labels={"cluster": "Cluster", "video_count": "Number of videos"},
        )
        plots_class.save_plotly_figure(
            fig,
            f"cluster_sizes{suffix}",
            width=1600,
            height=900,
            scale=SCALE,
        )

        fig = px.bar(
            agg,
            x="cluster",
            y="mean_views_per_day",
            title="",
            labels={
                "cluster": "Cluster",
                "mean_views_per_day": "Mean views per day",
            },
        )
        plots_class.save_plotly_figure(
            fig,
            f"cluster_mean_views_per_day{suffix}",
            width=1600,
            height=900,
            scale=SCALE,
        )

        if "embedding_x" not in df_plot.columns or "embedding_y" not in df_plot.columns:
            logger.warning(
                "No embedding_x / embedding_y columns found; skipping cluster scatter plot."
            )
            return

        df_emb = df_plot.dropna(subset=["embedding_x", "embedding_y"]).copy()
        if df_emb.empty:
            logger.warning(
                "Embedding columns are empty; skipping cluster scatter plot."
            )
            return

        df_emb["cluster"] = df_emb["cluster"].astype(int)
        df_emb["cluster_str"] = df_emb["cluster"].astype(str)
        cluster_order = sorted(df_emb["cluster_str"].unique())
        color_discrete_map = plots_class._get_cluster_color_map(df_emb["cluster"].values)  # type: ignore

        fig = px.scatter(
            df_emb,
            x="embedding_x",
            y="embedding_y",
            color="cluster_str",
            hover_data=["video_id", "title", "language", "views", "duration_minutes"],
            title="",
            labels={
                "embedding_x": "",
                "embedding_y": "",
                "cluster_str": "",
            },
            category_orders={"cluster_str": cluster_order},
            color_discrete_map=color_discrete_map,
        )

        # Optionally highlight a single cluster (larger markers + black outline)
        if highlight_cluster is not None:
            highlight_label = str(highlight_cluster)

            fig.for_each_trace(
                lambda trace: trace.update(
                    marker=dict(
                        size=10,
                        line=dict(width=2, color="black"),
                    )
                )
                if trace.name == highlight_label
                else trace.update(
                    marker=dict(
                        size=5,
                        opacity=0.3,
                    )
                )
            )

        shapes = []
        for cluster_id, group in df_emb.groupby("cluster"):
            if len(group) < 2:
                continue

            cx = group["embedding_x"].mean()
            cy = group["embedding_y"].mean()
            distances = np.sqrt(
                (group["embedding_x"] - cx) ** 2 + (group["embedding_y"] - cy) ** 2
            )

            radius = float(distances.quantile(0.8))  # type: ignore
            if not np.isfinite(radius) or radius <= 0:
                continue

            shapes.append(
                dict(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=cx - radius,
                    y0=cy - radius,
                    x1=cx + radius,
                    y1=cy + radius,
                    line=dict(width=1, dash="dot"),
                    opacity=0.3,
                )
            )

        if shapes:
            fig.update_layout(shapes=shapes)

        plots_class.save_plotly_figure(fig, f"cluster_scatter_embedding{suffix}", width=1600, height=900, scale=SCALE)

    def plot_tsne_research(self, df: pd.DataFrame, name_suffix: str = "tsne_research", label_clusters: bool = True,
                           ellipse_scale: float = 1.2):
        """
        Produce a research-style t-SNE plot with:
        - stable color map
        - ellipse-like outlines around clusters (via Plotly path shapes)
        - labeled cluster centroids
        - extensive logger output to interpret clusters
        """

        # Require necessary columns
        if {"cluster", "embedding_x", "embedding_y"} - set(df.columns):
            logger.warning("t-SNE research plot: missing embedding_x/y or cluster.")
            return

        df = df.dropna(subset=["embedding_x", "embedding_y"]).copy()
        if df.empty:
            logger.warning("t-SNE research plot: no non-null embeddings.")
            return

        df["cluster"] = df["cluster"].astype(int)
        df["cluster_str"] = df["cluster"].astype(str)

        # ------------------------------------------------------------------
        # LOGGING
        # ------------------------------------------------------------------
        logger.info("===== t-SNE RESEARCH PLOT SUMMARY =====")
        logger.info(f"Total points in t-SNE: {len(df)}")

        logger.info(
            "Cluster sizes:\n"
            + df["cluster"].value_counts().sort_index().to_string()
        )

        if "language" in df.columns:
            logger.info("Top languages per cluster:")
            for cid, grp in df.groupby("cluster"):
                lang = grp["language"].value_counts().head(5)
                logger.info(f"  cluster {cid}:\n{lang.to_string()}")

        theme_cols = [c for c in df.columns if c.startswith("has_")]
        for theme_col in theme_cols:
            mean_vals = df.groupby("cluster")[theme_col].mean().round(3)
            logger.info(f"{theme_col} prevalence per cluster:\n{mean_vals.to_string()}")

        logger.info("================================================")

        # ------------------------------------------------------------------
        # COLOR MAP (stable by cluster id)
        # ------------------------------------------------------------------
        palette = px.colors.qualitative.Plotly
        unique_clusters = sorted(df["cluster"].unique())
        color_map = {
            str(c): palette[i % len(palette)]
            for i, c in enumerate(unique_clusters)
        }

        # ------------------------------------------------------------------
        # SCATTER LAYER
        # ------------------------------------------------------------------
        fig = go.Figure()

        for cid, grp in df.groupby("cluster"):
            fig.add_trace(
                go.Scatter(
                    x=grp["embedding_x"],
                    y=grp["embedding_y"],
                    mode="markers",
                    marker=dict(
                        size=7,  # bigger dots
                        color=color_map[str(cid)],
                        opacity=0.3,
                        # thin outline to make markers pop a bit more
                        line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
                    ),
                    name=f"Cluster {cid}",
                    text=grp.get("title", "").fillna(""),  # type: ignore
                    hovertemplate="<b>%{text}</b><br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
                )
            )

        # ------------------------------------------------------------------
        # CLUSTER OUTLINES (ellipse-like using Plotly path shapes)
        # ------------------------------------------------------------------
        shapes = []
        for cid, grp in df.groupby("cluster"):
            if len(grp) < 3:
                continue

            x = grp["embedding_x"].values
            y = grp["embedding_y"].values

            cx, cy = x.mean(), y.mean()  # type: ignore
            cov = np.cov(x, y)  # type: ignore

            # Eigendecomposition for principal axes
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            # 1-sigma ellipse radii
            width, height = 2 * ellipse_scale * np.sqrt(eigvals)
            angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

            # Parameterize ellipse and build path string
            theta = np.linspace(0, 2 * np.pi, 200)
            ex = (width / 2.0) * np.cos(theta)
            ey = (height / 2.0) * np.sin(theta)

            # Rotate and translate
            xr = cx + ex * np.cos(angle) - ey * np.sin(angle)
            yr = cy + ex * np.sin(angle) + ey * np.cos(angle)

            # Build SVG-like path for Plotly
            path_cmds = [f"M {xr[0]},{yr[0]}"] + [
                f"L {xr[i]},{yr[i]}" for i in range(1, len(xr))
            ]
            path_cmds.append("Z")
            path = " ".join(path_cmds)

            shapes.append(
                dict(
                    type="path",
                    path=path,
                    line=dict(
                        color="black",
                        width=4,        # thicker line
                        dash="dot",
                    ),
                    fillcolor="rgba(0,0,0,0)",
                    opacity=1,      # more visible
                )
            )

            # Optional centroid label (A, B, C...; fall back to cluster id if > 26)
            if label_clusters:
                if 0 <= cid < 26:  # type: ignore
                    label = chr(65 + cid)  # type: ignore
                else:
                    label = f"C{cid}"

                # make it bold and bigger
                label_html = f"<b>{label}</b>"

                fig.add_trace(
                    go.Scatter(
                        x=[cx],
                        y=[cy],
                        mode="text",
                        text=[label_html],
                        textfont=dict(size=30, color="black"),
                        showlegend=False,
                    )
                )

        fig.update_layout(shapes=shapes)

        fig.update_layout(
            width=1200,
            height=900,
            title="",
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="",
            yaxis_title="",
            showlegend=False
        )

        plots_class.save_plotly_figure(fig, f"cluster_tsne_research_{name_suffix}", width=1600, height=900, scale=3)
