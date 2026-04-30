import pandas as pd
import numpy as np
import os
from custom_logger import CustomLogger
import common
from typing import Any, Dict, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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

    def _sample_positions_by_cluster(self, labels: pd.Series, max_samples: int,
                                     random_state: int = 42) -> np.ndarray:
        """
        Return integer row positions for a stratified sample by cluster.

        The full dataset is still clustered, but expensive visual embeddings can be
        computed on a representative subset when the corpus is large.
        """
        n_rows = len(labels)
        if max_samples is None or max_samples <= 0 or n_rows <= max_samples:
            return np.arange(n_rows)

        tmp = pd.DataFrame(
            {
                "cluster": labels.to_numpy(),
                "position": np.arange(n_rows),
            }
        )
        counts = tmp["cluster"].value_counts().sort_index()
        raw_alloc = counts / counts.sum() * max_samples
        allocation = np.floor(raw_alloc).astype(int).clip(lower=1)
        allocation = np.minimum(allocation, counts)

        remaining = int(max_samples - allocation.sum())
        if remaining > 0:
            residual_order = (raw_alloc - allocation).sort_values(ascending=False).index
            while remaining > 0:
                changed = False
                for cluster_id in residual_order:
                    if allocation.loc[cluster_id] < counts.loc[cluster_id]:
                        allocation.loc[cluster_id] += 1
                        remaining -= 1
                        changed = True
                        if remaining == 0:
                            break
                if not changed:
                    break
        elif remaining < 0:
            removable_order = allocation.sort_values(ascending=False).index
            while remaining < 0:
                changed = False
                for cluster_id in removable_order:
                    if allocation.loc[cluster_id] > 1:
                        allocation.loc[cluster_id] -= 1
                        remaining += 1
                        changed = True
                        if remaining == 0:
                            break
                if not changed:
                    break

        sampled_positions = []
        for cluster_id, group in tmp.groupby("cluster"):
            n_take = int(allocation.loc[cluster_id])
            sampled = group.sample(n=n_take, random_state=random_state)
            sampled_positions.extend(sampled["position"].tolist())

        sampled_positions = np.array(sorted(sampled_positions), dtype=int)
        logger.info(
            f"Using {len(sampled_positions)} / {n_rows} videos for t-SNE embedding "
            "with stratified sampling by cluster."
        )
        return sampled_positions

    def cluster_videos_tsne(self, df: pd.DataFrame, n_clusters: int = 11, random_state: int = 42,
                            text_source: str = "both", tsne_perplexity: float = 30.0,
                            tsne_learning_rate: float = 200.0, tsne_n_iter: int = 1000,
                            tsne_max_samples: int = 15000, svd_components: int = 50
                            ) -> Tuple[pd.DataFrame, Optional[Pipeline]]:
        """
        Cluster all videos, then compute a t-SNE layout for a representative subset.

        t-SNE and interactive browser plots do not scale well to tens of thousands
        of points. For large corpora, KMeans labels are still assigned to every
        video, while embedding_x / embedding_y are filled only for the sampled rows.
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

        df_copy["embedding_x"] = np.nan
        df_copy["embedding_y"] = np.nan

        try:
            logger.info("Computing 2D t-SNE embedding for cluster visualization...")
            features = pipeline.named_steps["preprocess"].transform(X)

            sample_positions = self._sample_positions_by_cluster(
                df_copy["cluster"],
                max_samples=tsne_max_samples,
                random_state=random_state,
            )
            features_for_tsne = features[sample_positions]

            n_features = features_for_tsne.shape[1]
            n_samples = features_for_tsne.shape[0]
            max_components = min(int(svd_components), n_features - 1, n_samples - 1)

            if max_components >= 2:
                svd = TruncatedSVD(
                    n_components=max_components,
                    random_state=random_state,
                )
                features_reduced = svd.fit_transform(features_for_tsne)
                logger.info(
                    f"Reduced sampled feature space to {max_components} dimensions "
                    "via TruncatedSVD before t-SNE."
                )
            else:
                features_reduced = (
                    features_for_tsne.toarray()
                    if hasattr(features_for_tsne, "toarray")
                    else np.array(features_for_tsne)
                )
                logger.info(
                    "Skipped SVD reduction before t-SNE because the sampled dataset is very small."
                )

            tsne_kwargs = dict(
                n_components=2,
                perplexity=min(tsne_perplexity, max(1.0, (n_samples - 1) / 3.0)),
                learning_rate=tsne_learning_rate,
                random_state=random_state,
                init="random",
            )
            try:
                tsne = TSNE(**tsne_kwargs, max_iter=tsne_n_iter)
            except TypeError:
                tsne = TSNE(**tsne_kwargs, n_iter=tsne_n_iter)

            embedding_2d = tsne.fit_transform(features_reduced)

            df_copy.iloc[sample_positions, df_copy.columns.get_loc("embedding_x")] = embedding_2d[:, 0]
            df_copy.iloc[sample_positions, df_copy.columns.get_loc("embedding_y")] = embedding_2d[:, 1]

            logger.info(
                "Completed t-SNE embedding with "
                f"{embedding_2d.shape[0]} sampled points. "
                "Rows without sampled embeddings remain NaN."
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

    def cluster_videos_umap(self, df: pd.DataFrame, n_clusters: int = 11, random_state: int = 42,
                            text_source: str = "both", umap_n_neighbors: int = 30,
                            umap_min_dist: float = 0.1, umap_metric: str = "cosine",
                            umap_n_epochs: Optional[int] = None,
                            svd_components: int = 50
                            ) -> Tuple[pd.DataFrame, Optional[Pipeline]]:
        """
        Cluster all videos, then compute a UMAP layout for every row.

        This is intended for larger corpora where t-SNE is too slow or where a
        complete 2D embedding is needed. KMeans labels and UMAP coordinates are
        both assigned to the full dataset. The high-dimensional TF-IDF feature
        matrix is first reduced with TruncatedSVD to keep UMAP fast and memory
        friendly.
        """
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "UMAP support requires the 'umap-learn' package. Install it with: "
                "pip install umap-learn"
            ) from exc

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
            f"Fitting clustering model with UMAP embedding on {len(df_copy)} videos "
            f"(text_source={text_source}, n_clusters={n_clusters})"
        )
        pipeline.fit(X)

        logger.info("Assigning cluster labels (UMAP version)...")
        df_copy["cluster"] = pipeline.predict(X)

        df_copy["embedding_x"] = np.nan
        df_copy["embedding_y"] = np.nan

        try:
            logger.info("Computing 2D UMAP embedding for all videos...")
            features = pipeline.named_steps["preprocess"].transform(X)

            n_features = features.shape[1]
            n_samples = features.shape[0]
            max_components = min(int(svd_components), n_features - 1, n_samples - 1)

            if max_components >= 2:
                svd = TruncatedSVD(
                    n_components=max_components,
                    random_state=random_state,
                )
                features_reduced = svd.fit_transform(features)
                logger.info(
                    f"Reduced full feature space to {max_components} dimensions "
                    "via TruncatedSVD before UMAP."
                )
            else:
                features_reduced = (
                    features.toarray()
                    if hasattr(features, "toarray")
                    else np.array(features)
                )
                logger.info(
                    "Skipped SVD reduction before UMAP because the dataset is very small."
                )

            safe_n_neighbors = min(int(umap_n_neighbors), max(2, n_samples - 1))
            reducer_kwargs = dict(
                n_components=2,
                n_neighbors=safe_n_neighbors,
                min_dist=float(umap_min_dist),
                metric=umap_metric,
                random_state=random_state,
                low_memory=True,
            )
            if umap_n_epochs is not None:
                reducer_kwargs["n_epochs"] = int(umap_n_epochs)

            reducer = umap.UMAP(**reducer_kwargs)
            embedding_2d = reducer.fit_transform(features_reduced)

            df_copy["embedding_x"] = embedding_2d[:, 0]
            df_copy["embedding_y"] = embedding_2d[:, 1]

            logger.info(
                "Completed UMAP embedding with "
                f"{embedding_2d.shape[0]} points. Every row has UMAP coordinates."
            )

            if "cluster" in df_copy.columns:
                cluster_counts = df_copy["cluster"].value_counts().sort_index()
                logger.info(
                    "Cluster sizes (UMAP embedding version):\n"
                    f"{cluster_counts.to_string()}"
                )

        except Exception as exc:
            logger.warning(f"Could not compute 2D UMAP embedding for clusters: {exc}")
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
                                  highlight_cluster: Optional[int] = None,
                                  max_scatter_points: int = 20000) -> None:
        """Visualize clusters: bar charts + a sampled 2D scatter with circles around clusters."""
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
        if max_scatter_points and len(df_emb) > max_scatter_points:
            sample_positions = self._sample_positions_by_cluster(
                df_emb["cluster"].reset_index(drop=True),
                max_samples=max_scatter_points,
                random_state=42,
            )
            logger.info(
                f"Sampling cluster scatter from {len(df_emb)} to {len(sample_positions)} points "
                "to keep the Plotly output responsive."
            )
            df_emb = df_emb.iloc[sample_positions].copy()

        df_emb["cluster_str"] = df_emb["cluster"].astype(str)
        cluster_order = sorted(df_emb["cluster_str"].unique())
        color_discrete_map = plots_class._get_cluster_color_map(df_emb["cluster"].values)  # type: ignore

        hover_cols = [
            c for c in ["video_id", "language", "views", "duration_minutes"]
            if c in df_emb.columns
        ]
        fig = px.scatter(
            df_emb,
            x="embedding_x",
            y="embedding_y",
            color="cluster_str",
            hover_data=hover_cols,
            title="",
            labels={
                "embedding_x": "",
                "embedding_y": "",
                "cluster_str": "",
            },
            category_orders={"cluster_str": cluster_order},
            color_discrete_map=color_discrete_map,
            render_mode="webgl",
        )
        fig.update_traces(marker=dict(size=4, opacity=0.35))

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

        plots_class.save_plotly_figure(
            fig,
            f"cluster_scatter_embedding{suffix}",
            width=1600,
            height=900,
            scale=SCALE,
            save_eps=False,
            auto_open=True,
        )
        logger.info(f"2D cluster scatter saved as cluster_scatter_embedding{suffix}.html/.png")

    def plot_embedding_static_full(self, df: pd.DataFrame, name_suffix: str = "umap_full",
                                   embedding_name: str = "UMAP",
                                   filename_prefix: str = "cluster_umap_full_static") -> None:
        """Save a full-dataset static 2D cluster plot as PNG and EPS.

        This uses every row with embedding_x / embedding_y. It is intentionally
        static because very large interactive HTML files can fail to open in the
        browser even when the underlying embedding was computed correctly.
        """
        required = {"cluster", "embedding_x", "embedding_y"}
        if required - set(df.columns):
            logger.warning(f"{embedding_name} full static plot: missing cluster or embedding columns.")
            return

        df_plot = df.dropna(subset=["embedding_x", "embedding_y"]).copy()
        if df_plot.empty:
            logger.warning(f"{embedding_name} full static plot: no non-null embeddings.")
            return

        df_plot["cluster"] = df_plot["cluster"].astype(int)
        unique_clusters = sorted(df_plot["cluster"].unique())
        color_map = plots_class._get_cluster_color_map(df_plot["cluster"].values)  # type: ignore

        output_final = os.path.join(common.root_dir, "figures")
        os.makedirs(common.output_dir, exist_ok=True)
        os.makedirs(output_final, exist_ok=True)

        filename = f"{filename_prefix}_{name_suffix}"
        output_png = os.path.join(common.output_dir, filename + ".png")
        final_png = os.path.join(output_final, filename + ".png")
        output_eps = os.path.join(common.output_dir, filename + ".eps")
        final_eps = os.path.join(output_final, filename + ".eps")

        fig, ax = plt.subplots(figsize=(14, 10), dpi=180)
        for cluster_id in unique_clusters:
            grp = df_plot[df_plot["cluster"] == cluster_id]
            ax.scatter(
                grp["embedding_x"],
                grp["embedding_y"],
                s=3,
                alpha=0.35,
                label=f"Cluster {cluster_id}",
                c=color_map[str(cluster_id)],
                linewidths=0,
            )

        for cluster_id, grp in df_plot.groupby("cluster"):
            ax.text(
                grp["embedding_x"].mean(),
                grp["embedding_y"].mean(),
                str(cluster_id),
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="center",
            )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("")
        ax.legend(markerscale=4, frameon=False, fontsize=9, loc="best")
        fig.tight_layout()

        fig.savefig(output_png, bbox_inches="tight")
        fig.savefig(output_eps, bbox_inches="tight")
        plt.close(fig)

        try:
            import shutil
            shutil.copy(output_png, final_png)
            shutil.copy(output_eps, final_eps)
        except Exception as exc:
            logger.warning(f"Could not copy full static embedding plot to figures directory: {exc}")

        logger.info(
            f"Full {embedding_name} static cluster plot saved to {output_png} "
            f"and {final_png} with {len(df_plot)} points."
        )

    def plot_tsne_research(self, df: pd.DataFrame, name_suffix: str = "tsne_research", label_clusters: bool = True,
                           ellipse_scale: float = 1.2, max_points: Optional[int] = 12000,
                           random_state: int = 42, embedding_name: str = "t-SNE",
                           filename_prefix: str = "cluster_tsne_research"):
        """
        Produce a research-style 2D embedding plot with:
        - stable color map
        - ellipse-like outlines around clusters (via Plotly path shapes)
        - labeled cluster centroids
        - extensive logger output to interpret clusters
        """

        # Require necessary columns
        if {"cluster", "embedding_x", "embedding_y"} - set(df.columns):
            logger.warning(f"{embedding_name} research plot: missing embedding_x/y or cluster.")
            return

        df = df.dropna(subset=["embedding_x", "embedding_y"]).copy()
        if df.empty:
            logger.warning(f"{embedding_name} research plot: no non-null embeddings.")
            return

        df["cluster"] = df["cluster"].astype(int)
        df["cluster_str"] = df["cluster"].astype(str)

        if max_points and len(df) > max_points:
            sample_positions = self._sample_positions_by_cluster(
                df["cluster"].reset_index(drop=True),
                max_samples=max_points,
                random_state=random_state,
            )
            logger.info(
                f"Sampling {embedding_name} research plot from {len(df)} to {len(sample_positions)} points "
                "to keep the HTML file responsive."
            )
            df = df.iloc[sample_positions].copy()
            df["cluster_str"] = df["cluster"].astype(str)

        # ------------------------------------------------------------------
        # LOGGING
        # ------------------------------------------------------------------
        logger.info(f"===== {embedding_name} RESEARCH PLOT SUMMARY =====")
        logger.info(f"Total points in {embedding_name}: {len(df)}")

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
                go.Scattergl(
                    x=grp["embedding_x"],
                    y=grp["embedding_y"],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=color_map[str(cid)],
                        opacity=0.35,
                    ),
                    name=f"Cluster {cid}",
                    hovertemplate=(
                        f"Cluster {cid}<br>"
                        "x=%{x:.2f}<br>"
                        "y=%{y:.2f}<extra></extra>"
                    ),
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

        plots_class.save_plotly_figure(
            fig,
            f"{filename_prefix}_{name_suffix}",
            width=1600,
            height=900,
            scale=3,
            save_eps=False,
            auto_open=True,
        )

    def plot_umap_research(self, df: pd.DataFrame, name_suffix: str = "umap_research",
                           label_clusters: bool = True, ellipse_scale: float = 1.2,
                           max_points: Optional[int] = None, random_state: int = 42):
        """Produce a research-style UMAP plot using all available embedded rows by default."""
        return self.plot_tsne_research(
            df=df,
            name_suffix=name_suffix,
            label_clusters=label_clusters,
            ellipse_scale=ellipse_scale,
            max_points=max_points,
            random_state=random_state,
            embedding_name="UMAP",
            filename_prefix="cluster_umap_research",
        )

