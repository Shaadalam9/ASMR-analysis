from wordcloud import WordCloud
from typing import Set, Dict, Any
import plotly as py
import plotly.express as px
from custom_logger import CustomLogger
import common
import os
import warnings
import shutil
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import webbrowser


logger = CustomLogger(__name__)

# Default scaling factor for saved PNG images.
SCALE = 3

font_family = common.get_configs("font_family")
font_size = common.get_configs("font_size")


class Plots():
    def __init__(self) -> None:
        pass

    def generate_wordcloud_image(self, text: str, stopwords: Set[str]):
        """Generate a word cloud image array from raw text."""
        wordcloud = WordCloud(
            width=1000,
            height=600,
            background_color="white",
            stopwords=stopwords,
            collocations=False,
        ).generate(text)
        img = wordcloud.to_array()
        logger.info(
            f"Generated word cloud with {len(wordcloud.words_)} unique words."
        )
        return img

    def generate_wordcloud_from_frequencies(self, frequencies: Dict[str, int], stopwords: Set[str]):
        """
        Generate a word cloud image array from a lemma frequency dict.

        Keys are lemmas (already normalised), values are video counts.
        """
        wordcloud = WordCloud(
            width=1000,
            height=600,
            background_color="white",
            stopwords=stopwords,
            collocations=False,
        ).generate_from_frequencies(frequencies)
        img = wordcloud.to_array()
        logger.info(
            f"Generated word cloud from frequencies with "
            f"{len(wordcloud.words_)} unique lemmas."
        )
        return img

    def create_plotly_figure(self, img, title: str = "") -> Any:
        """Create a Plotly figure to display the word cloud image."""
        fig = px.imshow(img)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(
            title=title,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return fig

    def save_plotly_figure(self, fig: Any, filename: str, width: int = 1600, height: int = 900,
                           scale: int = SCALE, save_final: bool = True, save_png: bool = True,
                           save_eps: bool = True, auto_open: bool = True) -> None:
        """Save a Plotly figure as HTML, PNG, and EPS formats."""
        auto_open = bool(auto_open and common.get_configs("auto_open_plots"))
        output_final = os.path.join(common.root_dir, "figures")
        os.makedirs(common.output_dir, exist_ok=True)
        os.makedirs(output_final, exist_ok=True)

        fig.update_layout(
            template=common.get_configs("plotly_template"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(
                family=font_family,
                size=font_size,
            ),
        )

        fig.update_xaxes(
            tickfont=dict(
                family=font_family,
                size=font_size,
            ),
            title_font=dict(
                family=font_family,
                size=font_size,
            ),
        )
        fig.update_yaxes(
            tickfont=dict(
                family=font_family,
                size=font_size,
            ),
            title_font=dict(
                family=font_family,
                size=font_size,
            ),
        )

        html_path = os.path.join(common.output_dir, filename + ".html")
        py.offline.plot(
            fig,
            filename=html_path,
            auto_open=auto_open,
        )

        if save_final:
            final_html_path = os.path.join(output_final, filename + ".html")
            py.offline.plot(
                fig,
                filename=final_html_path,
                auto_open=False,
            )

        try:
            if save_png:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*Support for Kaleido versions less than 1.0.0.*",
                        category=DeprecationWarning,
                    )
                    png_path = os.path.join(common.output_dir, filename + ".png")
                    fig.write_image(
                        png_path,
                        width=width,
                        height=height,
                        scale=scale,
                    )
                if save_final:
                    final_png_path = os.path.join(output_final, filename + ".png")
                    shutil.copy(png_path, final_png_path)

            if save_eps:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    eps_path = os.path.join(common.output_dir, filename + ".eps")
                    fig.write_image(
                        eps_path,
                        width=width,
                        height=height,
                    )
                if save_final:
                    final_eps_path = os.path.join(output_final, filename + ".eps")
                    shutil.copy(eps_path, final_eps_path)
        except Exception as exc:
            logger.error(
                f"Error raised when attempting to save image {filename}: {exc}"
            )

    def _get_cluster_color_map(self, clusters: np.ndarray) -> Dict[str, str]:
        """
        Return a stable mapping from cluster id (as string) to a fixed color.
        This ensures that cluster '0' is always the same color, '1' is another, etc.
        """
        unique_clusters = sorted(int(c) for c in np.unique(clusters))
        cluster_labels = [str(c) for c in unique_clusters]

        palette = px.colors.qualitative.Plotly

        color_map: Dict[str, str] = {}
        for i, label in enumerate(cluster_labels):
            color_map[label] = palette[i % len(palette)]

        return color_map

    def analyze_log_views_normality(self, df: pd.DataFrame) -> None:
        """
        Check whether log10(views) is approximately normally distributed.
        """
        views = pd.to_numeric(df["views"], errors="coerce")
        views = views[views > 0].dropna()

        if views.empty:
            logger.warning("No positive views available; cannot analyze normality.")
            return

        log_views = np.log10(views)

        logger.info("===== LOG10(VIEWS) DISTRIBUTION ANALYSIS =====")
        logger.info(f"N = {len(log_views)}")
        logger.info(f"Mean(log10(views)) = {log_views.mean():.3f}")
        logger.info(f"Std(log10(views))  = {log_views.std():.3f}")

        k2, p_normaltest = stats.normaltest(log_views)
        logger.info(
            f"normaltest: statistic = {k2:.3f}, p-value = {p_normaltest:.3g} "
            "(H0: data come from a normal distribution)"
        )

        sample = log_views
        max_n_shapiro = 5000
        if len(sample) > max_n_shapiro:
            sample = sample.sample(max_n_shapiro, random_state=42)  # type: ignore

        w_stat, p_shapiro = stats.shapiro(sample)
        logger.info(
            f"Shapiro–Wilk: W = {w_stat:.3f}, p-value = {p_shapiro:.3g} "
            "(H0: data come from a normal distribution)"
        )

        logger.info(
            "Interpretation: if p-values are << 0.05, log10(views) deviates from a "
            "perfect Gaussian; larger p-values mean you cannot reject normality."
        )

        try:
            (osm, osr), (slope, intercept, r) = stats.probplot(
                log_views, dist="norm", plot=None
            )

            qq_df = pd.DataFrame(
                {
                    "theoretical_quantiles": osm,
                    "ordered_log_views": osr,
                }
            )

            fig = px.scatter(
                qq_df,
                x="theoretical_quantiles",
                y="ordered_log_views",
                title="",
                labels={
                    "theoretical_quantiles": "Theoretical quantiles (Normal)",
                    "ordered_log_views": "Ordered log10(views)",
                },
                opacity=0.7,
            )

            x_min = qq_df["theoretical_quantiles"].min()
            x_max = qq_df["theoretical_quantiles"].max()
            line_x = np.array([x_min, x_max])
            line_y = slope * line_x + intercept  # type: ignore

            fig.add_scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(dash="dash"),
                showlegend=False,
            )

            self.save_plotly_figure(
                fig,
                filename="log_views_qq_plot",
                width=1600,
                height=900,
                scale=SCALE,
            )

            logger.info(
                "Q–Q plot for log10(views) saved as 'log_views_qq_plot.*'. "
                "Points close to the dashed line indicate approximate normality."
            )
        except Exception as exc:
            logger.warning(f"Could not create Q–Q plot for log10(views): {exc}")

        logger.info("===== END LOG10(VIEWS) DISTRIBUTION ANALYSIS =====")

    def plot_duration_vs_views(self, df: pd.DataFrame) -> None:
        """
        Hexbin plot: log–log duration (seconds) vs views for all videos.
        Uses Matplotlib and saves HTML, PNG, and EPS. The HTML is opened
        in the default browser.
        """
        df_plot = df[["duration_seconds", "views"]].copy()
        df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna()
        df_plot = df_plot[
            (df_plot["duration_seconds"] > 0) &
            (df_plot["views"] > 0)
        ]

        if df_plot.empty:
            logger.warning("No data for duration vs views plot.")
            return

        logger.info(
            f"Duration vs views hexbin plot uses {len(df_plot)} videos with positive "
            "duration and views."
        )
        logger.info(
            "Duration (seconds) summary for plotted videos:\n"
            f"{df_plot['duration_seconds'].describe().to_string()}"
        )
        logger.info(
            "Views summary for plotted videos:\n"
            f"{df_plot['views'].describe().to_string()}"
        )

        # Matplotlib hexbin on log–log axes
        fig, ax = plt.subplots(figsize=(12, 8))

        hb = ax.hexbin(
            df_plot["duration_seconds"].to_numpy(),
            df_plot["views"].to_numpy(),
            gridsize=60,
            xscale="log",
            yscale="log",
            bins="log",
            mincnt=1,
        )
        # Colorbar: control size + font sizes
        cb = fig.colorbar(
            hb,
            ax=ax,
            shrink=0.9,   # < 1.0 = shorter, > 1.0 = longer
            aspect=30,    # larger = thinner bar, smaller = thicker bar
        )
        cb.set_label("")   # colorbar label size
        cb.ax.tick_params(labelsize=14)       # colorbar tick label size

        ax.set_xscale("log")
        ax.set_yscale("log")

        # Bigger axis labels
        ax.set_xlabel("Duration (seconds)", fontsize=16)
        ax.set_ylabel("Views", fontsize=16)

        # Bigger tick labels
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.tick_params(axis="both", which="minor", labelsize=12)

        ax.set_title("")

        fig.tight_layout()

        # Save files
        filename = "duration_vs_views"
        os.makedirs(common.output_dir, exist_ok=True)
        output_final = os.path.join(common.root_dir, "figures")
        os.makedirs(output_final, exist_ok=True)

        png_path = os.path.join(common.output_dir, f"{filename}.png")
        eps_path = os.path.join(common.output_dir, f"{filename}.eps")
        html_path = os.path.join(common.output_dir, f"{filename}.html")

        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        fig.savefig(eps_path, format="eps", dpi=300, bbox_inches="tight")

        final_png_path = os.path.join(output_final, f"{filename}.png")
        final_eps_path = os.path.join(output_final, f"{filename}.eps")
        shutil.copy(png_path, final_png_path)
        shutil.copy(eps_path, final_eps_path)

        # Simple HTML referencing the PNG (works from both output/ and figures/)
        rel_png_name = f"{filename}.png"
        html_content = f"""<!DOCTYPE html>
        <html>
        <head>
            <meta charset='utf-8'>
            <title>{filename}</title>
        </head>
        <body>
            <img src='{rel_png_name}' alt='{filename}' />
        </body>
        </html>
        """

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        final_html_path = os.path.join(output_final, f"{filename}.html")
        with open(final_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        plt.close(fig)

        # Try to auto-open the HTML file in the default browser
        if common.get_configs("auto_open_plots"):
            try:
                abs_html = os.path.abspath(html_path)
                webbrowser.open(f"file://{abs_html}", new=2)
                logger.info(f"Opened HTML for {filename} at {abs_html}")
            except Exception as exc:
                logger.warning(f"Could not auto-open HTML for {filename}: {exc}")
