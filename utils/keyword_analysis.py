from custom_logger import CustomLogger
import common
import pandas as pd
from typing import Any, Dict, Optional, Set
import os
from collections import Counter

from utils.tool import Tools  # for normalize_lemma_form, stopwords, etc.
from utils.preprocessing import Preprocessing
from utils.viz_core import Plots

logger = CustomLogger(__name__)

tool_class = Tools()
pre_process_class = Preprocessing()
plots_class = Plots()


class Keyword_analysis():
    def __init__(self) -> None:
        pass

    def compute_spacy_keyword_counts(self, data: Dict[str, Any], target_lemmas: Optional[Set[str]] = None,
                                     text_source: str = "both", model_name: str = "en_core_web_sm", top_k: int = 30,
                                     extra_stopwords: Optional[Set[str]] = None,
                                     allowed_pos: Optional[Set[str]] = None) -> pd.DataFrame:
        """
        Count lemmas across videos using spaCy.

        - Each lemma is counted at most once per video.
        - Lemmas are normalised via Tools.normalize_lemma_form so that
          variants (e.g. "whispers", "whispering") collapse to one key.
        - If allowed_pos is provided (e.g. {"VERB"}), only those parts of
          speech are considered.
        """
        nlp = pre_process_class.get_spacy_nlp(model_name)
        if nlp is None:
            return pd.DataFrame(columns=["lemma", "count"])

        # Build per-video texts
        texts: list[str] = []
        for _, info in data.items():
            title = info.get("title") or ""
            description = info.get("description") or ""

            if text_source == "title":
                txt = title
            elif text_source == "description":
                txt = description
            elif text_source == "both":
                txt = f"{title}\n{description}"
            else:
                raise ValueError(f"Unsupported text_source: {text_source!r}")

            texts.append(tool_class.clean_text(txt))

        mode = "explicit" if target_lemmas else "auto-topk"
        allowed_pos_str = ", ".join(sorted(allowed_pos)) if allowed_pos else "None"
        logger.info(
            f"Running spaCy over {len(texts)} videos for lemma counts "
            f"(text_source={text_source}, mode={mode}, allowed_pos={allowed_pos_str})"
        )

        # Normalise target lemma set once
        if target_lemmas is not None:
            target_lemmas = {
                tool_class.normalize_lemma_form(w.lower())
                for w in target_lemmas
            }

        # Lowercased extra stopwords
        if extra_stopwords is None:
            extra_stopwords_lc: Set[str] = set()
        else:
            extra_stopwords_lc = {w.lower() for w in extra_stopwords}

        counts: Counter = Counter()

        for doc in nlp.pipe(texts, batch_size=256):
            lemma_set: Set[str] = set()

            for token in doc:
                if not token.is_alpha:
                    continue
                if token.is_stop:
                    continue
                if allowed_pos is not None and token.pos_ not in allowed_pos:
                    continue

                raw_lemma = token.lemma_.lower()
                lemma = tool_class.normalize_lemma_form(raw_lemma)

                if lemma in extra_stopwords_lc:
                    continue

                lemma_set.add(lemma)

            if not lemma_set:
                continue

            if target_lemmas is not None:
                for lemma in lemma_set.intersection(target_lemmas):
                    counts[lemma] += 1
            else:
                for lemma in lemma_set:
                    counts[lemma] += 1

        if not counts:
            logger.warning("No lemmas counted; returning empty keyword table.")
            return pd.DataFrame(columns=["lemma", "count"])

        df = pd.DataFrame(
            {"lemma": list(counts.keys()), "count": list(counts.values())}
        ).sort_values("count", ascending=False)

        if top_k is not None:
            df = df.head(top_k)

        logger.info(
            "Top spaCy keyword lemmas (first 10 rows):\n"
            f"{df.head(10).to_string(index=False)}"
        )
        logger.info(
            f"Total unique lemmas in table: {len(df)}, "
            f"min count={df['count'].min()}, "
            f"median count={df['count'].median():.1f}, "
            f"max count={df['count'].max()}"
        )

        return df

    def compute_lemma_trend_over_time(self, df: pd.DataFrame, lemma_name: str, lemma_targets: Set[str],
                                      text_source: str = "both", model_name: str = "en_core_web_sm") -> pd.DataFrame:
        """
        Number of videos per year containing any of the given lemmas.

        Lemmas are normalised via Tools.normalize_lemma_form so that
        variants (e.g. "relaxation", "relaxing") all count as one.
        """
        nlp = pre_process_class.get_spacy_nlp(model_name)
        if nlp is None:
            logger.warning("spaCy not available; lemma trend not computed.")
            return pd.DataFrame(columns=["upload_year", "theme_count", "total_videos"])

        df_tmp = df.dropna(subset=["upload_year"]).copy()
        df_tmp["upload_year"] = df_tmp["upload_year"].astype(int)

        texts = pre_process_class.get_text_series(df_tmp, text_source=text_source).tolist()
        years = df_tmp["upload_year"].tolist()

        # Normalise target lemmas once
        lemma_targets_norm = {
            tool_class.normalize_lemma_form(le.lower())
            for le in lemma_targets
        }

        records: list[tuple[int, bool]] = []

        for year, doc in zip(years, nlp.pipe(texts, batch_size=256)):
            lemma_set = {
                tool_class.normalize_lemma_form(tok.lemma_.lower())
                for tok in doc
                if tok.is_alpha and not tok.is_stop
            }
            has_lemma = bool(lemma_set & lemma_targets_norm)
            records.append((year, has_lemma))

        if not records:
            return pd.DataFrame(columns=["upload_year", "theme_count", "total_videos"])

        tmp = pd.DataFrame(records, columns=["upload_year", "has_lemma"])

        grouped = tmp.groupby("upload_year")["has_lemma"]
        trend = (
            grouped.agg(
                theme_count=lambda s: int(s.sum()),
                total_videos="count",
            )
            .reset_index()
        )

        if not trend.empty:
            logger.info(
                f"Lemma trend for '{lemma_name}': {len(trend)} years, "
                f"total videos with lemma={int(trend['theme_count'].sum())}"
            )
            logger.info(
                "Lemma trend table:\n"
                f"{trend.to_string(index=False)}"
            )

        return trend

    def run_verb_lemma_wordcloud_pipeline(self, data: Dict[str, Any], text_source: str = "both",
                                          model_name: str = "en_core_web_sm", top_k: int = 200) -> None:
        """
        Build a wordcloud of lemmatised words, restricted to verbs.

        - Uses spaCy lemmas.
        - Each lemma is counted at most once per video.
        - Variants are collapsed via Tools.normalize_lemma_form.
        - Result is cached in a PKL file for fast subsequent runs.
        """
        analysis_dir = os.path.join(common.output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        verb_pickle = os.path.join(
            analysis_dir,
            f"spacy_keywords_verbs_{text_source}.pkl",
        )

        if os.path.isfile(verb_pickle):
            logger.info(f"Loading verb spaCy keyword counts from {verb_pickle}")
            keyword_df = pd.read_pickle(verb_pickle)
        else:
            logger.info(
                "No verb spaCy keyword pickle found; computing verb-only keyword counts..."
            )
            extra_stopwords = tool_class.get_custom_stopwords()
            keyword_df = self.compute_spacy_keyword_counts(
                data=data,
                target_lemmas=None,
                text_source=text_source,
                model_name=model_name,
                top_k=top_k,
                extra_stopwords=extra_stopwords,
                allowed_pos={"VERB"},
            )
            logger.info(f"Saving verb spaCy keyword counts to pickle {verb_pickle}")
            keyword_df.to_pickle(verb_pickle)

        if keyword_df.empty:
            logger.warning("No verb lemmas found; skipping verb wordcloud.")
            return

        frequencies = dict(zip(keyword_df["lemma"], keyword_df["count"]))

        img = plots_class.generate_wordcloud_from_frequencies(
            frequencies=frequencies,
            stopwords=tool_class.get_custom_stopwords(),
        )

        fig = plots_class.create_plotly_figure(
            img,
            title="",
        )
        plots_class.save_plotly_figure(
            fig,
            filename=f"wordcloud_verbs_{text_source}",
        )
