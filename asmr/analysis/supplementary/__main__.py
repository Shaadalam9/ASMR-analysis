"""Run the supplementary analyses.

Usage (after ``python -m asmr.analysis.publication`` has written the enriched dataset)::

    python -m asmr.analysis.supplementary                 # all stages
    python -m asmr.analysis.supplementary models cluster  # selected stages

Stages:
    measurement        theme-rule audit: title-only robustness, English versus non-English detection,
                       multilingual-lexicon sensitivity, annotation sample
    ecosystem          diversification, creator concentration, and attention inequality
    models             multivariable models of views per day and engagement
    models_title_only  the same models with themes defined from titles only
    cluster            cluster-number diagnostics and stability for k-means
    reversal           means versus medians and pooled versus within-channel duration figure (needs ``models``)
    robustness         stability of the duration associations across periods, languages and prolific channels
"""
import sys

from asmr.analysis.supplementary.cluster_diagnostics import cluster_diagnostics
from asmr.analysis.supplementary.ecosystem import ecosystem_structure
from asmr.analysis.supplementary.measurement import measurement_audit
from asmr.analysis.supplementary.models import models, models_title_only
from asmr.analysis.supplementary.reversal import reversal_figure
from asmr.analysis.supplementary.robustness import duration_robustness
from asmr.analysis.supplementary.shared import load_df, logger

STAGES = {
    "measurement": measurement_audit,
    "ecosystem": ecosystem_structure,
    "models": models,
    "models_title_only": models_title_only,
    "cluster": cluster_diagnostics,
    "reversal": reversal_figure,
    "robustness": duration_robustness,
}


def main(argv: list[str]) -> None:
    wanted = argv or list(STAGES)
    unknown = [name for name in wanted if name not in STAGES]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {', '.join(unknown)}. Choose from: {', '.join(STAGES)}")
    df = load_df()
    for name in wanted:
        logger.info(f"=== stage: {name} ===")
        STAGES[name](df)


if __name__ == "__main__":
    main(sys.argv[1:])
