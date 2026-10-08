"""Audit of the rule-based theme labels: text field, language and multilingual-lexicon sensitivity."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from asmr.analysis.supplementary.shared import (
    SEED,
    TEXT_SOURCE,
    THEME_NAMES,
    THEMES,
    pre,
    save_table,
    theme_flags_for_text,
)

MULTILINGUAL_LEXICON = {
    "has_whisper": [
        r"susurr", r"sussurr", r"chuchot", r"fl[uü]ster", r"fluister", r"속삭", r"ささやき", r"囁",
        r"ウィスパー", r"шёпот", r"шепот", r"шепч", r"шёпотом", r"bisik", r"fısıl", r"fisil", r"thì thầm",
        r"耳语", r"低语", r"กระซิบ", r"szept", r"همس",
    ],
    "has_sleep": [
        r"dormir", r"sueño", r"insomnio", r"\bsono\b", r"insônia", r"insonia", r"sommeil",
        r"endormi", r"insomnie", r"schlaf", r"einschlaf", r"sonno", r"addorment", r"수면", r"꿀잠",
        r"잠자", r"잠들", r"불면", r"睡眠", r"眠", r"ねむ", r"寝", r"засып", r"спать", r"бессонниц",
        r"для сна", r"tidur", r"uyku", r"ngủ", r"助眠", r"入睡", r"睡觉", r"นอน", r"หลับ",
        r"slaap", r"slapen", r"inslapen", r"zasyp", r"spać", r"bezsenn", r"نوم",
    ],
}


def measurement_audit(df: pd.DataFrame) -> None:
    titles = df["title"].fillna("").astype(str)
    title_flags = theme_flags_for_text(df, titles)

    rows = []
    for col in THEMES:
        full, ttl = df[col], title_flags[col]
        both = int((full & ttl).sum())
        rows.append({
            "theme": THEME_NAMES[col],
            "n_title_or_description": int(full.sum()),
            "pct_title_or_description": 100 * full.mean(),
            "n_title_only_match": int(ttl.sum()),
            "pct_title_only_match": 100 * ttl.mean(),
            "share_of_flagged_with_title_match_pct": 100 * both / max(int(full.sum()), 1),
        })
    save_table(pd.DataFrame(rows), "theme_title_vs_full_text.csv")

    # English versus non-English detection rates
    is_en = df["language"].eq("English")
    rows = []
    for col in THEMES:
        rows.append({
            "theme": THEME_NAMES[col],
            "pct_english": 100 * df.loc[is_en, col].mean(),
            "pct_non_english": 100 * df.loc[~is_en, col].mean(),
        })
    any_theme = df[THEMES].any(axis=1)
    rows.append({
        "theme": "At least one theme",
        "pct_english": 100 * any_theme[is_en].mean(),
        "pct_non_english": 100 * any_theme[~is_en].mean(),
    })
    save_table(pd.DataFrame(rows), "theme_detection_english_vs_non_english.csv")

    # Lexicon sensitivity for whisper and sleep among non-English videos
    text = pre.get_text_series(df, text_source=TEXT_SOURCE).fillna("").astype(str).str.lower()
    top_langs = df.loc[~is_en, "language"].value_counts().head(10).index.tolist()
    rows = []
    for col, stems in MULTILINGUAL_LEXICON.items():
        regex = re.compile("|".join(stems), re.IGNORECASE)
        ext = text.map(lambda t: bool(regex.search(t)))
        combined = df[col] | ext
        for lang in top_langs + ["All non-English"]:
            mask = (~is_en) if lang == "All non-English" else df["language"].eq(lang)
            rows.append({
                "theme": THEME_NAMES[col],
                "language": lang,
                "n_videos": int(mask.sum()),
                "pct_english_rules": 100 * df.loc[mask, col].mean(),
                "pct_with_multilingual_lexicon": 100 * combined[mask].mean(),
            })
        df[col + "_ml"] = combined
    save_table(pd.DataFrame(rows), "theme_multilingual_lexicon_sensitivity.csv")

    # Does the growth ordering of whisper / sleep survive the extended lexicon?
    vpd = df["views_per_day"]
    rows = []
    for col in ("has_whisper", "has_sleep"):
        for label, flag in ((col, df[col]), (col + "_ml", df[col + "_ml"])):
            sel = flag & vpd.notna()
            rows.append({
                "theme": THEME_NAMES[col],
                "definition": "English rules" if label == col else "English rules + multilingual lexicon",
                "n": int(flag.sum()),
                "mean_views_per_day": vpd[sel].mean(),
                "median_views_per_day": vpd[sel].median(),
            })
    save_table(pd.DataFrame(rows), "theme_lexicon_growth_sensitivity.csv")

    # Annotation sample for manual validation (not labelled here)
    rng = np.random.default_rng(SEED)
    parts = []
    for col in THEMES:
        for flagged in (True, False):
            pool = df.index[df[col] == flagged].to_numpy()
            pick = rng.choice(pool, size=min(20, len(pool)), replace=False)
            part = df.loc[pick, ["video_id", "language", "title"]].copy()
            part["description_start"] = df.loc[pick, "description"].fillna("").astype(str).str.slice(0, 300)
            part["theme"] = THEME_NAMES[col]
            part["rule_flag"] = flagged
            part["human_label_theme_present"] = ""
            parts.append(part)
    sample = pd.concat(parts, ignore_index=True)
    sample["url"] = "https://www.youtube.com/watch?v=" + sample["video_id"]
    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    save_table(sample.drop(columns=["rule_flag"]).assign(_rule_flag_hidden_key=sample["rule_flag"]),
               "theme_validation_sample_to_annotate.csv")
