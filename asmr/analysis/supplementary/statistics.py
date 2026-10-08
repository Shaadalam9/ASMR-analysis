"""Concentration and diversity measures used in the supplementary analyses."""
import numpy as np


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def share_top(values: np.ndarray, frac: float) -> float:
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    x = x[np.isfinite(x)]
    k = max(int(np.ceil(frac * x.size)), 1)
    return float(x[:k].sum() / x.sum()) if x.sum() > 0 else float("nan")


def shannon(counts: np.ndarray, normalise: bool = True) -> float:
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.size == 0:
        return float("nan")
    p = c / c.sum()
    h = -(p * np.log(p)).sum()
    return float(h / np.log(len(counts))) if normalise and len(counts) > 1 else float(h)


def hhi(counts: np.ndarray) -> float:
    c = np.asarray(counts, dtype=float)
    p = c / c.sum()
    return float((p ** 2).sum())
