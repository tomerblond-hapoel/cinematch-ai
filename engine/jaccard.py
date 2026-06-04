"""
Jaccard similarity filter.
Lifted and extended from Assignment2_similarity_analysis.py (Tomer Blond).

Jaccard(A, B) = |A ∩ B| / |A ∪ B|

Feature set per title: genres ∪ decade_bucket ∪ language_bucket
Returns the top-K most similar titles by Jaccard score.
"""

import numpy as np
import pandas as pd


def _build_feature_set(row: pd.Series) -> frozenset:
    items = set()
    # genres
    genres = str(row.get("genre_set_str", "") or "")
    for g in genres.split("|"):
        g = g.strip()
        if g:
            items.add(f"genre:{g}")
    # decade
    dec = row.get("decade_str", "")
    if dec and dec != "Unknown":
        items.add(f"decade:{dec}")
    # language family
    lang = str(row.get("language", "") or "").lower()[:2]
    if lang:
        items.add(f"lang:{lang}")
    return frozenset(items)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def top_k_jaccard(
    query_title: str,
    catalog: pd.DataFrame,
    k: int = 50,
    exclude_titles: set = None,
) -> pd.DataFrame:
    """Return top-K rows from catalog most similar to query_title by Jaccard."""
    exclude_titles = exclude_titles or set()
    mask = catalog["title"].str.lower() == query_title.strip().lower()
    if not mask.any():
        # soft match: contains
        mask = catalog["title"].str.lower().str.contains(
            query_title.strip().lower(), regex=False, na=False)
    if not mask.any():
        return catalog.head(0)

    query_row = catalog[mask].iloc[0]
    query_set = _build_feature_set(query_row)
    query_idx = query_row.name

    scores = []
    for idx, row in catalog.iterrows():
        if idx == query_idx:
            continue
        if row["title"] in exclude_titles:
            continue
        s = jaccard(query_set, _build_feature_set(row))
        scores.append((idx, s))

    scores.sort(key=lambda x: -x[1])
    top_idx = [i for i, _ in scores[:k]]
    result = catalog.loc[top_idx].copy()
    result["jaccard_score"] = [s for _, s in scores[:k]]
    return result.reset_index(drop=True)


def batch_jaccard_matrix(catalog: pd.DataFrame) -> np.ndarray:
    """Compute full N×N Jaccard matrix (used in evaluate.py)."""
    feature_sets = [_build_feature_set(row) for _, row in catalog.iterrows()]
    n = len(feature_sets)
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            s = jaccard(feature_sets[i], feature_sets[j])
            M[i, j] = s
            M[j, i] = s
    return M
