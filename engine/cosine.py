"""
Cosine similarity — two variants.
Lifted and extended from Assignment2_similarity_analysis.py (Tomer Blond).

Cosine(A, B) = (A·B) / (||A|| * ||B||)

Variant 1 — Numeric features:
    [rating_z, votes_z, start_year_z, popularity_z]
    (pre-computed z-scores stored in catalog)

Variant 2 — Text embeddings:
    384-dim multilingual sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2)
    pre-computed and L2-normalized → dot product = cosine similarity
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


# ── Numeric Cosine ────────────────────────────────────────────────────────────

NUMERIC_COLS = ["rating_z", "votes_z", "start_year_z", "popularity_z"]


def _numeric_matrix(catalog: pd.DataFrame) -> np.ndarray:
    X = catalog[NUMERIC_COLS].fillna(0).values.astype(np.float32)
    return sk_cosine(X).astype(np.float32)


def top_k_cosine_numeric(
    query_title: str,
    catalog: pd.DataFrame,
    numeric_matrix: np.ndarray,
    k: int = 50,
    exclude_titles: set = None,
) -> pd.DataFrame:
    exclude_titles = exclude_titles or set()
    mask = catalog["title"].str.lower() == query_title.strip().lower()
    if not mask.any():
        mask = catalog["title"].str.lower().str.contains(
            query_title.strip().lower(), regex=False, na=False)
    if not mask.any():
        return catalog.head(0)

    q_idx = catalog[mask].index[0]
    row_pos = catalog.index.get_loc(q_idx)

    scores_vec = numeric_matrix[row_pos].copy()
    scores_vec[row_pos] = -1

    top_pos = np.argsort(-scores_vec)[:k + len(exclude_titles) + 1]
    result_rows, result_scores = [], []
    for pos in top_pos:
        title = catalog.iloc[pos]["title"]
        if pos == row_pos or title in exclude_titles:
            continue
        result_rows.append(catalog.iloc[pos])
        result_scores.append(float(scores_vec[pos]))
        if len(result_rows) == k:
            break

    if not result_rows:
        return catalog.head(0)

    result = pd.DataFrame(result_rows).reset_index(drop=True)
    result["cosine_numeric_score"] = result_scores
    return result


# ── Text Embedding Cosine ──────────────────────────────────────────────────────

def top_k_cosine_text(
    query_embedding: np.ndarray,
    catalog: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 50,
    exclude_titles: set = None,
    query_title: str = None,
) -> pd.DataFrame:
    """
    query_embedding: 1-D L2-normalized 384-dim vector (already normalized)
    embeddings: (N, 384) L2-normalized matrix
    """
    exclude_titles = exclude_titles or set()

    # dot product = cosine since both are L2-normalized
    scores_vec = embeddings @ query_embedding.astype(np.float32)

    # optionally exclude the seed title itself
    if query_title:
        seed_mask = catalog["title"].str.lower() == query_title.strip().lower()
        if seed_mask.any():
            seed_pos = catalog[seed_mask].index[0]
            scores_vec[catalog.index.get_loc(seed_pos)] = -1

    top_pos = np.argsort(-scores_vec)[:k + len(exclude_titles) + 1]
    result_rows, result_scores = [], []
    for pos in top_pos:
        title = catalog.iloc[pos]["title"]
        if title in exclude_titles:
            continue
        result_rows.append(catalog.iloc[pos])
        result_scores.append(float(scores_vec[pos]))
        if len(result_rows) == k:
            break

    if not result_rows:
        return catalog.head(0)

    result = pd.DataFrame(result_rows).reset_index(drop=True)
    result["cosine_text_score"] = result_scores
    return result


def build_numeric_matrix(catalog: pd.DataFrame) -> np.ndarray:
    return _numeric_matrix(catalog)
