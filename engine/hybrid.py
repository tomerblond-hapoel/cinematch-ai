"""
Hybrid scorer.
Combines Jaccard, Cosine-numeric, and Cosine-text into one ranked list.

Score = α·Jaccard + β·Cosine_numeric + (1-α-β)·Cosine_text

Default weights tuned on a 10-pair hand-rated preference set:
  α = 0.35  (genre/era matching matters most)
  β = 0.30  (numeric profile: rating, popularity)
  γ = 0.35  (semantic synopsis similarity)
"""

import numpy as np
import pandas as pd
from engine.jaccard import top_k_jaccard
from engine.cosine  import top_k_cosine_numeric, top_k_cosine_text

ALPHA = 0.35   # Jaccard weight (genre/era)
BETA  = 0.30   # Cosine-numeric weight (rating/popularity profile)
# GAMMA = 1 - ALPHA - BETA = 0.35  → Cosine-text weight (semantic plot)

# ── Bias 2 mitigation (see report Section 8) ─────────────────────────────────
# The multilingual model performs better on English text; Hebrew embedding quality
# is lower. When the user queries in Hebrew we up-weight Jaccard (language-neutral,
# genre-set based) and down-weight text-cosine (semantic embedding).
HEBREW_WEIGHTS = {"alpha": 0.50, "beta": 0.30, "gamma": 0.20}


def recommend(
    query_title: str,
    catalog: pd.DataFrame,
    numeric_matrix: np.ndarray,
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_n: int = 5,
    alpha: float = ALPHA,
    beta: float = BETA,
    exclude_titles: set = None,
    query_lang: str = "en",
) -> pd.DataFrame:
    # ── Bias 2 mitigation: re-weight for Hebrew queries ─────────────────────
    if query_lang == "he":
        alpha = HEBREW_WEIGHTS["alpha"]
        beta  = HEBREW_WEIGHTS["beta"]
        # gamma derived below = 0.20
    """
    Returns top_n recommendations as a DataFrame with columns:
      title, genres, rating, votes, decade_str, overview, poster_path,
      jaccard_score, cosine_numeric_score, cosine_text_score, hybrid_score
    """
    gamma = 1.0 - alpha - beta
    exclude_titles = exclude_titles or set()

    # Get candidates from each method (union pool)
    j_res = top_k_jaccard(query_title, catalog, k=100, exclude_titles=exclude_titles)
    n_res = top_k_cosine_numeric(query_title, catalog, numeric_matrix, k=100,
                                 exclude_titles=exclude_titles)
    t_res = top_k_cosine_text(query_embedding, catalog, embeddings, k=100,
                              exclude_titles=exclude_titles, query_title=query_title)

    # Build a unified score table indexed by title
    scores: dict[str, dict] = {}

    def _add(df: pd.DataFrame, col: str):
        for _, row in df.iterrows():
            t = row["title"]
            if t not in scores:
                scores[t] = {
                    "jaccard_score": 0.0,
                    "cosine_numeric_score": 0.0,
                    "cosine_text_score": 0.0,
                    "_row": row,
                }
            scores[t][col] = float(row.get(col, 0.0) or 0.0)

    _add(j_res, "jaccard_score")
    _add(n_res, "cosine_numeric_score")
    _add(t_res, "cosine_text_score")

    # Compute hybrid
    for t in scores:
        d = scores[t]
        d["hybrid_score"] = (
            alpha * d["jaccard_score"]
            + beta  * d["cosine_numeric_score"]
            + gamma * d["cosine_text_score"]
        )

    if not scores:
        return pd.DataFrame()

    rows = []
    for t, d in sorted(scores.items(), key=lambda x: -x[1]["hybrid_score"]):
        r = d["_row"].copy()
        r["jaccard_score"]        = round(d["jaccard_score"], 4)
        r["cosine_numeric_score"] = round(d["cosine_numeric_score"], 4)
        r["cosine_text_score"]    = round(d["cosine_text_score"], 4)
        r["hybrid_score"]         = round(d["hybrid_score"], 4)
        rows.append(r)

    result = pd.DataFrame(rows).head(top_n).reset_index(drop=True)
    return result


def score_all_pairs(
    catalog: pd.DataFrame,
    numeric_matrix: np.ndarray,
    embeddings: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> np.ndarray:
    """Compute the full N×N hybrid matrix (used by anomaly + evaluate)."""
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos
    from engine.jaccard import batch_jaccard_matrix

    gamma = 1.0 - alpha - beta

    j_mat = batch_jaccard_matrix(catalog).astype(np.float32)
    n_mat = numeric_matrix.astype(np.float32)
    # embeddings are already L2-normalized → dot = cosine
    t_mat = (embeddings @ embeddings.T).astype(np.float32)

    return alpha * j_mat + beta * n_mat + gamma * t_mat
