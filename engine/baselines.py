"""
CineMatch AI — Baseline Models (for rubric Criterion 5: "compare 2 existing model options")

Baseline A — TF-IDF + Cosine on plot synopses
    Classical NLP, pre-LLM. Industry-standard baseline for content-based filters.

Baseline B — Popularity (rating × log(votes))
    Industry-standard sanity floor: "Top Rated" style recommender.

Baseline C — Random
    Lower-bound for Precision@5 — confirms our methods beat chance.

All three return a full N×N similarity matrix for fair comparison with
the Jaccard / Cosine-numeric / Cosine-text / Hybrid methods.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Baseline A: TF-IDF + Cosine on plot synopses ──────────────────────────────

def tfidf_matrix(catalog: pd.DataFrame) -> np.ndarray:
    """Classical NLP baseline: TF-IDF on plot synopses, then cosine."""
    texts = catalog["overview"].fillna("").astype(str).tolist()
    # Fallback to title for rows without a synopsis
    titles = catalog["title"].fillna("").astype(str).tolist()
    texts = [t if (t and len(t) > 20) else titles[i] for i, t in enumerate(texts)]
    vec = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vec.fit_transform(texts)
    return cosine_similarity(X).astype(np.float32)


# ── Baseline B: Popularity (rating × log votes) ────────────────────────────────

def popularity_matrix(catalog: pd.DataFrame) -> np.ndarray:
    """Every show recommends the most popular shows — same for everyone."""
    rating = pd.to_numeric(catalog["rating"], errors="coerce").fillna(0).values
    votes  = pd.to_numeric(catalog["votes"],  errors="coerce").fillna(0).values
    pop = rating * np.log1p(votes)
    pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)  # 0..1
    # Broadcast: row i's "similarity" to row j is just j's popularity score
    n = len(catalog)
    M = np.tile(pop.astype(np.float32), (n, 1))
    return M


# ── Baseline C: Random ─────────────────────────────────────────────────────────

def random_matrix(catalog: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Lower-bound sanity floor."""
    rng = np.random.default_rng(seed)
    n = len(catalog)
    M = rng.random((n, n)).astype(np.float32)
    # Make symmetric for fair comparison
    M = (M + M.T) / 2
    np.fill_diagonal(M, 1.0)
    return M


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_parquet(os.path.join(BASE, "data", "catalog.parquet"))
    df = df.nlargest(500, "votes").reset_index(drop=True)
    print(f"Baseline subset: {len(df)} rows")

    print("Building TF-IDF matrix...")
    A = tfidf_matrix(df)
    print(f"  Shape: {A.shape}, mean off-diagonal: {A[np.triu_indices_from(A, k=1)].mean():.4f}")

    print("Building Popularity matrix...")
    B = popularity_matrix(df)
    print(f"  Shape: {B.shape}, mean: {B.mean():.4f}")

    print("Building Random matrix...")
    C = random_matrix(df)
    print(f"  Shape: {C.shape}, mean off-diagonal: {C[np.triu_indices_from(C, k=1)].mean():.4f}")
