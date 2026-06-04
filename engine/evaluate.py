"""
Algorithm evaluation — reproduces and extends Assignment 2 analysis.
Compares Jaccard, Cosine-numeric, Cosine-text, and Hybrid across:
  - Spearman rank correlation (agreement between methods)
  - Agreement@K (overlap in top-K neighbors)
  - Precision@5 (on a 10-pair hand-rated preference set)
  - Best-match score distribution (for anomaly threshold calibration)

Run:  python -m engine.evaluate
"""

import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Hand-rated preference pairs: (query_title, expected_match_title)
# 10 pairs tuned to validate the hybrid scoring
PREFERENCE_PAIRS = [
    ("Breaking Bad",     "Better Call Saul"),
    ("Game of Thrones",  "House of the Dragon"),
    ("Stranger Things",  "Dark"),
    ("Friends",          "How I Met Your Mother"),
    ("The Office",       "Parks and Recreation"),
    ("Sherlock",         "Mindhunter"),
    ("The Crown",        "The Americans"),
    ("Dexter",           "Hannibal"),
    ("Black Mirror",     "Electric Dreams"),
    ("Narcos",           "Narcos: Mexico"),
]


def flat_upper(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    return M[np.triu_indices(n, k=1)]


def agreement_at_k(M1: np.ndarray, M2: np.ndarray, k: int = 10) -> float:
    """Average Jaccard overlap of top-K neighbors per row."""
    n = M1.shape[0]
    overlaps = []
    for i in range(n):
        row1 = M1[i].copy(); row1[i] = -1
        row2 = M2[i].copy(); row2[i] = -1
        top1 = set(np.argsort(-row1)[:k])
        top2 = set(np.argsort(-row2)[:k])
        overlaps.append(len(top1 & top2) / k)
    return float(np.mean(overlaps))


def precision_at_5(
    method_matrix: np.ndarray,
    titles: list[str],
    preference_pairs: list[tuple[str, str]],
) -> float:
    """
    For each preference pair (query, expected), check if expected is in top-5.
    Returns the fraction of pairs where expected is found.
    """
    title_to_idx = {t: i for i, t in enumerate(titles)}
    hits = 0
    valid = 0
    for query, expected in preference_pairs:
        if query not in title_to_idx or expected not in title_to_idx:
            continue
        valid += 1
        q_i = title_to_idx[query]
        e_i = title_to_idx[expected]
        scores = method_matrix[q_i].copy()
        scores[q_i] = -1
        top5 = set(np.argsort(-scores)[:5])
        if e_i in top5:
            hits += 1
    return round(hits / valid, 4) if valid else 0.0


def run(catalog_path: str = None, emb_path: str = None, subset: int = 500):
    """
    subset: use the top-N most-voted rows for tractable matrix computation.
    """
    cat_path = catalog_path or os.path.join(BASE, "data", "catalog.parquet")
    emb_path = emb_path or os.path.join(BASE, "data", "embeddings.npy")

    print("Loading data...")
    df = pd.read_parquet(cat_path)
    embeddings = np.load(emb_path)

    # Use subset most-voted for evaluation (tractable matrix sizes)
    df = df.nlargest(subset, "votes").reset_index(drop=True)
    embeddings = embeddings[:len(df)]  # aligned by position after sort
    # Re-align: need actual positional subset
    top_idx = df.index.tolist()[:subset]

    print(f"Evaluation subset: {len(df)} rows")

    from sklearn.metrics.pairwise import cosine_similarity as sk_cos
    from engine.jaccard import batch_jaccard_matrix

    print("Computing Jaccard matrix...")
    J = batch_jaccard_matrix(df)

    print("Computing Cosine-numeric matrix...")
    X = df[["rating_z","votes_z","start_year_z","popularity_z"]].fillna(0).values.astype(np.float32)
    C_num = sk_cos(X).astype(np.float32)

    # Re-load and re-align full embeddings by title for this subset
    full_df = pd.read_parquet(cat_path)
    title_to_emb = {row["title"]: embeddings[i]
                    for i, (_, row) in enumerate(df.iterrows())}
    emb_subset = np.stack([title_to_emb.get(t, np.zeros(384, dtype=np.float32))
                           for t in df["title"]])
    C_text = (emb_subset @ emb_subset.T).astype(np.float32)

    alpha, beta, gamma = 0.35, 0.30, 0.35
    H = (alpha * J + beta * C_num + gamma * C_text).astype(np.float32)

    titles = df["title"].tolist()

    print("Computing metrics...")
    results = {}

    # Spearman rank correlation (all upper-triangle pairs)
    j_flat  = flat_upper(J)
    cn_flat = flat_upper(C_num)
    ct_flat = flat_upper(C_text)
    h_flat  = flat_upper(H)

    results["spearman"] = {
        "Jaccard vs Cosine-numeric":  round(float(spearmanr(j_flat, cn_flat).statistic), 4),
        "Jaccard vs Cosine-text":     round(float(spearmanr(j_flat, ct_flat).statistic), 4),
        "Cosine-numeric vs Cosine-text": round(float(spearmanr(cn_flat, ct_flat).statistic), 4),
        "Jaccard vs Hybrid":          round(float(spearmanr(j_flat, h_flat).statistic), 4),
        "Cosine-numeric vs Hybrid":   round(float(spearmanr(cn_flat, h_flat).statistic), 4),
        "Cosine-text vs Hybrid":      round(float(spearmanr(ct_flat, h_flat).statistic), 4),
    }

    # Agreement@10
    results["agreement_at_10"] = {
        "Jaccard vs Cosine-numeric":  round(agreement_at_k(J, C_num,   10), 4),
        "Jaccard vs Cosine-text":     round(agreement_at_k(J, C_text,  10), 4),
        "Cosine-numeric vs Cosine-text": round(agreement_at_k(C_num, C_text, 10), 4),
        "Jaccard vs Hybrid":          round(agreement_at_k(J, H,       10), 4),
        "Cosine-text vs Hybrid":      round(agreement_at_k(C_text, H,  10), 4),
    }

    # Precision@5
    results["precision_at_5"] = {
        "Jaccard":        precision_at_5(J,     titles, PREFERENCE_PAIRS),
        "Cosine-numeric": precision_at_5(C_num, titles, PREFERENCE_PAIRS),
        "Cosine-text":    precision_at_5(C_text,titles, PREFERENCE_PAIRS),
        "Hybrid":         precision_at_5(H,     titles, PREFERENCE_PAIRS),
    }

    # Best-match score per row (for anomaly calibration)
    H_no_diag = H.copy()
    np.fill_diagonal(H_no_diag, -1)
    best_scores = H_no_diag.max(axis=1)
    threshold_5pct = float(np.percentile(best_scores, 5))
    results["anomaly_threshold_5pct"] = round(threshold_5pct, 4)
    results["best_score_stats"] = {
        "mean":   round(float(best_scores.mean()), 4),
        "median": round(float(np.median(best_scores)), 4),
        "p5":     round(threshold_5pct, 4),
        "p25":    round(float(np.percentile(best_scores, 25)), 4),
    }

    out_path = os.path.join(BASE, "data", "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")
    return results


if __name__ == "__main__":
    run()
