"""
CineMatch AI — Trend Detection
Satisfies brief requirement "מגמות" (trend detection).

Computes from the catalog:
  - Avg rating per decade (quality trends over time)
  - Rising genres (% in 2010s+2020s vs % in pre-2000s, ratio)
  - Declining genres (inverse)
  - Genre diversity per decade (Shannon entropy)
  - Production volume per decade (count of shows)

Output: data/trends.json
Run:   python -m engine.trends
"""

import os, json, math
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT  = os.path.join(BASE, "data", "catalog.parquet")
OUT  = os.path.join(BASE, "data", "trends.json")


def shannon_entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return round(ent, 3)


def parse_genres(g_str) -> list:
    if not g_str or pd.isna(g_str):
        return []
    return [g.strip() for g in str(g_str).split(",") if g.strip()]


def run() -> dict:
    print(f"Loading catalog from {CAT}...")
    df = pd.read_parquet(CAT)
    print(f"  {len(df)} rows")

    # Drop unknowns for trend analysis
    df = df[df["decade_str"] != "Unknown"].copy()
    df = df[df["start_year"].notna()].copy()
    df["start_year"] = df["start_year"].astype(int)
    print(f"  {len(df)} rows with known year")

    out: dict = {}

    # ── Decade-level quality trend ─────────────────────────────────────────
    decade_stats = (df.groupby("decade_str")
                      .agg(n_shows=("title", "count"),
                           avg_rating=("rating", "mean"),
                           median_rating=("rating", "median"),
                           total_votes=("votes", "sum"))
                      .reset_index()
                      .sort_values("decade_str"))
    decade_stats["avg_rating"]    = decade_stats["avg_rating"].round(3)
    decade_stats["median_rating"] = decade_stats["median_rating"].round(3)
    out["decade_stats"] = decade_stats.to_dict("records")

    # ── Rising / declining genres ──────────────────────────────────────────
    modern_mask  = df["start_year"] >= 2010
    classic_mask = df["start_year"] < 2000

    modern_genres  = Counter()
    classic_genres = Counter()
    for _, row in df[modern_mask].iterrows():
        for g in parse_genres(row.get("genres", "")):
            modern_genres[g] += 1
    for _, row in df[classic_mask].iterrows():
        for g in parse_genres(row.get("genres", "")):
            classic_genres[g] += 1

    n_modern  = max(1, sum(modern_genres.values()))
    n_classic = max(1, sum(classic_genres.values()))

    all_genres = set(modern_genres) | set(classic_genres)
    genre_trends = []
    for g in all_genres:
        m_share = modern_genres[g]  / n_modern
        c_share = classic_genres[g] / n_classic
        # Laplace-smoothed ratio (avoids div by 0)
        ratio = (m_share + 0.001) / (c_share + 0.001)
        genre_trends.append({
            "genre": g,
            "modern_count":  modern_genres[g],
            "classic_count": classic_genres[g],
            "modern_share":  round(m_share * 100, 2),
            "classic_share": round(c_share * 100, 2),
            "rise_ratio":    round(ratio, 3),
        })

    genre_trends.sort(key=lambda r: -r["rise_ratio"])
    # Require at least 30 modern shows to be considered (avoid niche noise)
    significant = [g for g in genre_trends if g["modern_count"] >= 30 or g["classic_count"] >= 30]
    out["top_rising_genres"]    = significant[:8]
    out["top_declining_genres"] = sorted(significant, key=lambda r: r["rise_ratio"])[:8]

    # ── Genre diversity per decade (Shannon entropy) ───────────────────────
    diversity_per_decade = {}
    for decade, grp in df.groupby("decade_str"):
        c = Counter()
        for _, row in grp.iterrows():
            for g in parse_genres(row.get("genres", "")):
                c[g] += 1
        diversity_per_decade[decade] = {
            "shannon_entropy": shannon_entropy(c),
            "unique_genres":   len(c),
            "total_shows":     len(grp),
        }
    out["diversity_per_decade"] = diversity_per_decade

    # ── Headline insights (for the report + UI) ────────────────────────────
    if len(decade_stats) >= 3:
        ratings = decade_stats["avg_rating"].tolist()
        out["headline"] = {
            "best_decade":   decade_stats.iloc[ratings.index(max(ratings))]["decade_str"],
            "worst_decade":  decade_stats.iloc[ratings.index(min(ratings))]["decade_str"],
            "rating_delta":  round(max(ratings) - min(ratings), 2),
            "top_rising_genre":    out["top_rising_genres"][0]["genre"]    if out["top_rising_genres"]    else None,
            "top_declining_genre": out["top_declining_genres"][0]["genre"] if out["top_declining_genres"] else None,
        }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\nSaved → {OUT}")
    print(f"\nHeadline:")
    print(json.dumps(out.get("headline", {}), indent=2))
    print(f"\nTop 3 rising genres: " +
          ", ".join(g["genre"] for g in out["top_rising_genres"][:3]))
    print(f"Top 3 declining genres: " +
          ", ".join(g["genre"] for g in out["top_declining_genres"][:3]))
    return out


if __name__ == "__main__":
    run()
