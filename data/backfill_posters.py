"""
Poster Backfill Script — CineMatch AI
Fetches missing poster_path values from TMDb API by title search.
Saves results to data/posters_backfill.json

Usage:
    set TMDB_API_KEY=your_key_here
    python data/backfill_posters.py

Get a free TMDb API key at: https://www.themoviedb.org/settings/api
"""

import os, json, time, requests
import pandas as pd
from pathlib import Path

BASE       = Path(__file__).parent
CATALOG    = BASE / "catalog.parquet"
OUT        = BASE / "posters_backfill.json"
TMDB_KEY   = os.environ.get("TMDB_API_KEY", "")
SEARCH_URL = "https://api.themoviedb.org/3/search/multi"

def search_poster(title: str, year: str = None) -> str:
    """Return poster_path string (e.g. '/abc123.jpg') or '' if not found."""
    params = {
        "api_key": TMDB_KEY,
        "query": title,
        "include_adult": "false",
        "language": "en-US",
        "page": 1,
    }
    if year and str(year).isdigit():
        params["first_air_date_year"] = int(year)

    try:
        r = requests.get(SEARCH_URL, params=params, timeout=8)
        r.raise_for_status()
        results = r.json().get("results", [])
        # Prefer TV results, then movies
        for item in results:
            if item.get("poster_path"):
                return item["poster_path"]
    except Exception as e:
        print(f"  ✗ Error for '{title}': {e}")
    return ""


def main():
    if not TMDB_KEY:
        print("ERROR: TMDB_API_KEY environment variable not set.")
        print("Get a free key at https://www.themoviedb.org/settings/api")
        return

    print("Loading catalog...")
    cat = pd.read_parquet(CATALOG)

    # Load existing backfill if any
    existing = {}
    if OUT.exists():
        existing = json.load(open(OUT))
        print(f"Loaded {len(existing)} existing backfill entries.")

    # Find shows missing posters
    missing_mask = (
        cat["poster_path"].isna() |
        (cat["poster_path"].astype(str).str.strip().str.len() <= 1)
    ) if "poster_path" in cat.columns else pd.Series([True] * len(cat))

    # Also skip titles already in backfill
    to_fetch = cat[missing_mask & ~cat["title"].isin(existing)][["title", "year"]].drop_duplicates("title")
    print(f"Need to fetch: {len(to_fetch)} titles")

    results = dict(existing)
    found = 0

    for i, (_, row) in enumerate(to_fetch.iterrows()):
        title = row["title"]
        year  = row.get("year", None)

        poster = search_poster(title, year)
        if poster:
            results[title] = poster
            found += 1

        # Progress every 50
        if (i + 1) % 50 == 0:
            pct = (i + 1) / len(to_fetch) * 100
            print(f"  {i+1}/{len(to_fetch)} ({pct:.0f}%) — found so far: {found}")
            # Save checkpoint
            json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        # Rate limit: TMDb free tier = 50 req/sec, stay safe
        time.sleep(0.05)

    # Final save
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nDone. {found} new posters found out of {len(to_fetch)} searched.")
    print(f"Total in backfill: {len(results)}")
    print(f"Saved to: {OUT}")


if __name__ == "__main__":
    main()
