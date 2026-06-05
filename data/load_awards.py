"""
CineMatch AI — Awards label loader.

Scrapes Wikipedia Emmy (TV) and Academy Award (movie) winner/nominee tables,
normalises titles, and joins them to data/catalog.parquet.

Output: data/awards_labeled.parquet with binary columns:
    oscar_nominated, oscar_won, emmy_nominated, emmy_won, num_award_mentions

If network scrape fails (offline / Wikipedia layout change), falls back to a
hardcoded high-confidence set of well-known winners so downstream training still
produces a usable model.

Public sources only — no API keys required.
"""
from __future__ import annotations

import re
import sys
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
CATALOG_PATH = BASE / "data" / "catalog.parquet"
OUT_PATH = BASE / "data" / "awards_labeled.parquet"
SOURCES_LOG = BASE / "data" / "awards_sources.json"

# ── Wikipedia pages we try to harvest ────────────────────────────────────────
EMMY_PAGES = [
    "https://en.wikipedia.org/wiki/Primetime_Emmy_Award_for_Outstanding_Drama_Series",
    "https://en.wikipedia.org/wiki/Primetime_Emmy_Award_for_Outstanding_Comedy_Series",
    "https://en.wikipedia.org/wiki/Primetime_Emmy_Award_for_Outstanding_Limited_or_Anthology_Series",
]
OSCAR_PAGES = [
    "https://en.wikipedia.org/wiki/Academy_Award_for_Best_Picture",
]

# ── Fallback hard-coded labels (well-known winners since 2000) ──────────────
# Used if Wikipedia scrape fails — keeps the pipeline reproducible offline.
FALLBACK_EMMY_WINNERS = {
    "The Sopranos", "Lost", "24", "Mad Men", "Breaking Bad", "Homeland",
    "Game of Thrones", "The Handmaid's Tale", "Succession", "The Crown",
    "Schitt's Creek", "Ted Lasso", "The Marvelous Mrs. Maisel",
    "Veep", "Modern Family", "30 Rock", "The Office", "Arrested Development",
    "Fleabag", "Hacks", "The Bear", "Chernobyl", "Big Little Lies",
    "The Queen's Gambit", "Mare of Easttown", "Beef",
}
FALLBACK_EMMY_NOMINEES = FALLBACK_EMMY_WINNERS | {
    "House of Cards", "Better Call Saul", "Westworld", "Stranger Things",
    "This Is Us", "Boardwalk Empire", "Six Feet Under", "Friday Night Lights",
    "The Wire", "Deadwood", "Dexter", "Damages", "Boston Legal", "The West Wing",
    "Curb Your Enthusiasm", "Atlanta", "Barry", "Russian Doll", "Black-ish",
    "The Good Place", "Master of None", "Silicon Valley", "Transparent",
    "Severance", "Yellowjackets", "Pachinko", "The Last of Us",
    "Andor", "House of the Dragon", "The White Lotus", "Squid Game",
}
FALLBACK_OSCAR_NOMINEES = {
    "The Godfather", "Forrest Gump", "Titanic", "The Shawshank Redemption",
    "Schindler's List", "Pulp Fiction", "Goodfellas", "The Dark Knight",
    "Inception", "Parasite", "Everything Everywhere All at Once", "Oppenheimer",
    "La La Land", "Moonlight", "Birdman", "Spotlight", "Argo", "12 Years a Slave",
    "The King's Speech", "Slumdog Millionaire", "No Country for Old Men",
}


def normalize_title(s) -> str:
    """Lowercase, strip articles + punctuation, collapse spaces."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower()
    # remove parenthetical year/qualifier
    s = re.sub(r"\([^)]*\)", "", s)
    # strip leading article
    s = re.sub(r"^(the|a|an)\s+", "", s)
    # drop punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_titles_from_tables(url: str) -> tuple[set[str], set[str]]:
    """Return (winners, all_mentioned). Heuristic: first-column-italicised entry
    per row is often the winner; remainder are nominees in the same row."""
    try:
        import io as _io
        import urllib.request as _ur
        req = _ur.Request(url, headers={
            "User-Agent": "CineMatchAI/1.0 (academic project; contact tomer.blond@winn.ai)"
        })
        with _ur.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        tables = pd.read_html(_io.StringIO(html), flavor="lxml")
    except Exception as e:
        print(f"  ⚠ failed {url}: {e}", file=sys.stderr)
        return set(), set()

    winners, all_titles = set(), set()
    for tbl in tables:
        # heuristic: tables that have a "Program" / "Series" / "Film" column
        cols = [str(c).lower() for c in tbl.columns]
        text_cols = [
            i for i, c in enumerate(cols)
            if any(k in c for k in ("program", "series", "film", "title", "show"))
        ]
        if not text_cols:
            continue
        for _, row in tbl.iterrows():
            for ci in text_cols:
                val = row.iloc[ci]
                if isinstance(val, str):
                    # split on newlines / "and"
                    for piece in re.split(r"[\n;]| and ", val):
                        piece = piece.strip()
                        if 2 < len(piece) < 80:
                            all_titles.add(piece)
        # winners often appear in a bolded sub-row; without DOM parsing we use the
        # first text-col value of each year-group as a proxy
        for _, row in tbl.iterrows():
            ci = text_cols[0]
            val = row.iloc[ci]
            if isinstance(val, str):
                first = re.split(r"[\n;]| and ", val)[0].strip()
                if 2 < len(first) < 80:
                    winners.add(first)
    return winners, all_titles


def scrape_emmys() -> tuple[set[str], set[str]]:
    winners, nominees = set(), set()
    for url in EMMY_PAGES:
        w, a = _extract_titles_from_tables(url)
        winners |= w
        nominees |= a
    return winners, nominees


def scrape_oscars() -> tuple[set[str], set[str]]:
    winners, nominees = set(), set()
    for url in OSCAR_PAGES:
        w, a = _extract_titles_from_tables(url)
        winners |= w
        nominees |= a
    return winners, nominees


def main() -> None:
    print("→ Loading catalog…")
    cat = pd.read_parquet(CATALOG_PATH)
    cat["title_norm"] = cat["title"].map(normalize_title)

    print("→ Scraping Wikipedia Emmy tables…")
    try:
        emmy_w, emmy_n = scrape_emmys()
        print(f"   Emmy: {len(emmy_w)} winners, {len(emmy_n)} nominee-candidates")
    except Exception as e:
        print(f"   ⚠ scrape failed: {e} — using fallback only", file=sys.stderr)
        emmy_w, emmy_n = set(), set()

    print("→ Scraping Wikipedia Oscar tables…")
    try:
        osc_w, osc_n = scrape_oscars()
        print(f"   Oscar: {len(osc_w)} winners, {len(osc_n)} nominee-candidates")
    except Exception as e:
        print(f"   ⚠ scrape failed: {e}", file=sys.stderr)
        osc_w, osc_n = set(), set()

    # fold in fallback sets so we always have signal
    emmy_w |= FALLBACK_EMMY_WINNERS
    emmy_n |= FALLBACK_EMMY_NOMINEES
    osc_n |= FALLBACK_OSCAR_NOMINEES

    emmy_w_norm = {normalize_title(s) for s in emmy_w}
    emmy_n_norm = {normalize_title(s) for s in emmy_n}
    osc_w_norm = {normalize_title(s) for s in osc_w}
    osc_n_norm = {normalize_title(s) for s in osc_n}

    cat["emmy_won"] = cat["title_norm"].isin(emmy_w_norm).astype(int)
    cat["emmy_nominated"] = (
        cat["title_norm"].isin(emmy_n_norm) | (cat["emmy_won"] == 1)
    ).astype(int)
    cat["oscar_won"] = cat["title_norm"].isin(osc_w_norm).astype(int)
    cat["oscar_nominated"] = (
        cat["title_norm"].isin(osc_n_norm) | (cat["oscar_won"] == 1)
    ).astype(int)
    cat["num_award_mentions"] = (
        cat["emmy_nominated"] + cat["emmy_won"]
        + cat["oscar_nominated"] + cat["oscar_won"]
    ).astype(int)

    label_cols = [
        "title", "title_norm",
        "emmy_nominated", "emmy_won",
        "oscar_nominated", "oscar_won",
        "num_award_mentions",
    ]
    out = cat[label_cols].copy()
    out.to_parquet(OUT_PATH, index=False)

    pos = int((out["emmy_nominated"] | out["oscar_nominated"]).sum())
    print(f"✅ Wrote {OUT_PATH} — {pos}/{len(out)} positive labels "
          f"(emmy_nom | oscar_nom)")

    SOURCES_LOG.write_text(json.dumps({
        "emmy_pages": EMMY_PAGES,
        "oscar_pages": OSCAR_PAGES,
        "n_emmy_winners": len(emmy_w),
        "n_emmy_nominees": len(emmy_n),
        "n_oscar_winners": len(osc_w),
        "n_oscar_nominees": len(osc_n),
        "n_positive_matches_in_catalog": pos,
    }, indent=2))


if __name__ == "__main__":
    main()
