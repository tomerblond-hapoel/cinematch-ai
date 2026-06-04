"""
CineMatch AI — Data Pipeline
Reads 4 raw source sheets from A1 xlsx + A2 CSV, normalizes, cleans, dedupes.
Output: data/catalog.parquet

Cleaning narrative (for report Section 4):
  Source 1  tmdb_tvs        152,971 raw rows  (TMDb TV Shows dataset)
  Source 2  disney_plus      993 raw rows  (Disney+ catalog with OMDb enrichment)
  Source 3  imdb_all       10,001 raw rows  (IMDb Top-10k scrape)
  Source 4  imdb_top5000    2,627 raw rows  (IMDb Top-5000 TV series)
  ─────────────────────────────────────────
  TOTAL                   166,592 raw rows  (across 4 sources)
"""

import zipfile, io, os, re, sys
import pandas as pd
import numpy as np
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(BASE, "..", "Assignment 1.xlsx (3).zip")
A2_CSV   = os.path.join(BASE, "..", "..", "תרגיל 2", "imdb_top_5000_tv_shows.csv")
OUT_PATH = os.path.join(BASE, "data", "catalog.parquet")
LOG_PATH = os.path.join(BASE, "data", "cleaning_log.json")

# ── helpers ──────────────────────────────────────────────────────────────────

def normalize_genres(s):
    if not s or pd.isna(s):
        return ""
    parts = re.split(r"[,|/]", str(s))
    return ", ".join(sorted({p.strip().title() for p in parts if p.strip()}))

def safe_float(v, default=None):
    try:
        f = float(v)
        return f if pd.notna(f) else default
    except:
        return default

def safe_int(v, default=None):
    try:
        return int(float(v))
    except:
        return default

def decade_str(year):
    try:
        if year is None or (isinstance(year, float) and np.isnan(year)):
            return "Unknown"
        d = (int(year) // 10) * 10
        return f"{d}s"
    except:
        return "Unknown"

def genre_set(genres_str):
    if not genres_str:
        return frozenset()
    return frozenset(g.strip() for g in genres_str.split(",") if g.strip())

# ── source 1: TMDb TV shows ───────────────────────────────────────────────────

def load_tmdb(xls):
    print("  Loading tmdb_tvs...")
    cols = {
        "name": "title",
        "original_language": "language",
        "first_air_date": "first_air_date",
        "last_air_date": "last_air_date",
        "number_of_episodes": "num_episodes",
        "number_of_seasons": "num_seasons",
        "vote_average": "rating",
        "vote_count": "votes",
        "popularity": "popularity",
        "overview": "overview",
        "poster_path": "poster_path",
        "genres[0].name": "g0",
        "genres[1].name": "g1",
        "genres[2].name": "g2",
        "genres[3].name": "g3",
    }
    df = pd.read_excel(xls, sheet_name="tvs", usecols=list(cols.keys()), engine="openpyxl")
    raw_n = len(df)
    df = df.rename(columns=cols)

    # build genres string
    df["genres"] = df[["g0","g1","g2","g3"]].fillna("").apply(
        lambda r: normalize_genres(",".join(v for v in r if v)), axis=1)
    df = df.drop(columns=["g0","g1","g2","g3"])

    # parse year from date
    def year_from(d):
        try:
            return int(str(d)[:4])
        except:
            return None

    df["start_year"] = df["first_air_date"].apply(year_from)
    df["end_year"]   = df["last_air_date"].apply(year_from)
    df = df.drop(columns=["first_air_date","last_air_date"])

    df["rating"]     = df["rating"].apply(lambda v: safe_float(v))
    df["votes"]      = df["votes"].apply(lambda v: safe_int(v))
    df["popularity"] = df["popularity"].apply(lambda v: safe_float(v, 0.0))
    df["num_episodes"] = df["num_episodes"].apply(lambda v: safe_int(v))
    df["num_seasons"]  = df["num_seasons"].apply(lambda v: safe_int(v))
    df["overview"]   = df["overview"].fillna("").astype(str)
    df["poster_path"]= df["poster_path"].fillna("").astype(str)
    df["source_dataset"] = "tmdb_tvs"

    # cleaning
    dropped = {}
    n0 = len(df)
    df = df[df["title"].notna() & (df["title"].astype(str).str.strip() != "")]
    dropped["no_title"] = n0 - len(df); n0 = len(df)

    df = df[df["votes"].notna() & (df["votes"] >= 10)]
    dropped["votes_lt_10"] = n0 - len(df); n0 = len(df)

    df = df[df["rating"].notna() & (df["rating"] >= 1.0)]
    dropped["rating_lt_1"] = n0 - len(df); n0 = len(df)

    df = df[df["genres"].str.strip() != ""]
    dropped["no_genres"] = n0 - len(df)

    print(f"    {raw_n} raw → {len(df)} clean | dropped: {dropped}")
    return df, raw_n, len(df)

# ── source 2: Disney+ shows ───────────────────────────────────────────────────

def load_disney(xls):
    print("  Loading disney_plus_shows...")
    df = pd.read_excel(xls, sheet_name="disney_plus_shows", engine="openpyxl")
    raw_n = len(df)

    # keep only series
    df = df[df["type"].str.lower().str.contains("series", na=False)]
    dropped = {"movies": raw_n - len(df)}

    df = df.rename(columns={
        "title": "title",
        "plot": "overview",
        "year": "year_str",
        "genre": "genres_raw",
        "imdb_rating": "rating",
        "imdb_votes": "votes",
        "language": "language",
    })

    def parse_year(y):
        try:
            return int(str(y)[:4])
        except:
            return None

    df["start_year"] = df["year_str"].apply(parse_year)
    df["end_year"]   = None
    df["genres"]     = df["genres_raw"].apply(normalize_genres)
    df["rating"]     = df["rating"].apply(lambda v: safe_float(v))
    df["votes"]      = df["votes"].apply(lambda v: safe_int(str(v).replace(",","")))
    df["popularity"] = 0.0
    df["num_episodes"] = None
    df["num_seasons"]  = None
    df["overview"]   = df["overview"].fillna("").astype(str)
    df["poster_path"]= ""
    df["source_dataset"] = "disney_plus"

    n0 = len(df)
    df = df[df["title"].notna() & (df["title"].astype(str).str.strip() != "")]
    dropped["no_title"] = n0 - len(df); n0 = len(df)
    df = df[df["genres"].str.strip() != ""]
    dropped["no_genres"] = n0 - len(df)

    print(f"    {raw_n} raw → {len(df)} clean | dropped: {dropped}")
    return df, raw_n, len(df)

# ── source 3: IMDb all cleaned ────────────────────────────────────────────────

def load_imdb_all(xls):
    print("  Loading imdb_data_cleaned...")
    df = pd.read_excel(xls, sheet_name="imdb_data_cleaned", engine="openpyxl")
    raw_n = len(df)

    # keep only TV shows
    df = df[df["type"].str.lower().str.contains("tv", na=False)]
    dropped = {"movies": raw_n - len(df)}

    df = df.rename(columns={
        "title": "title",
        "releaseyear": "start_year",
        "endyear": "end_year",
        "genres": "genres_raw",
        "imdbrating": "rating",
        "numvotes": "votes",
        "popularityrank": "popularity",
    })

    df["genres"]     = df["genres_raw"].apply(normalize_genres)
    df["rating"]     = df["rating"].apply(lambda v: safe_float(v))
    df["votes"]      = df["votes"].apply(lambda v: safe_int(v))
    df["start_year"] = df["start_year"].apply(lambda v: safe_int(v))
    df["end_year"]   = df["end_year"].apply(lambda v: safe_int(v) if str(v).isdigit() else None)
    df["popularity"] = df["popularity"].apply(lambda v: safe_float(v, 0.0))
    df["num_episodes"] = None
    df["num_seasons"]  = None
    df["overview"]   = ""
    df["poster_path"]= ""
    df["language"]   = "en"
    df["source_dataset"] = "imdb_all"

    n0 = len(df)
    df = df[df["title"].notna()]
    dropped["no_title"] = n0 - len(df)

    print(f"    {raw_n} raw → {len(df)} clean | dropped: {dropped}")
    return df, raw_n, len(df)

# ── source 4: IMDb Top-5000 TV ────────────────────────────────────────────────

def load_imdb_top5000():
    print("  Loading imdb_top_5000_tv_shows (A2 CSV)...")
    df = pd.read_csv(A2_CSV)
    raw_n = len(df)

    df = df.rename(columns={
        "primaryTitle": "title",
        "startYear": "start_year",
        "endYear": "end_year",
        "averageRating": "rating",
        "numVotes": "votes",
        "genres": "genres_raw",
        "rank": "popularity",
    })

    df["genres"]     = df["genres_raw"].apply(normalize_genres)
    df["rating"]     = df["rating"].apply(lambda v: safe_float(v))
    df["votes"]      = df["votes"].apply(lambda v: safe_int(v))
    df["start_year"] = df["start_year"].apply(lambda v: safe_int(v))
    df["end_year"]   = df["end_year"].apply(lambda v: safe_int(v) if str(v).isdigit() else None)
    df["popularity"] = df["popularity"].apply(lambda v: safe_float(v, 5000.0))
    df["num_episodes"] = None
    df["num_seasons"]  = None
    df["overview"]   = ""
    df["poster_path"]= ""
    df["language"]   = "en"
    df["source_dataset"] = "imdb_top5000"

    print(f"    {raw_n} raw → {len(df)} clean | no drops (already clean)")
    return df, raw_n, len(df)

# ── merge + dedupe ────────────────────────────────────────────────────────────

KEEP_COLS = ["title","language","start_year","end_year","genres","rating","votes",
             "popularity","overview","poster_path","num_episodes","num_seasons",
             "source_dataset"]

def merge_and_dedupe(frames):
    print("  Merging sources...")
    combined = pd.concat([f[KEEP_COLS] for f in frames], ignore_index=True)
    pre_dedupe = len(combined)

    # normalize title for dedup key
    combined["_key"] = (combined["title"].str.lower()
                            .str.replace(r"[^a-z0-9\s]", "", regex=True)
                            .str.strip())
    # keep row with highest votes per title key
    combined = (combined
                .sort_values("votes", ascending=False, na_position="last")
                .drop_duplicates(subset="_key", keep="first")
                .drop(columns=["_key"])
                .reset_index(drop=True))

    post_dedupe = len(combined)
    print(f"    Pre-dedupe: {pre_dedupe} rows → Post-dedupe: {post_dedupe} rows "
          f"(removed {pre_dedupe - post_dedupe} duplicates)")
    return combined, pre_dedupe, post_dedupe

# ── feature engineering ───────────────────────────────────────────────────────

def engineer_features(df):
    print("  Engineering features...")

    df["decade"]     = df["start_year"].apply(lambda y: (int(y)//10)*10 if y and not (isinstance(y, float) and np.isnan(y)) else None)
    df["decade_str"] = df["start_year"].apply(decade_str)

    # genre_set as string (pipe-separated for storage; convert to frozenset at query time)
    df["genre_set_str"] = df["genres"].apply(
        lambda g: "|".join(sorted(gs.strip() for gs in g.split(",") if gs.strip())))

    # z-scores for numeric cosine
    for col in ["rating","votes","start_year","popularity"]:
        vals = pd.to_numeric(df[col], errors="coerce")
        if col == "votes":
            vals = np.log1p(vals)
        mu, sigma = vals.mean(), vals.std()
        df[f"{col}_z"] = ((vals - mu) / sigma).round(4).fillna(0)

    # rating bucket (for display)
    def rate_bucket(r):
        if not r: return "No Rating"
        if r >= 8.5: return "Top-Rated (8.5+)"
        if r >= 7.5: return "Excellent (7.5–8.4)"
        if r >= 6.5: return "Good (6.5–7.4)"
        if r >= 5.0: return "Average (5–6.4)"
        return "Below Average (<5)"
    df["rating_bucket"] = df["rating"].apply(rate_bucket)

    # binge_fit_score (replicating A1 formula)
    def binge_score(row):
        r = safe_float(row["rating"], 0) or 0
        v = safe_int(row["votes"], 0) or 0
        vote_pts = 3 if v >= 100000 else (2 if v >= 10000 else (1 if v >= 1000 else 0))
        decade_pts = 1 if row["decade_str"] in ("2010s","2020s") else 0
        return round((r / 10) * 4 + vote_pts + decade_pts, 1)
    df["binge_fit_score"] = df.apply(binge_score, axis=1)

    print(f"    Feature engineering complete. Final shape: {df.shape}")
    return df

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("Loading A1 xlsx from zip...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        xlsx_bytes = z.read("Assignment 1.xlsx")
    xls = io.BytesIO(xlsx_bytes)

    log = {"sources": {}, "total_raw": 0, "total_clean": 0}

    frames = []
    for name, loader, args in [
        ("tmdb_tvs",     load_tmdb,       [xls]),
        ("disney_plus",  load_disney,     [xls]),
        ("imdb_all",     load_imdb_all,   [xls]),
        ("imdb_top5000", load_imdb_top5000, []),
    ]:
        frame, raw, clean = loader(*args)
        frames.append(frame)
        log["sources"][name] = {"raw": raw, "clean": clean,
                                "dropped": raw - clean,
                                "drop_pct": round((raw - clean) / raw * 100, 1)}
        log["total_raw"] += raw

    merged, pre, post = merge_and_dedupe(frames)
    log["pre_dedupe"]  = pre
    log["post_dedupe"] = post
    log["deduped"]     = pre - post

    catalog = engineer_features(merged)
    log["total_clean"] = len(catalog)
    log["drop_rate_overall"] = round((log["total_raw"] - log["total_clean"]) / log["total_raw"] * 100, 1)

    catalog.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved catalog → {OUT_PATH} ({len(catalog)} rows)")

    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved cleaning log → {LOG_PATH}")
    print("\nCleaning summary:")
    print(json.dumps(log, indent=2))

if __name__ == "__main__":
    main()
