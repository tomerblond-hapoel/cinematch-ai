# CineMatch AI — Claude Code Context

Final project for AI & ML Innovation Workshop 2025. Author: Tomer Blond.

## What this is

A bilingual (Hebrew / English) TV & movie recommendation agent. User types free-text query in either language → Claude parses intent → hybrid similarity engine ranks → Claude explains. 11,013 cleaned TV titles from 4 sources (TMDb, Disney+, IMDb top-10k, IMDb top-5000).

## Key files

```
data/
  load_catalog.py     # 4-source merge → catalog.parquet
  embed.py            # sentence-transformers → embeddings.npy
  catalog.parquet     # 11,013 rows × 22 cols
  embeddings.npy      # 11,013 × 384 float32
  cleaning_log.json   # per-source drop stats
  evaluation_results.json  # Spearman/Agreement/Precision@5 incl. baselines
  trends.json         # rising/declining genres, decade quality
engine/
  jaccard.py          # discrete set similarity
  cosine.py           # numeric + text embedding cosine
  hybrid.py           # weighted combo; Hebrew bias mitigation here
  anomaly.py          # 5th-percentile threshold detector
  baselines.py        # TF-IDF / Popularity / Random for rubric
  trends.py           # rising/declining genres + Shannon entropy
  evaluate.py         # full comparison harness
agent/
  llm.py              # Claude intent parser + bilingual explainer + regex fallback
app.py                # polished Streamlit UI
i18n.py               # bilingual strings dict
tests/test_cases.py   # 3 success + 1 anomaly failure case
report/
  CineMatch_Report.docx   # 10-page final report
```

## Hard constraints (from the Hebrew brief)

1. **No local server allowed** — must deploy to Render.com (or college server).
2. **10 page max** report with cover + TOC.
3. **Live demo** — instructor must be able to run the agent at any time.
4. **Bilingual** is a project differentiator — don't break Hebrew RTL.

## Hot keys for upgrading later

- **Change UI colors**: edit `BRAND` dict at top of `app.py`.
- **Change weights**: edit `ALPHA / BETA / HEBREW_WEIGHTS` in `engine/hybrid.py`.
- **Change LLM model**: edit model name in `agent/llm.py` (two places: parse_intent + explain_recommendations).
- **Add a new data source**: write a loader in `data/load_catalog.py` matching the `KEEP_COLS` schema, then re-run `python data/load_catalog.py && python data/embed.py && python -m engine.evaluate && python -m engine.trends`.
- **Push to deploy**: `git push` → Render auto-redeploys.

## Re-run everything (full pipeline)

```bash
pip install -r requirements.txt
python data/load_catalog.py     # ~30s
python data/embed.py            # ~1 min (cached after first run)
python -m engine.evaluate       # ~30s
python -m engine.trends         # ~5s
python tests/test_cases.py      # 4/4 pass
streamlit run app.py            # local preview
```

## Brief + rubric (original, in Hebrew)

- `/Users/tomerblond/Desktop/שנה ג/סדנא AI/Final project/תרגיל סיום 2025 סדנת חדשנות מבוססת AI (2).docx`
- `/Users/tomerblond/Desktop/שנה ג/סדנא AI/Final project/מחוון הערכת פרוייקט סיום סדנה מבוססת AI ו ML (1).docx`
