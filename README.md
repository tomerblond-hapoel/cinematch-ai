# CineMatch AI

Bilingual (Hebrew/English) AI-powered TV & Movie recommendation agent.  
Final project — AI & ML Innovation Workshop 2025.  
**Author:** Tomer Blond

## Live Demo
Deployed on Render.com: [link added after deploy]

## GitHub
https://github.com/tomerblond-hapoel/cinematch-ai

## Setup

```bash
git clone <repo>
cd CineMatch

# Install dependencies
pip install -r requirements.txt

# Set your API key (get free key at anthropic.com)
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...

# Step 1: Build the data catalog (takes ~3 min first run)
python data/load_catalog.py

# Step 2: Generate synopsis embeddings (takes ~5 min, downloads 120MB model once)
python data/embed.py

# Step 3 (optional): Run evaluation metrics
python -m engine.evaluate

# Step 4: Run the app
streamlit run app.py
```

## Architecture

```
User query (Hebrew or English)
    │
    ▼
Claude claude-sonnet-4-6 — intent parser
  → JSON: {seeds, mood, length_pref, lang}
    │
    ▼
┌─────────────────────────────────────────┐
│  Jaccard (genres ∪ decade ∪ language)  │
│  Cosine  (rating, votes, year, pop)     │
│  Cosine  (384-dim plot embeddings)      │
│  Hybrid  α·J + β·C_num + γ·C_text      │
│  Anomaly detector (5th-pct threshold)   │
└─────────────────────────────────────────┘
    │
    ▼
Claude claude-sonnet-4-6 — explanation generator
  → Bilingual natural-language explanation
    │
    ▼
Streamlit UI — poster grid + score breakdown
```

## Data Sources (4 total, 166,592 raw rows)

| Source | Raw | Platform |
|---|---|---|
| TMDb TV Shows | 152,971 | The Movie Database |
| Disney+ catalog | 993 | Disney+ / OMDb |
| IMDb Top-10k Scrape | 10,001 | IMDb |
| IMDb Top-5000 TV | 2,627 | IMDb |

## Tests

```bash
python tests/test_cases.py
```

3 success cases + 1 anomaly/failure case.

## Deploy to Render.com

1. Push to GitHub.
2. Create a new Render Web Service, select the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Add env var: `ANTHROPIC_API_KEY=sk-ant-...`
6. Data files (catalog.parquet, embeddings.npy) are already committed to the repo — Render will pick them up automatically.
