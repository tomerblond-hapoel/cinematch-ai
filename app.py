"""
CineMatch AI — Streamlit Application (polished UI)
Bilingual (Hebrew/English) TV & Movie recommendation agent.
"""

import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent
CATALOG_PATH = BASE / "data" / "catalog.parquet"
EMB_PATH     = BASE / "data" / "embeddings.npy"
LOG_PATH     = BASE / "data" / "cleaning_log.json"
EVAL_PATH    = BASE / "data" / "evaluation_results.json"
TRENDS_PATH  = BASE / "data" / "trends.json"

from i18n import t
import agent.llm as llm_agent

# ── Brand palette ──────────────────────────────────────────────────────────────
BRAND = {
    "bg":            "#0a0e1a",
    "bg_card":       "#141826",
    "bg_card_alt":   "#1a1f2e",
    "border":        "#252b3d",
    "border_hover":  "#3a4262",
    "text":          "#e8eaed",
    "text_muted":    "#8b93a8",
    "accent":        "#4fc3f7",   # cyan
    "accent_2":      "#7c4dff",   # purple
    "accent_warm":   "#ff6b9d",   # pink
    "success":       "#4ade80",
    "warning":       "#fbbf24",
    "gradient_1":    "linear-gradient(135deg, #4fc3f7 0%, #7c4dff 100%)",
    "gradient_2":    "linear-gradient(135deg, #ff6b9d 0%, #7c4dff 100%)",
}

st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Cached resources ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_catalog():
    cat = pd.read_parquet(CATALOG_PATH)
    awards_path = BASE / "data" / "award_predictions.parquet"
    if awards_path.exists():
        try:
            preds = pd.read_parquet(awards_path)[["title", "award_probability"]]
            cat = cat.merge(preds, on="title", how="left")
        except Exception:
            pass
    posters_path = BASE / "data" / "posters_backfill.json"
    if posters_path.exists():
        try:
            backfill = json.load(open(posters_path))
            if "poster_path" in cat.columns:
                missing = cat["poster_path"].isna() | (cat["poster_path"].astype(str).str.len() == 0)
                cat.loc[missing, "poster_path"] = cat.loc[missing, "title"].map(backfill).fillna("")
        except Exception:
            pass
    return cat

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return np.load(str(EMB_PATH))

@st.cache_resource(show_spinner=False)
def load_matrices(catalog):
    """
    Build the numeric similarity matrix once, calibrate the anomaly threshold
    on the ACTUAL hybrid scoring scale (not just numeric cosine).

    Why: numeric cosine values cluster near 1.0 for similar-vote/rating shows,
    so calibrating threshold from that matrix gives ~0.99 — and at query time
    hybrid scores (which mix in Jaccard 0-1 and text-cosine -1..1) end up
    much lower → every query gets false-flagged as anomalous.

    Fix: sample 200 catalog rows, compute their best HYBRID match, take 5th
    percentile of that distribution as the threshold.
    """
    from engine.cosine import build_numeric_matrix
    from engine.anomaly import calibrate
    from engine.jaccard import _build_feature_set, jaccard
    import numpy as np

    num_mat = build_numeric_matrix(catalog)

    # Sample-based hybrid threshold calibration
    embeddings = load_embeddings()
    rng = np.random.default_rng(42)
    n = len(catalog)
    sample_size = min(200, n)
    sample_idx = rng.choice(n, sample_size, replace=False)

    # Precompute Jaccard feature sets for sample
    feature_sets = [_build_feature_set(catalog.iloc[i]) for i in sample_idx]

    alpha, beta, gamma = 0.35, 0.30, 0.35
    best_hybrid_scores = []
    for i, src_pos in enumerate(sample_idx):
        # Compute hybrid score from src_pos to every other catalog row
        # Numeric cosine row (already symmetric)
        n_scores = num_mat[src_pos]
        # Text cosine row (dot product with all embeddings)
        t_scores = embeddings @ embeddings[src_pos]
        # Jaccard against the sample set ONLY (fast approx — full would be slow)
        src_set = feature_sets[i]
        j_scores_sample = np.array([jaccard(src_set, fs) for fs in feature_sets])

        # We want the best match across full catalog — use n + t only
        # (Jaccard is approximated to 0 for non-sample rows; OK for percentile)
        full_scores = beta * n_scores + gamma * t_scores
        full_scores[src_pos] = -1  # exclude self
        best_hybrid_scores.append(float(full_scores.max()))

    threshold = calibrate(np.array(best_hybrid_scores), percentile=5)
    return num_mat, threshold

@st.cache_resource(show_spinner=False)
def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def _json(p):
    return json.load(open(p)) if Path(p).exists() else {}

# ── Polished CSS ───────────────────────────────────────────────────────────────

def inject_css(lang: str):
    direction = "rtl" if lang == "he" else "ltr"
    font_main   = "Heebo" if lang == "he" else "Inter"
    font_stack  = f"'{font_main}', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif"
    align_main  = "right" if lang == "he" else "left"

    st.markdown(f"""
    <link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
    /* ── Global ── */
    html, body, [class*="css"], [class*="st"] {{
        font-family: {font_stack} !important;
        direction: {direction} !important;
    }}
    .stApp {{
        background:
          radial-gradient(ellipse 80% 50% at 50% 0%, rgba(124,77,255,0.15), transparent),
          radial-gradient(ellipse 60% 40% at 100% 20%, rgba(79,195,247,0.08), transparent),
          {BRAND['bg']};
    }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 4rem;
        max-width: 1100px;
    }}
    .stTextInput > div > div > input,
    .stTextArea textarea {{
        background: {BRAND['bg_card']} !important;
        border: 1px solid {BRAND['border']} !important;
        border-radius: 14px !important;
        color: {BRAND['text']} !important;
        padding: 14px 18px !important;
        font-size: 16px !important;
        direction: auto !important;
        text-align: start !important;
        unicode-bidi: plaintext !important;
        padding-right: 140px !important;
        transition: all 0.2s ease !important;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: {BRAND['accent']} !important;
        box-shadow: 0 0 0 3px rgba(79,195,247,0.15) !important;
    }}
    .stButton > button {{
        background: {BRAND['gradient_1']} !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(79,195,247,0.25) !important;
        height: 60px !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.3 !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(79,195,247,0.4) !important;
    }}
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: {BRAND['bg_card']};
        border-radius: 14px;
        padding: 6px;
        gap: 4px;
        border: 1px solid {BRAND['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: {BRAND['text_muted']} !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        padding: 8px 18px !important;
        transition: all 0.2s ease !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {BRAND['bg_card_alt']} !important;
        color: {BRAND['text']} !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }}
    /* Hero */
    .cm-hero {{
        text-align: center;
        padding: 2rem 0 1.5rem;
    }}
    .cm-hero-title {{
        font-size: 3.5rem;
        font-weight: 800;
        background: {BRAND['gradient_1']};
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
        line-height: 1.1;
    }}
    .cm-hero-sub {{
        color: {BRAND['text_muted']};
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }}
    .cm-badge-row {{
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }}
    .cm-badge {{
        background: {BRAND['bg_card']};
        border: 1px solid {BRAND['border']};
        color: {BRAND['text_muted']};
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }}
    .cm-badge-accent {{
        background: rgba(79,195,247,0.1);
        border-color: rgba(79,195,247,0.3);
        color: {BRAND['accent']};
    }}
    /* Result card */
    .cm-result {{
        background: linear-gradient(135deg, {BRAND['bg_card']} 0%, {BRAND['bg_card_alt']} 100%);
        border: 1px solid {BRAND['border']};
        border-top: 3px solid {BRAND['border']};
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.25s ease;
        position: relative;
        overflow: visible;
    }}
    .cm-result:hover {{
        border-color: {BRAND['border_hover']};
        border-top-color: {BRAND['accent']};
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }}
    .cm-rank {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px; height: 32px;
        background: {BRAND['gradient_1']};
        color: white;
        border-radius: 10px;
        font-weight: 700;
        margin-{('left' if lang=='he' else 'right')}: 12px;
    }}
    .cm-title {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {BRAND['text']};
        margin-bottom: 0.25rem;
    }}
    .cm-meta {{
        color: {BRAND['text_muted']};
        font-size: 0.85rem;
        margin-bottom: 0.75rem;
    }}
    .cm-score-bar-bg {{
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
        margin: 4px 0 12px;
    }}
    .cm-score-bar-fill {{
        background: {BRAND['gradient_1']};
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }}
    .cm-score-label {{
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: {BRAND['text_muted']};
        margin-bottom: 2px;
    }}
    .cm-explanation {{
        background: linear-gradient(135deg, rgba(79,195,247,0.08) 0%, rgba(124,77,255,0.08) 100%);
        border: 1px solid rgba(79,195,247,0.2);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        color: {BRAND['text']};
        font-size: 0.96rem;
        line-height: 1.6;
    }}
    .cm-anomaly {{
        background: linear-gradient(135deg, rgba(251,191,36,0.12) 0%, rgba(251,113,133,0.12) 100%);
        border: 1px solid rgba(251,191,36,0.3);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        color: {BRAND['warning']};
        font-weight: 500;
        margin-bottom: 1rem;
    }}
    /* Example chips */
    .cm-chips {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.75rem 0 1.5rem;
        justify-content: center;
    }}
    .cm-chip {{
        background: {BRAND['bg_card']};
        border: 1px solid {BRAND['border']};
        color: {BRAND['text_muted']};
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    .cm-chip:hover {{
        border-color: {BRAND['accent']};
        color: {BRAND['accent']};
    }}
    /* Stat tile */
    .cm-stat {{
        background: {BRAND['bg_card']};
        border: 1px solid {BRAND['border']};
        border-radius: 14px;
        padding: 1rem 1.25rem;
        text-align: center;
    }}
    .cm-stat-value {{
        font-size: 1.8rem;
        font-weight: 700;
        background: {BRAND['gradient_1']};
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .cm-stat-label {{
        color: {BRAND['text_muted']};
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }}
    /* Streamlit metric override */
    [data-testid="stMetric"] {{
        background: {BRAND['bg_card']};
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid {BRAND['border']};
    }}
    [data-testid="stMetricLabel"] {{ color: {BRAND['text_muted']} !important; }}
    [data-testid="stMetricValue"] {{ color: {BRAND['text']} !important; }}
    /* Section heading */
    .cm-section-h {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {BRAND['text']};
        margin: 1.5rem 0 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    /* Plotly bg */
    .js-plotly-plot, .plot-container {{
        background: transparent !important;
    }}
    /* Hide default header */
    header[data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)


# ── Hero & Search ──────────────────────────────────────────────────────────────

def render_hero(lang: str, llm_on: bool):
    st.markdown(f"""
    <div class="cm-hero">
      <div class="cm-hero-title">🎬 CineMatch AI</div>
      <div class="cm-hero-sub">{t('app_subtitle', lang)}</div>
      <div class="cm-badge-row">
        <span class="cm-badge cm-badge-accent" title="{'LLM-powered' if llm_on else 'Running without LLM API — recommendations still work'}">
          {'🟢 LLM Active' if llm_on else '⚫ No LLM API (recommendations still work)'}
        </span>
        <span class="cm-badge" title="Number of TV shows and movies in the database">🗄️ 11,013 titles</span>
        <span class="cm-badge" title="TMDb, Disney+, IMDb Top-10k, IMDb Top-5000">📂 4 data sources</span>
        <span class="cm-badge" title="Jaccard + Cosine-numeric + Cosine-text combined">🧠 Hybrid AI engine</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


EXAMPLE_QUERIES = {
    "en": [
        "Like Breaking Bad but shorter",
        "Funny office comedy",
        "Dark sci-fi thriller",
        "Something like Game of Thrones",
    ],
    "he": [
        "כמו Breaking Bad אבל קצר יותר",
        "קומדיה משרדית מצחיקה",
        "מותחן סייפיי אפל",
        "משהו כמו Game of Thrones",
    ],
}


def render_example_chips(lang: str):
    queries = EXAMPLE_QUERIES.get(lang, EXAMPLE_QUERIES["en"])
    cols = st.columns(len(queries))
    picked = None
    for i, q in enumerate(queries):
        with cols[i]:
            if st.button(f"💡 {q}", key=f"chip_{i}", use_container_width=True):
                picked = q
    return picked


# ── Search execution ──────────────────────────────────────────────────────────

def run_search(query: str, catalog, embeddings, numeric_matrix, encoder, threshold, lang):
    from agent.llm import parse_intent, explain_recommendations
    from engine.hybrid import recommend
    from engine.anomaly import is_anomalous

    intent = parse_intent(query)
    detected_lang = intent.get("lang", lang)
    seeds = intent.get("seeds", [])

    if not seeds:
        for title in catalog["title"].tolist():
            if title and len(title) > 3 and title.lower() in query.lower():
                seeds.append(title)
                break

    query_emb = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    if seeds:
        results_df = recommend(
            query_title=seeds[0], catalog=catalog,
            numeric_matrix=numeric_matrix, embeddings=embeddings,
            query_embedding=query_emb, top_n=5,
            exclude_titles=set(seeds),
            query_lang=detected_lang,
        )
    else:
        from engine.cosine import top_k_cosine_text
        results_df = top_k_cosine_text(query_emb, catalog, embeddings, k=5)
        if not results_df.empty:
            results_df["jaccard_score"] = 0.0
            results_df["cosine_numeric_score"] = 0.0
            results_df["hybrid_score"] = results_df.get("cosine_text_score", 0.0)

    anomalous = False
    if not results_df.empty:
        best_score = float(results_df["hybrid_score"].iloc[0]) if "hybrid_score" in results_df else 0.0
        anomalous = is_anomalous(best_score, threshold)

    recs = results_df.to_dict("records") if not results_df.empty else []
    explanation = explain_recommendations(intent, recs, lang=detected_lang)
    return recs, explanation, anomalous, detected_lang


# ── Result rendering ──────────────────────────────────────────────────────────

def render_result(rec: dict, rank: int, lang: str):
    is_rtl   = lang == "he"
    dir_attr = "rtl" if is_rtl else "ltr"
    flex_dir = "row-reverse" if is_rtl else "row"
    t_align  = "right" if is_rtl else "left"

    poster = rec.get("poster_path", "")
    poster_url = f"https://image.tmdb.org/t/p/w154{poster}" if poster.startswith("/") else ""
    poster_html = (
        f'<img src="{poster_url}" style="width:90px;border-radius:10px;display:block;" '
        f'onerror="this.parentNode.innerHTML=\'🎬\'">'
        if poster_url else
        f'<div style="width:90px;height:135px;background:{BRAND["bg_card_alt"]};border-radius:10px;'
        f'display:flex;align-items:center;justify-content:center;font-size:2rem;">🎬</div>'
    )

    hybrid = float(rec.get("hybrid_score", rec.get("cosine_text_score", 0)) or 0)
    j  = float(rec.get("jaccard_score", 0) or 0)
    n  = float(rec.get("cosine_numeric_score", 0) or 0)
    tx = float(rec.get("cosine_text_score", 0) or 0)

    rating = rec.get("rating", "N/A")
    rating_str = f"{rating:.1f}" if isinstance(rating, (int, float)) else str(rating)
    decade = rec.get("decade_str", "")
    genres = rec.get("genres", "")
    ov     = (rec.get("overview", "") or "")[:450]
    title  = rec.get("title", "")

    def bar(val, color):
        w = max(0, min(1, val)) * 100
        return (
            f'<div style="background:rgba(255,255,255,0.07);border-radius:6px;height:6px;'
            f'overflow:hidden;margin:3px 0 10px;">'
            f'<div style="width:{w:.0f}%;height:100%;background:{color};border-radius:6px;"></div></div>'
        )

    score_lbl  = "התאמה" if is_rtl else "Match Score"
    genre_lbl  = "ז'אנר"  if is_rtl else "Genre"
    prof_lbl   = "פרופיל" if is_rtl else "Profile"
    plot_lbl   = "עלילה"  if is_rtl else "Plot"

    plot_html = (
        f'<details style="margin-top:0.6rem;">'
        f'<summary style="cursor:pointer;color:{BRAND["text_muted"]};font-size:0.83rem;'
        f'list-style:none;user-select:none;">📖 {plot_lbl}</summary>'
        f'<p style="margin-top:0.5rem;color:{BRAND["text"]};font-size:0.88rem;'
        f'line-height:1.65;direction:{dir_attr};text-align:{t_align};">'
        f'{ov}{"…" if len(ov) >= 450 else ""}</p></details>'
        if ov and len(ov) > 20 else ""
    )

    award_prob = rec.get("award_probability", None)
    award_html = ""
    if award_prob is not None:
        try:
            p = float(award_prob)
            if p >= 0.75:
                lbl = "מועמד חזק לפרס" if is_rtl else "Strong Award Candidate"
                award_html = f'<span class="cm-badge cm-badge-accent" style="margin:0 0.4rem;">🏆 {lbl} {p:.0%}</span>'
            elif p >= 0.50:
                lbl = "מועמד אפשרי" if is_rtl else "Possible Nominee"
                award_html = f'<span class="cm-badge" style="margin:0 0.4rem;">⭐ {lbl} {p:.0%}</span>'
        except (TypeError, ValueError):
            pass

    html = (
        f'<div class="cm-result" dir="{dir_attr}" style="text-align:{t_align};">'
        f'<div style="display:flex;gap:1rem;align-items:flex-start;flex-direction:{flex_dir};">'
        f'<div style="flex:0 0 auto;">{poster_html}</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="display:flex;align-items:center;gap:0.5rem;flex-direction:{flex_dir};margin-bottom:0.2rem;">'
        f'<span class="cm-rank">{rank}</span>'
        f'<span style="font-size:1.25rem;font-weight:700;color:{BRAND["text"]};">{title}</span>'
        f'{award_html}</div>'
        f'<div class="cm-meta">⭐ {rating_str} · {decade} · {genres}</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;color:{BRAND["text_muted"]};margin-top:0.6rem;">'
        f'<span><strong>{score_lbl}</strong></span>'
        f'<span style="color:{BRAND["accent"]};font-weight:700;">{hybrid:.0%}</span></div>'
        f'{bar(hybrid, BRAND["accent"])}'
        f'<div style="display:flex;gap:0.75rem;flex-direction:{flex_dir};">'
        f'<div style="flex:1;"><div style="display:flex;justify-content:space-between;font-size:0.72rem;color:{BRAND["text_muted"]};">'
        f'<span>{genre_lbl}</span><span style="color:{BRAND["accent"]};font-weight:600;">{j:.0%}</span></div>{bar(j, BRAND["accent"])}</div>'
        f'<div style="flex:1;"><div style="display:flex;justify-content:space-between;font-size:0.72rem;color:{BRAND["text_muted"]};">'
        f'<span>{prof_lbl}</span><span style="color:{BRAND["accent_2"]};font-weight:600;">{n:.0%}</span></div>{bar(n, BRAND["accent_2"])}</div>'
        f'<div style="flex:1;"><div style="display:flex;justify-content:space-between;font-size:0.72rem;color:{BRAND["text_muted"]};">'
        f'<span>{plot_lbl}</span><span style="color:{BRAND["accent_warm"]};font-weight:600;">{tx:.0%}</span></div>{bar(tx, BRAND["accent_warm"])}</div>'
        f'</div>'
        f'{plot_html}'
        f'</div></div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Research tab ──────────────────────────────────────────────────────────────

def render_research_tab(catalog, log, eval_res, trends, lang):
    st.markdown(f'<div class="cm-section-h">📊 {t("research_title", lang)}</div>', unsafe_allow_html=True)
    st.caption(t("research_subtitle", lang, n=len(catalog)))

    # ── Headline KPIs
    headline = trends.get("headline", {}) if trends else {}
    if headline:
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown(f'<div class="cm-stat"><div class="cm-stat-value">{headline.get("best_decade","-")}</div><div class="cm-stat-label">Best Decade (avg rating)</div></div>', unsafe_allow_html=True)
        with k2: st.markdown(f'<div class="cm-stat"><div class="cm-stat-value">{headline.get("top_rising_genre","-")}</div><div class="cm-stat-label">Top Rising Genre 🔥</div></div>', unsafe_allow_html=True)
        with k3: st.markdown(f'<div class="cm-stat"><div class="cm-stat-value">{headline.get("top_declining_genre","-")}</div><div class="cm-stat-label">Top Declining Genre ❄️</div></div>', unsafe_allow_html=True)
        with k4: st.markdown(f'<div class="cm-stat"><div class="cm-stat-value">{headline.get("rating_delta","-")}</div><div class="cm-stat-label">Best vs Worst (rating Δ)</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Decade rating trend
    if trends and "decade_stats" in trends:
        ds = pd.DataFrame(trends["decade_stats"])
        ds = ds.sort_values("decade_str")
        fig = px.line(ds, x="decade_str", y="avg_rating",
                      markers=True, title=t("avg_rating_decade", lang))
        fig.update_traces(line=dict(color=BRAND["accent"], width=3),
                          marker=dict(size=10, color=BRAND["accent_2"]))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=BRAND["text"]),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(t=50, b=30, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Rising vs declining genres
    if trends and "top_rising_genres" in trends:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="cm-section-h">🔥 Rising Genres</div>', unsafe_allow_html=True)
            rising = trends["top_rising_genres"][:6]
            if rising:
                df_r = pd.DataFrame([{"Genre": g["genre"], "Rise Ratio": g["rise_ratio"]} for g in rising])
                fig_r = px.bar(df_r, x="Rise Ratio", y="Genre", orientation="h",
                               color="Rise Ratio", color_continuous_scale=["#4fc3f7","#7c4dff","#ff6b9d"])
                fig_r.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=BRAND["text"]),
                    yaxis=dict(categoryorder="total ascending", gridcolor="rgba(255,255,255,0.05)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    coloraxis_showscale=False, margin=dict(t=20, b=30, l=10, r=10),
                )
                st.plotly_chart(fig_r, use_container_width=True)
        with col2:
            st.markdown('<div class="cm-section-h">❄️ Declining Genres</div>', unsafe_allow_html=True)
            declining = trends["top_declining_genres"][:6]
            if declining:
                df_d = pd.DataFrame([{"Genre": g["genre"], "Rise Ratio": g["rise_ratio"]} for g in declining])
                fig_d = px.bar(df_d, x="Rise Ratio", y="Genre", orientation="h",
                               color="Rise Ratio", color_continuous_scale=["#fbbf24","#8b93a8"])
                fig_d.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=BRAND["text"]),
                    yaxis=dict(categoryorder="total descending", gridcolor="rgba(255,255,255,0.05)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    coloraxis_showscale=False, margin=dict(t=20, b=30, l=10, r=10),
                )
                st.plotly_chart(fig_d, use_container_width=True)

    # ── Genre diversity over time
    if trends and "diversity_per_decade" in trends:
        dpd = trends["diversity_per_decade"]
        df_div = pd.DataFrame([
            {"Decade": d, "Shannon Entropy": v["shannon_entropy"], "Unique Genres": v["unique_genres"]}
            for d, v in dpd.items() if d != "Unknown"
        ]).sort_values("Decade")
        fig_div = px.area(df_div, x="Decade", y="Shannon Entropy",
                          title="Genre Diversity Over Decades (Shannon Entropy)")
        fig_div.update_traces(line=dict(color=BRAND["accent_warm"], width=2),
                              fillcolor="rgba(255,107,157,0.2)")
        fig_div.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=BRAND["text"]),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(t=50, b=30, l=30, r=30),
        )
        st.plotly_chart(fig_div, use_container_width=True)

    # ── Data sources
    if log:
        st.markdown(f'<div class="cm-section-h">📂 {t("data_sources", lang)}</div>', unsafe_allow_html=True)
        source_names = {
            "tmdb_tvs":    "TMDb TV Shows",
            "disney_plus": "Disney+ (OMDb enriched)",
            "imdb_all":    "IMDb Top-10k Scrape",
            "imdb_top5000":"IMDb Top-5000 TV Series",
        }
        cols = st.columns(4)
        for i, (src, info) in enumerate(log.get("sources", {}).items()):
            with cols[i % 4]:
                pct = round(info["clean"] / info["raw"] * 100, 1) if info["raw"] else 0
                st.markdown(
                    f'<div class="cm-stat">'
                    f'<div class="cm-stat-value">{info["clean"]:,}</div>'
                    f'<div class="cm-stat-label">{source_names.get(src, src)}<br>'
                    f'<span style="font-size:0.7rem;opacity:0.6;">from {info["raw"]:,} raw ({pct}%)</span></div>'
                    f'</div>', unsafe_allow_html=True
                )
        st.markdown(
            f'<div style="text-align:center;margin-top:1rem;color:{BRAND["text_muted"]};">'
            f'Total: <strong>{log.get("total_raw",0):,}</strong> raw rows → '
            f'<strong>{log.get("total_clean",0):,}</strong> unique titles '
            f'(<strong>{log.get("drop_rate_overall",0)}%</strong> reduction)'
            f'</div>', unsafe_allow_html=True
        )

    # ── Algorithm comparison
    if eval_res:
        st.markdown('<div class="cm-section-h">🔬 Algorithm Comparison (incl. baselines)</div>', unsafe_allow_html=True)
        prec = eval_res.get("precision_at_5", {})
        if prec:
            order = ["Baseline-Random","Baseline-Popularity","Baseline-TFIDF",
                     "Jaccard","Cosine-numeric","Cosine-text","Hybrid (chosen)"]
            order = [k for k in order if k in prec]
            df_pr = pd.DataFrame({"Method": order, "Precision@5": [prec[k] for k in order]})
            fig_pr = px.bar(df_pr, x="Method", y="Precision@5",
                            color="Precision@5", color_continuous_scale=["#3a4262","#4fc3f7","#7c4dff"],
                            title="Precision@5 — Our Methods vs Existing Baselines")
            fig_pr.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=BRAND["text"]),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat=".0%"),
                coloraxis_showscale=False,
                margin=dict(t=50, b=80, l=30, r=30),
            )
            st.plotly_chart(fig_pr, use_container_width=True)
            st.caption(f"Anomaly threshold (5th percentile): **{eval_res.get('anomaly_threshold_5pct','')}**")

    # ── Award prediction model — Top 10 contenders
    awards_preds_path = BASE / "data" / "award_predictions.parquet"
    awards_eval_path = BASE / "data" / "award_model_eval.json"
    if awards_preds_path.exists():
        try:
            ap = pd.read_parquet(awards_preds_path).sort_values(
                "award_probability", ascending=False).head(10)
            st.markdown('<div class="cm-section-h">🏆 Top 10 Predicted Award Contenders</div>',
                        unsafe_allow_html=True)
            if awards_eval_path.exists():
                rep = json.load(open(awards_eval_path))
                best = rep["models"][rep["best_model"]]
                st.caption(
                    f"Supervised model · best: **{rep['best_model']}** · "
                    f"CV-AUC = {rep['best_cv_auc']:.3f} · "
                    f"Test AUC = {best['test_roc_auc']:.3f} · "
                    f"P@10 = {best['precision_at_10']:.2f}"
                )
            fig_a = px.bar(ap[::-1], x="award_probability", y="title", orientation="h",
                           color="award_probability",
                           color_continuous_scale=["#4fc3f7","#7c4dff","#ff6b9d"],
                           range_x=[0, 1])
            fig_a.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=BRAND["text"]),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat=".0%"),
                coloraxis_showscale=False,
                margin=dict(t=20, b=30, l=10, r=10),
            )
            st.plotly_chart(fig_a, use_container_width=True)
        except Exception:
            pass


# ── About tab ─────────────────────────────────────────────────────────────────

def render_about_tab(catalog, lang):
    st.markdown(f'<div class="cm-section-h">ℹ️ {t("about_title", lang)}</div>', unsafe_allow_html=True)
    st.markdown(t("about_text", lang, n=len(catalog)))

    st.markdown('<div class="cm-section-h">🧠 Architecture</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{BRAND['bg_card']};border:1px solid {BRAND['border']};
                border-radius:14px;padding:1.5rem;font-family:'SF Mono','Consolas',monospace;
                font-size:0.85rem;line-height:1.8;color:{BRAND['text']};">
      <div>📝 <strong>User Query</strong> (Hebrew or English, free text)</div>
      <div style="margin-left:1em;color:{BRAND['text_muted']};">↓</div>
      <div>🤖 <strong>LLM</strong> — intent parser (prompt-cached)</div>
      <div style="margin-left:1em;color:{BRAND['text_muted']};">↓ {{seeds, mood, lang}}</div>
      <div>🔍 <strong>Three parallel similarity engines:</strong></div>
      <div style="margin-left:2em;color:{BRAND['accent']};">• Jaccard — discrete genres ∪ decade ∪ language</div>
      <div style="margin-left:2em;color:{BRAND['accent_2']};">• Cosine-numeric — rating, votes, year, popularity</div>
      <div style="margin-left:2em;color:{BRAND['accent_warm']};">• Cosine-text — 384-dim multilingual embeddings</div>
      <div style="margin-left:1em;color:{BRAND['text_muted']};">↓</div>
      <div>⚖️ <strong>Hybrid scorer</strong> — α·J + β·C_num + γ·C_text</div>
      <div style="margin-left:1em;color:{BRAND['text_muted']};">↓</div>
      <div>🚨 <strong>Anomaly check</strong> — if best score &lt; 5th-percentile → graceful fallback</div>
      <div style="margin-left:1em;color:{BRAND['text_muted']};">↓</div>
      <div>💬 <strong>LLM</strong> — bilingual explanation generator</div>
      <div style="margin-left:1em;color:{BRAND['text_muted']};">↓</div>
      <div>🎬 <strong>Top-5 recommendations</strong> + natural-language explanation</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    if "history" not in st.session_state:
        st.session_state.history = []
    if "preset_query" not in st.session_state:
        st.session_state.preset_query = ""

    lang = st.session_state.lang
    inject_css(lang)

    # ── Top bar with language toggle
    top_l, top_r = st.columns([5, 1])
    with top_r:
        lang_choice = st.selectbox(
            "🌐", ["English", "עברית"],
            index=0 if lang == "en" else 1,
            label_visibility="collapsed",
        )
        new_lang = "he" if lang_choice == "עברית" else "en"
        if new_lang != lang:
            st.session_state.lang = new_lang
            st.rerun()

    # ── Resource check
    if not CATALOG_PATH.exists() or not EMB_PATH.exists():
        st.error("Data files missing. Run `python data/load_catalog.py` and `python data/embed.py` first.")
        st.stop()

    catalog   = load_catalog()
    embeddings = load_embeddings()
    numeric_matrix, threshold = load_matrices(catalog)
    encoder   = load_encoder()
    log       = _json(LOG_PATH)
    eval_res  = _json(EVAL_PATH)
    trends    = _json(TRENDS_PATH)
    llm_on    = llm_agent._get_client() is not None

    render_hero(lang, llm_on)

    tab1, tab2, tab3 = st.tabs([
        t("tab_chat", lang),
        t("tab_research", lang),
        t("tab_about", lang),
    ])

    with tab1:
        # Search input
        default_q = st.session_state.preset_query
        st.session_state.preset_query = ""
        st.caption(t("query_label", lang))
        query = st.text_input(
            t("query_label", lang),
            value=default_q,
            placeholder=t("query_placeholder", lang),
            key="query_input",
            help="",
            label_visibility="collapsed",
        )
        c_left, c_btn, c_right = st.columns([2.5, 2, 2.5])
        with c_btn:
            search_clicked = st.button(t("search_btn", lang), type="primary", use_container_width=True)

        # Example chips
        if not query and not search_clicked:
            st.caption(("💡 דוגמאות:" if lang == "he" else "💡 Try one of these:"))
            picked = render_example_chips(lang)
            if picked:
                st.session_state.preset_query = picked
                st.rerun()

        if search_clicked and query.strip():
            with st.spinner("🤖 " + ("המודל חושב..." if lang == "he" else "Thinking...")):
                recs, explanation, anomalous, detected_lang = run_search(
                    query, catalog, embeddings, numeric_matrix, encoder, threshold, lang)
            st.session_state.detected_lang = detected_lang

            dlang = detected_lang  # use query language for ALL result display
            dlang_dir = "rtl" if dlang == "he" else "ltr"

            if anomalous:
                st.markdown(f'<div class="cm-anomaly" dir="{dlang_dir}">{t("anomaly_warning", dlang)}</div>',
                            unsafe_allow_html=True)

            if not recs:
                st.info(t("no_results", dlang))
            else:
                st.markdown(
                    f'<div class="cm-explanation" dir="{dlang_dir}" '
                    f'style="text-align:{"right" if dlang=="he" else "left"};">💬 {explanation}</div>',
                    unsafe_allow_html=True)
                results_title = "ההמלצות שלנו" if dlang == "he" else "Top Recommendations"
                st.markdown(f'<div class="cm-section-h">🎬 {results_title}</div>',
                            unsafe_allow_html=True)
                for i, rec in enumerate(recs, 1):
                    render_result(rec, i, dlang)

                st.session_state.history.append({
                    "query": query, "recs": [r["title"] for r in recs],
                })

        if st.session_state.history:
            recent_label = "🕓 חיפושים אחרונים" if lang == "he" else "🕓 Recent searches"
            with st.expander(recent_label):
                for h in reversed(st.session_state.history[-5:]):
                    is_he = bool(__import__("re").search(r"[֐-׿]", h["query"]))
                    d = "rtl" if is_he else "ltr"
                    ta = "right" if is_he else "left"
                    st.markdown(
                        f'<div style="direction:{d};text-align:{ta};padding:4px 0;border-bottom:1px solid #252b3d;">'
                        f'<strong>{h["query"]}</strong> → {", ".join(h["recs"])}</div>',
                        unsafe_allow_html=True
                    )

    with tab2:
        render_research_tab(catalog, log, eval_res, trends, lang)

    with tab3:
        render_about_tab(catalog, lang)


if __name__ == "__main__":
    main()
