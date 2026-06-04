"""
CineMatch AI — Streamlit Application
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

from i18n import t
import agent.llm as llm_agent

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load resources ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading catalog...")
def load_catalog():
    df = pd.read_parquet(CATALOG_PATH)
    return df

@st.cache_resource(show_spinner="Loading embeddings...")
def load_embeddings():
    return np.load(str(EMB_PATH))

@st.cache_resource(show_spinner="Building similarity matrices...")
def load_matrices(catalog):
    from engine.cosine import build_numeric_matrix
    from engine.anomaly import calibrate
    from engine.jaccard import batch_jaccard_matrix
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos

    num_mat = build_numeric_matrix(catalog)
    # anomaly calibration using numeric matrix best scores
    H_no_diag = num_mat.copy()
    np.fill_diagonal(H_no_diag, -1)
    best_scores = H_no_diag.max(axis=1)
    threshold = calibrate(best_scores, percentile=5)
    return num_mat, threshold

@st.cache_resource(show_spinner="Loading sentence encoder...")
def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def load_cleaning_log():
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            return json.load(f)
    return {}

def load_eval_results():
    if EVAL_PATH.exists():
        with open(EVAL_PATH) as f:
            return json.load(f)
    return {}

# ── RTL / LTR CSS injection ────────────────────────────────────────────────────

def inject_css(lang: str):
    direction = "rtl" if lang == "he" else "ltr"
    font = "Heebo, Arial, sans-serif" if lang == "he" else "Inter, sans-serif"
    st.markdown(f"""
    <link href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;600;700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {{
        font-family: {font} !important;
        direction: {direction} !important;
    }}
    .stTextInput > div > div > input {{
        direction: {direction};
        font-family: {font};
    }}
    .result-card {{
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4fc3f7;
    }}
    .score-bar {{
        background: linear-gradient(90deg, #4fc3f7, #1565c0);
        border-radius: 4px;
        height: 6px;
    }}
    </style>
    """, unsafe_allow_html=True)


# ── Search logic ───────────────────────────────────────────────────────────────

def run_search(query: str, catalog: pd.DataFrame, embeddings: np.ndarray,
               numeric_matrix: np.ndarray, encoder, threshold: float, lang: str):
    from agent.llm import parse_intent, explain_recommendations
    from engine.hybrid import recommend
    from engine.anomaly import is_anomalous

    intent = parse_intent(query)
    detected_lang = intent.get("lang", lang)
    seeds = intent.get("seeds", [])

    # If no seeds detected from LLM, try keyword match in catalog
    if not seeds:
        # look for show titles mentioned verbatim in query
        for title in catalog["title"].tolist():
            if title.lower() in query.lower() and len(title) > 3:
                seeds.append(title)
                break

    # Embed the full query text for text-cosine
    query_emb = encoder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True)[0]

    if seeds:
        # recommend using the first seed title
        results_df = recommend(
            query_title=seeds[0],
            catalog=catalog,
            numeric_matrix=numeric_matrix,
            embeddings=embeddings,
            query_embedding=query_emb,
            top_n=5,
            exclude_titles=set(seeds),
        )
    else:
        # no seed → pure text-cosine search
        from engine.cosine import top_k_cosine_text
        results_df = top_k_cosine_text(
            query_embedding=query_emb,
            catalog=catalog,
            embeddings=embeddings,
            k=5,
        )
        if not results_df.empty:
            results_df["jaccard_score"] = 0.0
            results_df["cosine_numeric_score"] = 0.0
            results_df["hybrid_score"] = results_df.get("cosine_text_score", 0.0)

    # Anomaly check
    anomalous = False
    if not results_df.empty:
        best_score = float(results_df["hybrid_score"].iloc[0]) if "hybrid_score" in results_df else 0.0
        anomalous = is_anomalous(best_score, threshold)

    recs = results_df.to_dict("records") if not results_df.empty else []
    explanation = explain_recommendations(intent, recs, lang=detected_lang)

    return recs, explanation, anomalous, detected_lang


# ── UI Components ──────────────────────────────────────────────────────────────

def render_result_card(rec: dict, rank: int, lang: str):
    poster = rec.get("poster_path", "")
    poster_url = f"https://image.tmdb.org/t/p/w92{poster}" if poster.startswith("/") else ""

    with st.container():
        col_img, col_info = st.columns([1, 5])
        with col_img:
            if poster_url:
                st.image(poster_url, width=60)
            else:
                st.markdown("🎬")
        with col_info:
            st.markdown(f"### {rank}. {rec.get('title','')}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(t("rating", lang), f"⭐ {rec.get('rating','N/A')}")
            with col2:
                st.metric(t("decade", lang), rec.get("decade_str",""))
            with col3:
                hybrid = rec.get("hybrid_score", rec.get("cosine_text_score", 0))
                st.metric(t("score", lang), f"{hybrid:.2%}")

            st.caption(f"**{t('genres', lang)}:** {rec.get('genres','')}")

            # Score breakdown
            j = rec.get("jaccard_score", 0)
            n = rec.get("cosine_numeric_score", 0)
            tx = rec.get("cosine_text_score", 0)
            if j or n or tx:
                st.caption(
                    f"{t('jaccard',lang)}: {j:.2%} · "
                    f"{t('num_cosine',lang)}: {n:.2%} · "
                    f"{t('text_cosine',lang)}: {tx:.2%}"
                )

            ov = rec.get("overview","")
            if ov and len(ov) > 20:
                with st.expander("📖 Plot synopsis"):
                    st.write(ov[:400] + ("..." if len(ov) > 400 else ""))
        st.divider()


def render_research_tab(catalog: pd.DataFrame, log: dict, eval_res: dict, lang: str):
    st.subheader(t("research_title", lang))
    st.caption(t("research_subtitle", lang, n=len(catalog)))

    # ── Average rating by decade
    df_dec = (catalog
              .dropna(subset=["decade","rating"])
              .groupby("decade_str")["rating"]
              .agg(["mean","count"])
              .reset_index()
              .rename(columns={"mean":"avg_rating","count":"n_shows"}))
    df_dec = df_dec[df_dec["n_shows"] >= 5].sort_values("decade_str")

    fig1 = px.bar(df_dec, x="decade_str", y="avg_rating",
                  text="n_shows",
                  labels={"decade_str":"Decade","avg_rating":"Avg Rating","n_shows":"# Shows"},
                  color="avg_rating",
                  color_continuous_scale="Blues",
                  title=t("avg_rating_decade", lang))
    fig1.update_traces(texttemplate="%{text} shows", textposition="outside")
    fig1.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig1, use_container_width=True)

    # ── Top genres overall
    genre_counts: dict = {}
    for g_str in catalog["genres"].dropna():
        for g in g_str.split(","):
            g = g.strip()
            if g:
                genre_counts[g] = genre_counts.get(g, 0) + 1
    top_genres = sorted(genre_counts.items(), key=lambda x: -x[1])[:15]
    df_genres = pd.DataFrame(top_genres, columns=["Genre","Count"])
    fig2 = px.bar(df_genres, x="Count", y="Genre", orientation="h",
                  color="Count", color_continuous_scale="Teal",
                  title=t("top_genres_decade", lang))
    fig2.update_layout(yaxis={"categoryorder":"total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Cleaning log
    if log:
        st.subheader(t("data_sources", lang))
        source_names = {
            "tmdb_tvs":    "TMDb TV Shows",
            "disney_plus": "Disney+ (OMDb enriched)",
            "imdb_all":    "IMDb Top-10k Scrape",
            "imdb_top5000":"IMDb Top-5000 TV Series",
        }
        for src, info in log.get("sources", {}).items():
            pct = round(info["clean"] / info["raw"] * 100, 1) if info["raw"] else 0
            st.write(t("source_row", lang,
                       name=source_names.get(src, src),
                       raw=info["raw"], clean=info["clean"], pct=pct))
        st.write(f"**Total raw:** {log.get('total_raw',0):,} rows  →  "
                 f"**After dedupe:** {log.get('total_clean',0):,} unique titles "
                 f"({log.get('drop_rate_overall',0)}% reduction)")

    # ── Evaluation results
    if eval_res:
        st.subheader("Algorithm Comparison")
        spearman = eval_res.get("spearman", {})
        prec = eval_res.get("precision_at_5", {})
        agr  = eval_res.get("agreement_at_10", {})

        st.write("**Spearman rank correlation between methods:**")
        sp_df = pd.DataFrame(list(spearman.items()), columns=["Method Pair","Spearman ρ"])
        st.dataframe(sp_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Precision@5 on preference set:**")
            pr_df = pd.DataFrame(list(prec.items()), columns=["Method","Precision@5"])
            st.dataframe(pr_df, use_container_width=True, hide_index=True)
        with col2:
            st.write("**Agreement@10 between methods:**")
            ag_df = pd.DataFrame(list(agr.items()), columns=["Method Pair","Agreement@10"])
            st.dataframe(ag_df, use_container_width=True, hide_index=True)

        st.info(f"Anomaly threshold (5th percentile of best-match scores): "
                f"**{eval_res.get('anomaly_threshold_5pct','')}**")


def render_about_tab(catalog: pd.DataFrame, lang: str):
    st.subheader(t("about_title", lang))
    st.markdown(t("about_text", lang, n=len(catalog)))


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    # ── Session state
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    if "history" not in st.session_state:
        st.session_state.history = []

    lang = st.session_state.lang
    inject_css(lang)

    # ── Sidebar
    with st.sidebar:
        st.title("🎬 CineMatch AI")
        lang_choice = st.radio(
            t("sidebar_lang", lang),
            ["English", "עברית"],
            index=0 if lang == "en" else 1,
        )
        new_lang = "he" if lang_choice == "עברית" else "en"
        if new_lang != lang:
            st.session_state.lang = new_lang
            st.rerun()

        st.divider()
        client = llm_agent._get_client()
        if client:
            st.success(t("llm_status_on", lang))
        else:
            st.warning(t("llm_status_off", lang))
            st.caption("Set ANTHROPIC_API_KEY in .env to enable Claude.")

        st.caption(t("powered_by", lang))

    # ── Load data
    if not CATALOG_PATH.exists():
        st.error("❌ catalog.parquet not found. Run `python data/load_catalog.py` first.")
        st.stop()
    if not EMB_PATH.exists():
        st.error("❌ embeddings.npy not found. Run `python data/embed.py` first.")
        st.stop()

    catalog   = load_catalog()
    embeddings = load_embeddings()
    numeric_matrix, threshold = load_matrices(catalog)
    encoder   = load_encoder()
    log       = load_cleaning_log()
    eval_res  = load_eval_results()

    # ── Tabs
    tab1, tab2, tab3 = st.tabs([
        t("tab_chat", lang),
        t("tab_research", lang),
        t("tab_about", lang),
    ])

    # ── Tab 1: Recommender
    with tab1:
        st.title(t("app_title", lang))
        st.subheader(t("app_subtitle", lang))

        query = st.text_input(
            t("query_label", lang),
            placeholder=t("query_placeholder", lang),
            key="query_input",
        )

        search_clicked = st.button(t("search_btn", lang), type="primary")

        if search_clicked and query.strip():
            with st.spinner(t("searching", lang)):
                recs, explanation, anomalous, detected_lang = run_search(
                    query, catalog, embeddings, numeric_matrix, encoder, threshold, lang)

            if anomalous:
                st.warning(t("anomaly_warning", lang))

            if not recs:
                st.info(t("no_results", lang))
            else:
                st.subheader(t("results_title", lang))
                st.markdown(explanation)
                st.divider()
                for i, rec in enumerate(recs, 1):
                    render_result_card(rec, i, lang)

                # Save to history
                st.session_state.history.append({
                    "query": query,
                    "recs": [r["title"] for r in recs],
                })

        # Show recent history
        if st.session_state.history:
            with st.expander("🕘 Recent searches"):
                for h in reversed(st.session_state.history[-5:]):
                    st.write(f"**{h['query']}** → {', '.join(h['recs'])}")

    # ── Tab 2: Research
    with tab2:
        render_research_tab(catalog, log, eval_res, lang)

    # ── Tab 3: About
    with tab3:
        render_about_tab(catalog, lang)


if __name__ == "__main__":
    main()
