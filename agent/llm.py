"""
CineMatch AI — LLM Agent
Supports Gemini (primary, free) and Claude (fallback).
Priority: GEMINI_API_KEY → ANTHROPIC_API_KEY → offline regex fallback.
"""

import os, re, json
from typing import Optional

# ── Client state ───────────────────────────────────────────────────────────────

_gemini_model = None
_anthropic_client = None
_provider = None  # "gemini" | "anthropic" | None


def _read_secret(key: str) -> Optional[str]:
    """Read a secret from st.secrets (Streamlit Cloud) or os.environ (local)."""
    # 1. Try st.secrets first (Streamlit Cloud & local .streamlit/secrets.toml)
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    # 2. Fallback to environment variable
    return os.environ.get(key)


def _get_client():
    global _gemini_model, _anthropic_client, _provider
    if _provider is not None:
        return True

    # Try Gemini first
    gemini_key = _read_secret("GEMINI_API_KEY") or _read_secret("GOOGLE_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            _provider = "gemini"
            return True
        except Exception:
            pass

    # Fallback to Anthropic
    anthropic_key = _read_secret("ANTHROPIC_API_KEY") or _read_secret("ANTHROPIC_KEY")
    if anthropic_key:
        try:
            import anthropic
            _anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            _provider = "anthropic"
            return True
        except Exception:
            pass

    _provider = None
    return None


# ── System prompts ─────────────────────────────────────────────────────────────

_PARSER_SYSTEM = """\
You are an assistant that extracts structured intent from TV/movie recommendation queries.
The user may write in Hebrew or English. Always reply with valid JSON only.

Output schema:
{
  "seeds": ["Title 1", "Title 2"],   // shows the user mentions as reference points
  "mood": ["dark", "funny", ...],    // mood/tone tags (English, lowercase)
  "length_pref": "short|long|any",  // shorter = short, longer = long, otherwise any
  "exclude_genres": ["Reality-TV"], // genres to exclude
  "lang": "he" | "en",              // detected query language
  "free_text": "..."                 // original query for display
}

Rules:
- seeds: titles the user explicitly mentions (TV or movie names)
- mood: infer from adjectives ("dark"→dark, "מצחיק"→funny, "מרגש"→emotional, "אפל"→dark, etc.)
- length_pref: "short" if user says "short/קצר/less episodes", "long" if "long/ארוך/epic"
- exclude_genres: only if user explicitly says "not X" or "without Y"
- lang: "he" if query contains Hebrew characters, else "en"
- If a field cannot be determined, use null
"""

_EXPLAINER_SYSTEM = """\
You are a friendly bilingual TV/movie recommendation assistant.
When lang is "he": reply in Hebrew. When lang is "en": reply in English.
Be concise: 2-3 sentences total. Start with a warm opener, then one sentence per show explaining WHY it matches.
"""


# ── LLM call helper ────────────────────────────────────────────────────────────

def _call_llm(system: str, user: str, max_tokens: int = 600) -> Optional[str]:
    if _provider == "gemini":
        try:
            prompt = f"{system}\n\n{user}"
            response = _gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            # Surface error so we can debug
            import streamlit as st
            st.warning(f"⚠️ Gemini error: {type(e).__name__}: {e}")
            return None

    if _provider == "anthropic":
        try:
            response = _anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text.strip()
        except Exception:
            return None

    return None


# ── Intent parser ─────────────────────────────────────────────────────────────

def parse_intent(query: str) -> dict:
    if not _get_client():
        return _regex_parse(query)

    raw = _call_llm(_PARSER_SYSTEM, f"Query: {query}", max_tokens=512)
    if raw:
        try:
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            parsed = json.loads(raw)
            parsed.setdefault("free_text", query)
            return parsed
        except Exception:
            pass

    return _regex_parse(query)


def _regex_parse(query: str) -> dict:
    q = query.strip()
    has_hebrew = bool(re.search(r"[֐-׿]", q))
    lang = "he" if has_hebrew else "en"

    mood_map = {
        "dark":      ["dark","אפל","כהה","depressing"],
        "funny":     ["funny","comedy","fun","מצחיק","הומור","קומדי"],
        "emotional": ["emotional","sad","cry","מרגש","עצוב"],
        "thrilling": ["thriller","thrill","suspense","מותחן","מפחיד","horror"],
        "light":     ["light","lighthearted","קליל","קל","cheerful"],
    }
    mood = []
    for tag, kws in mood_map.items():
        for kw in kws:
            if kw.lower() in q.lower():
                mood.append(tag)
                break

    if re.search(r"\b(short|קצר|פחות|fewer)\b", q, re.IGNORECASE):
        length_pref = "short"
    elif re.search(r"\b(long|ארוך|epic|longer)\b", q, re.IGNORECASE):
        length_pref = "long"
    else:
        length_pref = "any"

    return {
        "seeds": [], "mood": mood, "length_pref": length_pref,
        "exclude_genres": [], "lang": lang, "free_text": query,
    }


# ── Explanation generator ──────────────────────────────────────────────────────

def explain_recommendations(intent: dict, recommendations: list[dict], lang: str = "en") -> str:
    if not recommendations:
        return "לא נמצאו תוצאות מתאימות." if lang == "he" else "No matching results found."

    if not _get_client():
        return _fallback_explanation(intent, recommendations, lang)

    recs_text = "\n".join(
        f"{i+1}. {r['title']} ({r.get('decade_str','')}, {r.get('genres','')}) "
        f"— Rating: {r.get('rating','')} — Hybrid score: {r.get('hybrid_score','')}"
        for i, r in enumerate(recommendations)
    )

    user_msg = (
        f"User query: {intent.get('free_text','')}\n"
        f"Seeds: {intent.get('seeds',[])}, Mood: {intent.get('mood',[])}, Language: {lang}\n\n"
        f"Recommendations:\n{recs_text}\n\n"
        f"Write a warm explanation in {'Hebrew' if lang=='he' else 'English'}."
    )

    result = _call_llm(_EXPLAINER_SYSTEM, user_msg, max_tokens=600)
    return result if result else _fallback_explanation(intent, recommendations, lang)


def _fallback_explanation(intent: dict, recommendations: list[dict], lang: str) -> str:
    if lang == "he":
        lines = ["הנה ההמלצות שלנו עבורך:"]
        for r in recommendations:
            lines.append(f"• {r['title']} ({r.get('genres','')}) — ציון: {r.get('rating','')}")
    else:
        lines = ["Here are your personalized recommendations:"]
        for r in recommendations:
            lines.append(f"• {r['title']} ({r.get('genres','')}) — Rating: {r.get('rating','')}")
    return "\n".join(lines)
