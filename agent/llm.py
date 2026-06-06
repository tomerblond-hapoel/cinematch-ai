"""
CineMatch AI — LLM Agent
Priority: GROQ_API_KEY → ANTHROPIC_API_KEY → offline regex fallback.
"""

import os, re, json
from typing import Optional

# ── Client state ───────────────────────────────────────────────────────────────

_groq_client = None
_anthropic_client = None
_provider = None  # "groq" | "anthropic" | None


def _read_secret(key: str) -> Optional[str]:
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key)


def _get_client():
    global _groq_client, _anthropic_client, _provider
    if _provider is not None:
        return True

    # Primary: Groq
    groq_key = _read_secret("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=groq_key)
            _provider = "groq"
            return True
        except Exception:
            pass

    # Fallback: Anthropic
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
  "seeds": ["Title 1", "Title 2"],   // shows the user explicitly mentions as reference points
  "mood": ["dark", "funny", ...],    // mood/tone tags (English, lowercase)
  "length_pref": "short|long|any",  // shorter = short, longer = long, otherwise any
  "exclude_genres": ["Reality-TV"], // genres to exclude
  "lang": "he" | "en",              // detected query language
  "free_text": "..."                 // original query for display
}

Rules:
- seeds: ONLY titles the user explicitly mentions in their message. Do NOT add titles from your own knowledge.
- mood: infer from adjectives ("dark"→dark, "מצחיק"→funny, "מרגש"→emotional, "אפל"→dark, etc.)
- length_pref: "short" if user says "short/קצר/less episodes", "long" if "long/ארוך/epic"
- exclude_genres: only if user explicitly says "not X" or "without Y"
- lang: "he" if query contains Hebrew characters, else "en"
- If a field cannot be determined, use null
"""

_EXPLAINER_SYSTEM = """\
You are a bilingual TV/movie recommendation assistant.
Your ONLY job is to explain WHY the shows in the provided list match the user's query.
CRITICAL RULES:
- Do NOT suggest, mention, or reference any show that is not in the provided recommendations list.
- Do NOT add shows from your own knowledge. Only work with the exact list given to you.
- Do NOT say "you might also like X" or recommend anything beyond the list.
- When lang is "he": reply entirely in Hebrew.
- When lang is "en": reply entirely in English.
- Be concise: one warm opening sentence, then one short sentence per show explaining why it fits.
"""


# ── LLM call helper ────────────────────────────────────────────────────────────

def _call_llm(system: str, user: str, max_tokens: int = 600) -> Optional[str]:
    if _provider == "groq":
        try:
            response = _groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception:
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
        f"Detected mood: {intent.get('mood',[])}, Language: {lang}\n\n"
        f"The recommendation engine found EXACTLY these {len(recommendations)} shows from our database:\n"
        f"{recs_text}\n\n"
        f"Explain ONLY these shows and why they match the query. "
        f"Do not mention any other shows. Reply in {'Hebrew' if lang=='he' else 'English'}."
    )

    result = _call_llm(_EXPLAINER_SYSTEM, user_msg, max_tokens=600)
    return result if result else _fallback_explanation(intent, recommendations, lang)


def _fallback_explanation(intent: dict, recommendations: list[dict], lang: str) -> str:
    mood = intent.get("mood", [])
    seeds = intent.get("seeds", [])

    if lang == "he":
        if seeds:
            opener = f"מצאנו עבורך סדרות הדומות ל-{seeds[0]}:"
        elif mood:
            mood_str = ", ".join(mood)
            opener = f"על פי מה שחיפשת ({mood_str}), אלו ההמלצות המתאימות ביותר:"
        else:
            opener = "אלו ההמלצות המובילות שלנו עבורך:"
        lines = [opener]
        for r in recommendations:
            rating = r.get("rating", "")
            rating_str = f"{rating:.1f}" if isinstance(rating, float) else str(rating)
            lines.append(f"• {r['title']} — {r.get('genres','')} | ⭐ {rating_str}")
    else:
        if seeds:
            opener = f"Based on your interest in {seeds[0]}, here are the best matches:"
        elif mood:
            mood_str = ", ".join(mood)
            opener = f"Looking for something {mood_str}? Here are our top picks:"
        else:
            opener = "Here are our top recommendations for you:"
        lines = [opener]
        for r in recommendations:
            rating = r.get("rating", "")
            rating_str = f"{rating:.1f}" if isinstance(rating, float) else str(rating)
            lines.append(f"• {r['title']} — {r.get('genres','')} | ⭐ {rating_str}")
    return "\n".join(lines)
