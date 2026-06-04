"""
CineMatch AI — LLM Agent
Two Claude calls per query:
  1. Intent parser: NL query → structured JSON (seeds, mood, lang, length_pref, exclude)
  2. Explanation generator: recommendations → bilingual natural-language explanation

Uses Claude claude-sonnet-4-6 with prompt caching on the system prompt (5-min TTL).
Falls back to regex parser if ANTHROPIC_API_KEY is not set.
"""

import os, re, json
from typing import Optional
import anthropic

_client: Optional[anthropic.Anthropic] = None
USING_LLM = False

def _get_client():
    global _client, USING_LLM
    if _client is not None:
        return _client
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_KEY")
    if not key:
        return None
    _client = anthropic.Anthropic(api_key=key)
    USING_LLM = True
    return _client


# ── System prompts (cached) ────────────────────────────────────────────────────

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
When lang is "he": reply in Hebrew (RTL). When lang is "en": reply in English.
Be concise: 2-3 sentences per recommendation.
Start with a warm opener, then one sentence per show explaining WHY it matches.
"""


# ── Intent parser ─────────────────────────────────────────────────────────────

def parse_intent(query: str) -> dict:
    """Parse a natural-language query into structured intent."""
    client = _get_client()
    if client is None:
        return _regex_parse(query)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=[{
                "type": "text",
                "text": _PARSER_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": f"Query: {query}"}],
        )
        raw = response.content[0].text.strip()
        # strip markdown code fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw)
        parsed.setdefault("free_text", query)
        return parsed
    except Exception as e:
        return _regex_parse(query)


def _regex_parse(query: str) -> dict:
    """Offline fallback: simple regex-based intent extraction."""
    q = query.strip()
    has_hebrew = bool(re.search(r"[֐-׿]", q))
    lang = "he" if has_hebrew else "en"

    # Detect mood words
    mood_map = {
        "dark": ["dark","אפל","כהה","dark","depressing"],
        "funny": ["funny","comedy","fun","מצחיק","הומור","קומדי"],
        "emotional": ["emotional","sad","cry","מרגש","עצוב","מדהים"],
        "thrilling": ["thriller","thrill","suspense","מותחן","מפחיד","horror"],
        "light": ["light","lighthearted","קליל","קל","cheerful"],
    }
    mood = []
    for tag, kws in mood_map.items():
        for kw in kws:
            if kw.lower() in q.lower():
                mood.append(tag)
                break

    # length pref
    if re.search(r"\b(short|קצר|פחות|fewer)\b", q, re.IGNORECASE):
        length_pref = "short"
    elif re.search(r"\b(long|ארוך|epic|longer)\b", q, re.IGNORECASE):
        length_pref = "long"
    else:
        length_pref = "any"

    return {
        "seeds": [],
        "mood": mood,
        "length_pref": length_pref,
        "exclude_genres": [],
        "lang": lang,
        "free_text": query,
    }


# ── Explanation generator ──────────────────────────────────────────────────────

def explain_recommendations(
    intent: dict,
    recommendations: list[dict],
    lang: str = "en",
) -> str:
    """Generate a bilingual explanation for the top recommendations."""
    client = _get_client()

    if not recommendations:
        if lang == "he":
            return "לא נמצאו תוצאות מתאימות לחיפוש שלך. נסה שאילתה שונה."
        return "No matching results found. Try a different query."

    if client is None:
        return _fallback_explanation(intent, recommendations, lang)

    recs_text = "\n".join(
        f"{i+1}. {r['title']} ({r.get('decade_str','')}, {r.get('genres','')}) "
        f"— Rating: {r.get('rating','')} — "
        f"Hybrid score: {r.get('hybrid_score','')} "
        f"[Jaccard={r.get('jaccard_score','')} / NumCos={r.get('cosine_numeric_score','')} / TextCos={r.get('cosine_text_score','')}]"
        for i, r in enumerate(recommendations)
    )

    user_msg = (
        f"User query: {intent.get('free_text','')}\n"
        f"Detected seeds: {intent.get('seeds',[])}\n"
        f"Mood: {intent.get('mood',[])}\n"
        f"Language: {lang}\n\n"
        f"Recommendations:\n{recs_text}\n\n"
        f"Write a warm explanation of these recommendations in {'Hebrew' if lang=='he' else 'English'}."
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=[{
                "type": "text",
                "text": _EXPLAINER_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text.strip()
    except Exception:
        return _fallback_explanation(intent, recommendations, lang)


def _fallback_explanation(intent: dict, recommendations: list[dict], lang: str) -> str:
    if lang == "he":
        lines = ["הנה ההמלצות שלנו עבורך:"]
        for r in recommendations:
            lines.append(f"• {r['title']} ({r.get('genres','')}) — ציון: {r.get('rating','')}")
        return "\n".join(lines)
    else:
        lines = ["Here are your personalized recommendations:"]
        for r in recommendations:
            lines.append(f"• {r['title']} ({r.get('genres','')}) — Rating: {r.get('rating','')}")
        return "\n".join(lines)
