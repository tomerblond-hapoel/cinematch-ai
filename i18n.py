"""Bilingual string dictionary for CineMatch AI UI."""

STRINGS = {
    "en": {
        "app_title":        "CineMatch AI",
        "app_subtitle":     "Your bilingual AI-powered TV & Movie Recommender",
        "query_placeholder":"e.g. 'I just finished Breaking Bad, want something darker but shorter'",
        "query_label":      "What are you in the mood to watch?",
        "search_btn":       "🔍 Search",
        "searching":        "🔍 Searching...",
        "no_results":       "No results found. Try a different query.",
        "results_title":    "Top Recommendations",
        "rating":           "Rating",
        "genres":           "Genres",
        "decade":           "Decade",
        "score":            "Match Score",
        "jaccard":          "Genre Match",
        "num_cosine":       "Profile Match",
        "text_cosine":      "Plot Match",
        "anomaly_warning":  "Your query is quite unusual — no strong matches found. Try mentioning a specific show.",
        "tab_chat":         "💬 Recommend",
        "tab_research":     "📊 Research",
        "tab_about":        "ℹ️ About",
        "research_title":   "Data & Trend Analysis",
        "research_subtitle":"Patterns across {n} TV shows from 4 sources",
        "avg_rating_decade":"Average Rating by Decade",
        "top_genres_decade":"Top Genres by Decade",
        "data_sources":     "Data Sources",
        "about_title":      "About CineMatch AI",
        "about_text": (
            "**CineMatch AI** is a bilingual AI-powered TV & movie recommendation agent "
            "built as the final project for the AI & ML Innovation Workshop (2025).\n\n"
            "**Data:** 4 sources, 166,592 raw rows → {n} clean titles after filtering and deduplication.\n\n"
            "**Algorithm:** Hybrid of Jaccard similarity (genre/era matching), "
            "Cosine similarity on numeric features (rating, popularity, year), "
            "and Cosine similarity on multilingual plot embeddings (384-dim, sentence-transformers).\n\n"
            "**Agent:** Natural-language queries parsed by Claude claude-sonnet-4-6 with prompt caching. "
            "Explanations generated in the user's detected language (Hebrew or English).\n\n"
            "**Author:** Tomer Blond — AI & ML Innovation Workshop 2025"
        ),
        "source_row": "{name}: {raw:,} raw → {clean:,} clean ({pct}% kept)",
        "sidebar_lang":     "🌐 Language / שפה",
        "powered_by":       "Powered by Claude claude-sonnet-4-6 + sentence-transformers",
        "llm_status_on":    "✅ Claude API connected",
        "llm_status_off":   "⚡ Running in offline mode (no API key)",
    },
    "he": {
        "app_title":        "CineMatch AI",
        "app_subtitle":     "הממליץ הדו-לשוני שלך לסדרות וסרטים מבוסס בינה מלאכותית",
        "query_placeholder":"למשל: 'סיימתי את Breaking Bad, רוצה משהו אפל אבל קצר יותר'",
        "query_label":      "על מה בא לך לצפות?",
        "search_btn":       "🔍 חיפוש",
        "searching":        "🔍 מחפש...",
        "no_results":       "לא נמצאו תוצאות. נסה שאילתה שונה.",
        "results_title":    "ההמלצות המובילות",
        "rating":           "ציון",
        "genres":           "ז'אנרים",
        "decade":           "עשור",
        "score":            "ציון התאמה",
        "jaccard":          "התאמת ז'אנר",
        "num_cosine":       "התאמת פרופיל",
        "text_cosine":      "התאמת עלילה",
        "anomaly_warning":  "השאילתה שלך חריגה מאוד — לא נמצאו התאמות חזקות. נסה לציין סדרה ספציפית.",
        "tab_chat":         "💬 המלצות",
        "tab_research":     "📊 מחקר",
        "tab_about":        "ℹ️ אודות",
        "research_title":   "ניתוח נתונים ומגמות",
        "research_subtitle":"דפוסים מתוך {n} סדרות ממקורות 4",
        "avg_rating_decade":"ממוצע ציונים לפי עשור",
        "top_genres_decade":"ז'אנרים מובילים לפי עשור",
        "data_sources":     "מקורות הנתונים",
        "about_title":      "אודות CineMatch AI",
        "about_text": (
            "**CineMatch AI** הוא סוכן המלצות דו-לשוני לסדרות וסרטים, "
            "שפותח כפרויקט גמר לסדנת חדשנות מבוססת AI ו-ML (2025).\n\n"
            "**נתונים:** 4 מקורות, 166,592 שורות גולמיות → {n} כותרות נקיות לאחר סינון וביטול כפילויות.\n\n"
            "**אלגוריתם:** שילוב של דמיון Jaccard (התאמת ז'אנר/עידן), "
            "דמיון Cosine על מאפיינים מספריים (ציון, פופולריות, שנה), "
            "ודמיון Cosine על embeddings רב-לשוניים של עלילות (384 ממד).\n\n"
            "**סוכן:** שאילתות בשפה טבעית מנותחות על ידי Claude claude-sonnet-4-6 עם prompt caching. "
            "הסברים נוצרים בשפה המזוהה של המשתמש (עברית או אנגלית).\n\n"
            "**מחבר:** תומר בלונד — סדנת חדשנות מבוססת AI ו-ML 2025"
        ),
        "source_row": "{name}: {raw:,} שורות גולמיות → {clean:,} נקיות ({pct}% נשמרו)",
        "sidebar_lang":     "🌐 Language / שפה",
        "powered_by":       "מופעל על ידי Claude claude-sonnet-4-6 + sentence-transformers",
        "llm_status_on":    "✅ Claude API מחובר",
        "llm_status_off":   "⚡ מצב אופליין (ללא API key)",
    },
}


def t(key: str, lang: str = "en", **kwargs) -> str:
    s = STRINGS.get(lang, STRINGS["en"]).get(key, STRINGS["en"].get(key, key))
    if kwargs:
        try:
            s = s.format(**kwargs)
        except Exception:
            pass
    return s
