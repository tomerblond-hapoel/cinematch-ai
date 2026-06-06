"""Bilingual string dictionary for CineMatch AI UI."""

STRINGS = {
    "en": {
        "app_title":        "CineMatch AI",
        "app_subtitle":     "Your AI-powered TV & Movie Recommender",
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
        "anomaly_warning":  "⚠️ No strong match found in our dataset — the title or topic you're looking for may not be in our database. Showing the closest results we could find.",
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
            "**CineMatch AI** is a bilingual TV & movie recommender built for the AI & ML Innovation Workshop (2025).\n\n"
            "**Method:** Hybrid similarity over a cleaned catalog of {n} titles (4 sources, 166,592 raw rows) — Jaccard on discrete features, Cosine on numeric features, and Cosine on 384-dim multilingual plot embeddings. A supervised model predicts Emmy/Oscar candidacy from rating, votes, era and genre signals. An LLM parses free-text queries and writes explanations in the user's language.\n\n"
            "**Authors:** Tomer Blond & Omer Zion — AI & ML Innovation Workshop 2025"
        ),
        "source_row": "{name}: {raw:,} raw → {clean:,} clean ({pct}% kept)",
        "sidebar_lang":     "🌐 Language / שפה",
        "powered_by":       "Hybrid similarity engine + multilingual embeddings + LLM",
        "llm_status_on":    "✅ LLM connected",
        "llm_status_off":   "⚡ Running in offline mode (no API key)",
        "hero_line_1":      "YOUR NEXT OBSESSION",
        "hero_line_2":      "FINDS YOU.",
        "hero_line_3":      "🎬 CineMatch AI",
        "hero_skip":        "Skip intro →",
    },
    "he": {
        "app_title":        "CineMatch AI",
        "app_subtitle":     "הממליץ שלך לסדרות מבוסס בינה מלאכותית",
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
        "anomaly_warning":  "⚠️ לא נמצאה התאמה חזקה בבסיס הנתונים שלנו — ייתכן שהסדרה או הנושא שחיפשת אינם קיימים בדאטה. מציגים את התוצאות הקרובות ביותר שמצאנו.",
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
            "**CineMatch AI** הוא ממליץ דו-לשוני לסדרות וסרטים שפותח לסדנת חדשנות מבוססת AI ו-ML (2025).\n\n"
            "**שיטה:** דמיון היברידי על קטלוג נקי של {n} כותרות (4 מקורות, 166,592 שורות גולמיות) — Jaccard על מאפיינים בדידים, Cosine על מאפיינים מספריים, ו-Cosine על embeddings רב-לשוניים בגודל 384. מודל מבוסס למידה חזויה את הסיכוי למועמדות Emmy/Oscar מתוך ציון, פופולריות, עידן וז'אנר. שאילתות חופשיות מפוענחות על ידי מודל שפה והסברים נכתבים בשפת המשתמש.\n\n"
            "**מחברים:** תומר בלונד ועומר ציון — סדנת חדשנות מבוססת AI ו-ML 2025"
        ),
        "source_row": "{name}: {raw:,} שורות גולמיות → {clean:,} נקיות ({pct}% נשמרו)",
        "sidebar_lang":     "🌐 Language / שפה",
        "powered_by":       "מנוע דמיון היברידי + embeddings רב-לשוניים + מודל שפה",
        "llm_status_on":    "✅ מודל השפה מחובר",
        "llm_status_off":   "⚡ מצב אופליין (ללא API key)",
        "hero_line_1":      "האובססיה הבאה שלך",
        "hero_line_2":      "מוצאת אותך.",
        "hero_line_3":      "🎬 CineMatch AI",
        "hero_skip":        "← דלג על האינטרו",
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
