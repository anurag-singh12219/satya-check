from newsapi import NewsApiClient
from config import NEWS_API_KEY

newsapi = NewsApiClient(api_key=NEWS_API_KEY) if NEWS_API_KEY else None


def fetch_indian_news():
    if not newsapi:
        return [
            "Breaking: New policy claims zero tax up to 12 lakh income.",
            "Viral post says UPI now charges 2 percent fee on all transfers.",
            "Claim: PM-Kisan payout increased to Rs 10000 in 2025.",
        ]

    try:
        articles = newsapi.get_top_headlines(
            country="in",
            language="en",
            page_size=10,
        )
    except Exception:
        return ["Live news feed temporarily unavailable."]

    posts = []

    for a in articles.get("articles", []):
        if a["title"]:
            posts.append(a["title"])

    return posts