import time
from urllib.parse import quote_plus, urlparse
from xml.etree import ElementTree

import requests

CACHE_TTL_SECONDS = 300
_CACHE: dict[str, tuple[float, list[dict[str, str]]]] = {}

TRUSTED_SITE_FILTER = (
    "site:pib.gov.in OR site:rbi.org.in OR site:mohfw.gov.in OR "
    "site:npci.org.in OR site:uidai.gov.in OR site:eci.gov.in OR "
    "site:icmr.gov.in OR site:isro.gov.in OR site:indiabudget.gov.in OR "
    "site:factcheck.pib.gov.in"
)

DOMAIN_TRUST_WEIGHTS = {
    "pib.gov.in": 1.0,
    "factcheck.pib.gov.in": 1.0,
    "rbi.org.in": 1.0,
    "mohfw.gov.in": 0.95,
    "npci.org.in": 0.95,
    "uidai.gov.in": 0.95,
    "eci.gov.in": 0.95,
    "isro.gov.in": 0.95,
    "icmr.gov.in": 0.95,
    "indiabudget.gov.in": 0.9,
}


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return "unknown"


def get_domain_weight(domain: str) -> float:
    if not domain:
        return 0.4
    d = domain.lower()
    if d in DOMAIN_TRUST_WEIGHTS:
        return DOMAIN_TRUST_WEIGHTS[d]
    for suffix, weight in DOMAIN_TRUST_WEIGHTS.items():
        if d.endswith(suffix):
            return weight
    return 0.45


def _build_google_news_rss_url(query: str, language: str) -> str:
    hl = "en-IN"
    ceid = "IN:en"
    if language == "hi":
        hl = "hi-IN"
        ceid = "IN:hi"
    search = quote_plus(f"{query} {TRUSTED_SITE_FILTER}")
    return f"https://news.google.com/rss/search?q={search}&hl={hl}&gl=IN&ceid={ceid}"


def _fetch_google_news_rss(query: str, language: str, limit: int) -> list[dict[str, str]]:
    url = _build_google_news_rss_url(query, language)
    response = requests.get(url, timeout=12)
    response.raise_for_status()

    root = ElementTree.fromstring(response.text)
    items = root.findall("./channel/item")

    evidence: list[dict[str, str]] = []
    for item in items[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source_name = (item.findtext("source") or "").strip() if item.find("source") is not None else ""
        domain = _domain_from_url(link)

        if not title or not link:
            continue

        evidence.append(
            {
                "title": title,
                "snippet": title,
                "url": link,
                "source": source_name or domain,
                "domain": domain,
                "published_at": pub_date,
            }
        )

    return evidence


def _dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        key = row.get("url") or row.get("title", "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def fetch_live_evidence(query: str, language: str = "en", limit: int = 10) -> list[dict[str, str]]:
    cache_key = f"{language}:{query.strip().lower()}:{limit}"
    now = time.time()

    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if now - ts < CACHE_TTL_SECONDS:
            return data

    evidence: list[dict[str, str]] = []
    query_variants = [
        query,
        f"{query} fact check",
        f"{query} official statement",
    ]

    for q in query_variants:
        try:
            evidence.extend(_fetch_google_news_rss(query=q, language=language, limit=limit))
        except Exception:
            continue

    evidence = _dedupe_rows(evidence)[:limit]

    _CACHE[cache_key] = (now, evidence)
    return evidence


def _fetch_rss_generic(url: str, limit: int) -> list[dict[str, str]]:
    response = requests.get(url, timeout=12)
    response.raise_for_status()
    root = ElementTree.fromstring(response.text)

    rows: list[dict[str, str]] = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        if not title or not link:
            continue
        rows.append(
            {
                "title": title,
                "snippet": title,
                "url": link,
                "source": _domain_from_url(link),
                "domain": _domain_from_url(link),
                "published_at": pub_date,
            }
        )
    return rows


def fetch_social_feed(
    x_handles: list[str] | None = None,
    instagram_handles: list[str] | None = None,
    limit_per_handle: int = 5,
) -> list[dict[str, str]]:
    x_handles = x_handles or []
    instagram_handles = instagram_handles or []
    rows: list[dict[str, str]] = []

    for handle in x_handles:
        handle = handle.strip().lstrip("@").lower()
        if not handle:
            continue
        rss_url = f"https://nitter.net/{handle}/rss"
        try:
            feed = _fetch_rss_generic(rss_url, limit=limit_per_handle)
            for item in feed:
                item["platform"] = "x"
                item["handle"] = handle
            rows.extend(feed)
        except Exception:
            continue

    for handle in instagram_handles:
        handle = handle.strip().lstrip("@").lower()
        if not handle:
            continue
        rss_url = f"https://rsshub.app/instagram/user/{handle}"
        try:
            feed = _fetch_rss_generic(rss_url, limit=limit_per_handle)
            for item in feed:
                item["platform"] = "instagram"
                item["handle"] = handle
            rows.extend(feed)
        except Exception:
            continue

    return _dedupe_rows(rows)
