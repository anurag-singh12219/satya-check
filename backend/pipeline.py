import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

try:
    from langdetect import detect as _langdetect_detect
except Exception:
    _langdetect_detect = None

try:
    from config import (
        ENABLE_SEMANTIC_RAG,
        INPUT_COST_PER_1K,
        RAG_TOP_K_NAIVE,
        RAG_TOP_K_SMART,
        SCALEDOWN_API_KEY,
        SCALEDOWN_URL,
        SEMANTIC_MODEL_NAME,
    )
    from facts_loader import serialize_facts
    from hybrid_retriever import HybridFactRetriever
    from realtime_evidence import fetch_live_evidence, get_domain_weight
except ModuleNotFoundError:
    from .config import (
        ENABLE_SEMANTIC_RAG,
        INPUT_COST_PER_1K,
        RAG_TOP_K_NAIVE,
        RAG_TOP_K_SMART,
        SCALEDOWN_API_KEY,
        SCALEDOWN_URL,
        SEMANTIC_MODEL_NAME,
    )
    from .facts_loader import serialize_facts
    from .hybrid_retriever import HybridFactRetriever
    from .realtime_evidence import fetch_live_evidence, get_domain_weight


SCALEDOWN_CACHE_TTL_SECONDS = 180
SCALEDOWN_CACHE_MAX_ITEMS = 512
_SCALEDOWN_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_RETRIEVER_CACHE: HybridFactRetriever | None = None
_RETRIEVER_CACHE_KEY = ""


DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
TAMIL_DIGITS = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")

# Template replacements canonicalize common viral-claim wording into the same
# domain terms used by verified fact records.
HI_TEMPLATE_REPLACEMENTS = {
    "भारत": "india",
    "विश्व कप": "world cup",
    "जीता": "won",
    "नीति": "policy",
    "नई नीति": "new policy",
    "इनकम टैक्स": "income tax",
    "आयकर": "income tax",
    "कोई टैक्स नहीं": "tax free",
    "टैक्स फ्री": "tax free",
    "पूरी आय": "all income",
    "लाख": "lakh",
    "मुफ्त": "free",
    "सच": "true",
    "झूठ": "false",
    "फर्जी": "fake",
    "यूपीआई": "upi",
    "रिपो रेट": "repo rate",
    "कोविड वैक्सीन": "covid vaccine",
    "प्रधानमंत्री किसान": "pm kisan",
    "आधार": "aadhaar",
    "जीएसटी": "gst",
    "चार्ज": "charge",
    "सभी": "all users",
}

TA_TEMPLATE_REPLACEMENTS = {
    "இந்தியா": "india",
    "உலகக் கோப்பை": "world cup",
    "வென்றது": "won",
    "வருமானவரி": "income tax",
    "வரி இல்லை": "tax free",
    "உண்மை": "true",
    "பொய்": "false",
    "போலி": "fake",
    "யூபிஐ": "upi",
    "ரெப்போ விகிதம்": "repo rate",
    "கோவிட் தடுப்பூசி": "covid vaccine",
    "பிஎம் கிசான்": "pm kisan",
    "ஆதார்": "aadhaar",
    "ஜிஎஸ்டி": "gst",
    "கட்டணம்": "charge",
    "எல்லோருக்கும்": "all users",
    "உயர்ந்தது": "increased",
}

BN_TEMPLATE_REPLACEMENTS = {
    "ভারত": "india",
    "টি২০": "t20",
    "টি20": "t20",
    "বিশ্বকাপ": "world cup",
    "জিতেছে": "won",
    "আয়কর": "income tax",
    "কোন ট্যাক্স নেই": "tax free",
    "সত্য": "true",
    "মিথ্যা": "false",
    "ভুয়া": "fake",
    "ইউপিআই-এ": "upi",
    "ইউপিআই": "upi",
    "রেপো রেট": "repo rate",
    "কোভিড ভ্যাকসিন": "covid vaccine",
    "পিএম কিষান": "pm kisan",
    "আধার": "aadhaar",
    "জিএসটি": "gst",
    "চার্জ": "charge",
    "চার্জ বসানো": "charge imposed",
    "বসানো হয়েছে": "imposed",
    "সবার": "all users",
}

HI_TOKEN_REPLACEMENTS = {
    "भारत": "india",
    "विश्व": "world",
    "कप": "cup",
    "यूपीआई": "upi",
    "रेपो": "repo",
    "रेट": "rate",
    "टैक्स": "tax",
    "फ्री": "free",
}

TA_TOKEN_REPLACEMENTS = {
    "இந்தியா": "india",
    "உலகக்": "world",
    "கோப்பை": "cup",
    "யூபிஐ": "upi",
    "ரெப்போ": "repo",
    "விகிதம்": "rate",
}

BN_TOKEN_REPLACEMENTS = {
    "ভারত": "india",
    "বিশ্বকাপ": "world cup",
    "জিতেছে": "won",
    "টি20": "t20",
    "ইউপিআই": "upi",
    "ইউপিআইএ": "upi",
    "রেপো": "repo",
    "রেট": "rate",
    "চার্জ": "charge",
    "সবার": "all users",
    "বসানো": "imposed",
    "হয়েছে": "imposed",
}

NEGATIVE_CUES = {
    "fake",
    "false",
    "debunk",
    "misleading",
    "not true",
    "incorrect",
    "rumor",
    "hoax",
    "denies",
    "rejects",
}

POSITIVE_CUES = {
    "official",
    "announced",
    "confirmed",
    "approved",
    "issued",
    "declared",
    "won",
    "success",
    "safe",
}

NEGATIVE_STANCE_CUES = {
    "lost",
    "failed",
    "fake",
    "false",
    "not",
    "incorrect",
    "misleading",
    "hoax",
    "no",
    "नहीं",
    "பொய்",
    "மறுப்பு",
    "মিথ্যা",
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "and",
    "or",
    "that",
    "this",
    "it",
    "as",
    "all",
    "new",
    "now",
    "breaking",
    "government",
    "india",
    "bharat",
}

DOMAIN_ANCHORS = {
    "upi",
    "tax",
    "income",
    "repo",
    "rbi",
    "gst",
    "aadhaar",
    "kisan",
    "covid",
    "vaccine",
    "whatsapp",
    "chandrayaan",
    "moon",
    "t20",
    "world",
    "cup",
    "petrol",
    "diesel",
}


def detect_language(text: str) -> str:
    if _langdetect_detect is None:
        guessed = _guess_language_by_script(text)
        return guessed if guessed != "unknown" else "en"
    try:
        detected = _langdetect_detect(text)
        if detected in {"en", "hi", "bn", "ta"}:
            return detected
        guessed = _guess_language_by_script(text)
        return guessed if guessed != "unknown" else "en"
    except Exception:
        guessed = _guess_language_by_script(text)
        return guessed if guessed != "unknown" else "en"


def _guess_language_by_script(text: str) -> str:
    hi_count = len(re.findall(r"[\u0900-\u097F]", text))
    bn_count = len(re.findall(r"[\u0980-\u09FF]", text))
    ta_count = len(re.findall(r"[\u0B80-\u0BFF]", text))

    scores = {"hi": hi_count, "bn": bn_count, "ta": ta_count}
    best_lang = max(scores, key=scores.get)
    if scores[best_lang] > 0:
        return best_lang
    return "unknown"


def _estimate_tokens(text: str) -> int:
    # Lightweight approximation suitable for benchmarking without tokenizer dependency.
    return max(1, int(len(text) / 4))


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s₹.%/-\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _apply_template_replacements(text: str, mapping: dict[str, str]) -> str:
    output = text
    for source, target in mapping.items():
        output = output.replace(source, target)
    return output


def _apply_token_replacements(text: str, mapping: dict[str, str]) -> str:
    tokens = text.split()
    replaced = [mapping.get(token, token) for token in tokens]
    return " ".join(replaced)


def _normalize_multilingual_claim(claim: str, language: str) -> str:
    normalized = claim.translate(DEVANAGARI_DIGITS).translate(BENGALI_DIGITS).translate(TAMIL_DIGITS)
    lang = language if language in {"hi", "bn", "ta"} else _guess_language_by_script(normalized)

    if lang == "hi":
        normalized = _apply_template_replacements(normalized, HI_TEMPLATE_REPLACEMENTS)
    elif lang == "bn":
        normalized = _apply_template_replacements(normalized, BN_TEMPLATE_REPLACEMENTS)
    elif lang == "ta":
        normalized = _apply_template_replacements(normalized, TA_TEMPLATE_REPLACEMENTS)

    normalized = _normalize_text(normalized)

    if lang == "hi":
        normalized = _apply_token_replacements(normalized, HI_TOKEN_REPLACEMENTS)
    elif lang == "bn":
        normalized = _apply_token_replacements(normalized, BN_TOKEN_REPLACEMENTS)
    elif lang == "ta":
        normalized = _apply_token_replacements(normalized, TA_TOKEN_REPLACEMENTS)

    return _normalize_text(normalized)


def _to_terms(text: str) -> set[str]:
    terms = set(_normalize_text(text).split())
    return {t for t in terms if len(t) > 1 and t not in STOPWORDS}


def _fact_similarity(post: str, fact: dict[str, Any]) -> float:
    post_terms = _to_terms(post)
    text = f"{fact.get('topic', '')} {fact.get('claim', '')} {' '.join(fact.get('tags', []))}"
    fact_terms = _to_terms(text)
    if not post_terms or not fact_terms:
        return 0.0
    intersection = post_terms & fact_terms
    if not intersection:
        return 0.0

    anchor_overlap = intersection & DOMAIN_ANCHORS
    if not anchor_overlap and len(intersection) < 2:
        return 0.0

    overlap = len(intersection)
    return overlap / max(1, len(post_terms))


def _stance_polarity(text: str) -> int:
    normalized = _normalize_text(text)
    pos = sum(1 for cue in POSITIVE_CUES if cue in normalized)
    neg = sum(1 for cue in NEGATIVE_STANCE_CUES if cue in normalized)
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def _extract_number_unit_pairs(text: str) -> list[tuple[float, str]]:
    def _canonical_unit(unit: str) -> str:
        u = unit.strip().lower()
        if u in {"%", "percent"}:
            return "percent"
        if u in {"rs", "rupee", "rupees", "inr"}:
            return "inr"
        return u

    normalized = _normalize_text(text)
    pairs: list[tuple[float, str]] = []
    for match in re.finditer(r"(?<!\d)(\d+(?:\.\d+)?)\s*(%|percent|lakh|crore|rs|rupees?|inr)?(?=\D|$)", normalized):
        value = float(match.group(1))
        unit = _canonical_unit(match.group(2) or "")
        pairs.append((value, unit))
    return pairs


def _extract_years(text: str) -> list[int]:
    years = []
    for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", text):
        years.append(int(m.group(1)))
    return years


def _numeric_contradiction(claim_text: str, fact_text: str) -> str:
    claim_pairs = _extract_number_unit_pairs(claim_text)
    fact_pairs = _extract_number_unit_pairs(fact_text)
    if not claim_pairs or not fact_pairs:
        return ""

    claim_units = {u for _, u in claim_pairs if u}
    fact_units = {u for _, u in fact_pairs if u}
    common_units = claim_units & fact_units

    # Conservative check: only trigger when unit alignment is explicit.
    if common_units:
        for unit in common_units:
            claim_vals = [v for v, u in claim_pairs if u == unit]
            fact_vals = [v for v, u in fact_pairs if u == unit]
            for cv in claim_vals:
                nearest = min(fact_vals, key=lambda fv: abs(fv - cv))
                tolerance = 0.05 if unit == "percent" else max(1.0, nearest * 0.03)
                if abs(cv - nearest) > tolerance:
                    return f"Numeric mismatch on unit '{unit}': claim={cv}, fact={nearest}."

    # Fallback for financial anchors with no explicit unit token.
    anchors = {"tax", "repo", "rate", "gst", "upi", "income"}
    claim_terms = _to_terms(claim_text)
    if anchors & claim_terms and len(claim_pairs) == 1 and len(fact_pairs) == 1:
        cv = claim_pairs[0][0]
        fv = fact_pairs[0][0]
        if abs(cv - fv) >= max(1.0, fv * 0.15):
            return f"Numeric mismatch: claim={cv}, fact={fv}."

    return ""


def _temporal_contradiction(claim_text: str, fact_text: str, fact_date: str) -> str:
    claim_years = _extract_years(claim_text)
    fact_years = _extract_years(f"{fact_text} {fact_date}")
    if not claim_years or not fact_years:
        return ""

    if len(claim_years) == 1 and len(fact_years) == 1 and claim_years[0] != fact_years[0]:
        return f"Year mismatch: claim={claim_years[0]}, fact={fact_years[0]}."
    return ""


def _detect_hard_contradiction(claim_text: str, fact: dict[str, Any]) -> str:
    fact_text = str(fact.get("claim", ""))
    fact_date = str(fact.get("date", ""))

    numeric = _numeric_contradiction(claim_text, fact_text)
    if numeric:
        return numeric

    temporal = _temporal_contradiction(claim_text, fact_text, fact_date)
    if temporal:
        return temporal

    return ""


def _similarity_to_confidence(score: float, contradiction: bool) -> float:
    base = min(0.98, max(0.05, score * 1.8))
    if contradiction:
        base = min(base, 0.45)
    return round(base, 3)


def _filter_by_status(facts: list[dict[str, Any]], include_inactive: bool) -> list[dict[str, Any]]:
    if include_inactive:
        return facts
    return [f for f in facts if f.get("status", "active").lower() == "active"]


def _retrieve_candidate_facts(
    facts: list[dict[str, Any]], claim: str, top_k: int, include_inactive: bool
) -> list[dict[str, Any]]:
    retriever = _get_retriever(facts)
    if retriever is not None:
        return retriever.retrieve(claim, top_k=top_k, include_inactive=include_inactive)

    filtered = _filter_by_status(facts, include_inactive=include_inactive)
    scored: list[tuple[float, str, dict[str, Any]]] = []

    for fact in filtered:
        score = _fact_similarity(claim, fact)
        date = str(fact.get("date", ""))
        if score > 0:
            scored.append((score, date, fact))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if scored:
        return [item[2] for item in scored[:top_k]]

    return filtered[:top_k]


def _facts_cache_key(facts: list[dict[str, Any]]) -> str:
    if not facts:
        return "empty"
    first_id = str(facts[0].get("id", ""))
    last_id = str(facts[-1].get("id", ""))
    checksum = sum(len(str(f.get("claim", ""))) for f in facts)
    return f"{len(facts)}:{first_id}:{last_id}:{checksum}:{int(ENABLE_SEMANTIC_RAG)}"


def _get_retriever(facts: list[dict[str, Any]]) -> HybridFactRetriever | None:
    global _RETRIEVER_CACHE, _RETRIEVER_CACHE_KEY
    key = _facts_cache_key(facts)
    if _RETRIEVER_CACHE is not None and _RETRIEVER_CACHE_KEY == key:
        return _RETRIEVER_CACHE

    try:
        _RETRIEVER_CACHE = HybridFactRetriever(
            facts=facts,
            enable_semantic=ENABLE_SEMANTIC_RAG,
            semantic_model_name=SEMANTIC_MODEL_NAME,
        )
        _RETRIEVER_CACHE_KEY = key
        return _RETRIEVER_CACHE
    except Exception:
        _RETRIEVER_CACHE = None
        _RETRIEVER_CACHE_KEY = ""
        return None


def _cost_for_tokens(tokens: int) -> float:
    return (tokens / 1000) * INPUT_COST_PER_1K


def _safe_scaledown(context: str, prompt: str) -> dict[str, Any]:
    context_tokens = _estimate_tokens(context)
    if context_tokens <= 220:
        return {
            "compressed_prompt": context,
            "original_prompt_tokens": context_tokens,
            "compressed_prompt_tokens": context_tokens,
            "successful": True,
            "fallback": "context_already_small",
            "latency_ms": 0,
        }

    cache_key = (context, prompt)
    cached = _SCALEDOWN_CACHE.get(cache_key)
    now = time.time()
    if cached and (now - cached[0]) < SCALEDOWN_CACHE_TTL_SECONDS:
        cached_payload = dict(cached[1])
        cached_payload["cache_hit"] = True
        return cached_payload

    if not SCALEDOWN_API_KEY:
        original_tokens = context_tokens
        return {
            "compressed_prompt": context,
            "original_prompt_tokens": original_tokens,
            "compressed_prompt_tokens": original_tokens,
            "successful": False,
            "fallback": "scaledown_key_missing",
            "latency_ms": 0,
        }

    started = time.perf_counter()
    try:
        response = requests.post(
            SCALEDOWN_URL,
            headers={
                "x-api-key": SCALEDOWN_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "context": context,
                "prompt": prompt,
                "scaledown": {"rate": "auto"},
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        data.setdefault("latency_ms", int((time.perf_counter() - started) * 1000))
        if len(_SCALEDOWN_CACHE) >= SCALEDOWN_CACHE_MAX_ITEMS:
            stale_keys = sorted(_SCALEDOWN_CACHE.items(), key=lambda x: x[1][0])[:64]
            for key, _ in stale_keys:
                _SCALEDOWN_CACHE.pop(key, None)
        _SCALEDOWN_CACHE[cache_key] = (now, data)
        return data
    except Exception as exc:
        original_tokens = context_tokens
        return {
            "compressed_prompt": context,
            "original_prompt_tokens": original_tokens,
            "compressed_prompt_tokens": original_tokens,
            "successful": False,
            "fallback": f"scaledown_error:{type(exc).__name__}",
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }


def _extract_core_claim(post: str) -> tuple[str, dict[str, Any]]:
    # Mandatory optimization: remove conversational fluff before verification.
    instruction = (
        "Extract only factual claims from this social media post. "
        "Remove greetings, emotional language, calls to forward/share, and opinions. "
        "Return compact bullet-style factual claims only."
    )
    sd = _safe_scaledown(post, instruction)
    extracted = sd.get("compressed_prompt") or post
    return extracted.strip(), {
        "post_tokens_before": sd.get("original_prompt_tokens", _estimate_tokens(post)),
        "post_tokens_after": sd.get("compressed_prompt_tokens", _estimate_tokens(extracted)),
        "post_compression_latency_ms": sd.get("latency_ms", 0),
        "post_compression_success": bool(sd.get("successful", False)),
        "post_compression_fallback": sd.get("fallback", ""),
    }


def _extract_fact_ids(text: str) -> list[str]:
    return re.findall(r"\[FACT_ID:([A-Za-z0-9_-]+)\]", text)


def _verdict_from_facts(post: str, selected_facts: list[dict[str, Any]], language: str) -> dict[str, Any]:
    started = time.perf_counter()
    if not selected_facts:
        return {
            "result": "VERDICT: UNVERIFIABLE\nREASON: No relevant verified facts were found for this claim.",
            "verdict": "UNVERIFIABLE",
            "verifier_latency_ms": int((time.perf_counter() - started) * 1000),
            "matched_fact_id": None,
            "matched_fact": None,
            "citations": [],
            "confidence": 0.1,
        }

    claim_polarity = _stance_polarity(post)
    ranked = []
    for fact in selected_facts:
        raw_score = _fact_similarity(post, fact)
        fact_polarity = _stance_polarity(str(fact.get("claim", "")))
        adjusted = raw_score
        if claim_polarity != 0 and fact_polarity != 0 and claim_polarity != fact_polarity:
            adjusted = max(0.0, raw_score * 0.55)
        ranked.append(
            {
                "score": raw_score,
                "adjusted_score": adjusted,
                "fact": fact,
            }
        )

    ranked.sort(key=lambda x: (x["adjusted_score"], str(x["fact"].get("date", ""))), reverse=True)

    best = ranked[0]
    anchor_terms = {
        "upi",
        "income",
        "tax",
        "repo",
        "gst",
        "aadhaar",
        "kisan",
        "covid",
        "vaccine",
        "whatsapp",
        "chandrayaan",
    }
    post_terms = _to_terms(post)
    has_anchor = bool(post_terms & anchor_terms)

    threshold = 0.20
    if language in {"hi", "ta", "bn"} and has_anchor:
        threshold = 0.10

    if best["adjusted_score"] < threshold:
        return {
            "result": "VERDICT: UNVERIFIABLE\nREASON: The post does not align strongly enough with a verified fact entry.",
            "verdict": "UNVERIFIABLE",
            "verifier_latency_ms": int((time.perf_counter() - started) * 1000),
            "matched_fact_id": None,
            "matched_fact": None,
            "citations": [],
            "confidence": round(max(0.1, best["adjusted_score"]), 3),
        }

    verdict = str(best["fact"].get("verdict", "UNVERIFIABLE")).upper()
    claim = best["fact"].get("claim", "")
    source = best["fact"].get("source", "Unknown source")
    date = best["fact"].get("date", "")
    contradiction = _detect_hard_contradiction(post, best["fact"])

    if contradiction and verdict in {"TRUE", "UNVERIFIABLE"}:
        verdict = "FALSE"
        reason = (
            f"Claim contradicts verified fact '{claim}' from {source} ({date}). "
            f"{contradiction}"
        )
    elif contradiction and verdict == "FALSE":
        reason = (
            f"Matched debunked claim '{claim}' from {source} ({date}); "
            f"conflict signal: {contradiction}"
        )
    else:
        reason = f"Matched verified fact '{claim}' from {source} ({date})."

    confidence = _similarity_to_confidence(best["adjusted_score"], bool(contradiction))

    return {
        "result": f"VERDICT: {verdict}\nREASON: {reason}",
        "verdict": verdict,
        "verifier_latency_ms": int((time.perf_counter() - started) * 1000),
        "matched_fact_id": best["fact"].get("id"),
        "matched_fact": best["fact"],
        "citations": [
            {
                "source": source,
                "url": best["fact"].get("source_url", ""),
                "date": date,
            }
        ],
        "confidence": confidence,
    }


def _score_live_evidence_match(claim: str, evidence_text: str) -> float:
    claim_terms = _to_terms(claim)
    evidence_terms = _to_terms(evidence_text)
    if not claim_terms or not evidence_terms:
        return 0.0
    overlap = len(claim_terms & evidence_terms)
    return overlap / max(1, len(claim_terms))


def _verdict_from_live_evidence(claim: str, evidence_rows: list[dict[str, str]]) -> dict[str, Any]:
    started = time.perf_counter()
    if not evidence_rows:
        return {
            "result": "VERDICT: UNVERIFIABLE\nREASON: No recent trusted-source evidence was found for this claim.",
            "verdict": "UNVERIFIABLE",
            "verifier_latency_ms": int((time.perf_counter() - started) * 1000),
            "matched_fact_id": None,
            "matched_fact": None,
            "citations": [],
            "confidence": 0.1,
        }

    scored_rows: list[tuple[float, dict[str, str]]] = []
    for row in evidence_rows:
        text = f"{row.get('title', '')} {row.get('snippet', '')}"
        score = _score_live_evidence_match(claim, text)
        if score > 0:
            scored_rows.append((score, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    if not scored_rows:
        return {
            "result": "VERDICT: UNVERIFIABLE\nREASON: Recent sources were found but none were sufficiently aligned with the claim.",
            "verdict": "UNVERIFIABLE",
            "verifier_latency_ms": int((time.perf_counter() - started) * 1000),
            "matched_fact_id": None,
            "matched_fact": None,
            "citations": [],
            "confidence": 0.15,
        }

    top_rows = scored_rows[:5]
    best_score = top_rows[0][0]

    positive_score = 0.0
    negative_score = 0.0
    citations: list[dict[str, str]] = []

    for similarity, row in top_rows:
        evidence_text = _normalize_text(f"{row.get('title', '')} {row.get('snippet', '')}")
        trust = get_domain_weight(row.get("domain", ""))
        weight = similarity * trust

        pos_hits = sum(1 for cue in POSITIVE_CUES if cue in evidence_text)
        neg_hits = sum(1 for cue in NEGATIVE_CUES if cue in evidence_text)

        if pos_hits > 0:
            positive_score += weight * (1 + 0.2 * pos_hits)
        if neg_hits > 0:
            negative_score += weight * (1 + 0.2 * neg_hits)

        citations.append(
            {
                "source": row.get("source", "unknown"),
                "url": row.get("url", ""),
                "date": row.get("published_at", ""),
            }
        )

    verdict = "UNVERIFIABLE"
    reason = (
        "Recent trusted-source evidence was found, but stance is not explicit enough "
        "to mark TRUE or FALSE confidently."
    )

    if best_score >= 0.20 and negative_score >= max(0.35, positive_score * 1.15):
        verdict = "FALSE"
        reason = "Trusted-source coverage appears to debunk or reject this claim."
    elif best_score >= 0.22 and positive_score >= max(0.35, negative_score * 1.15):
        verdict = "TRUE"
        reason = "Trusted-source coverage appears to confirm this claim."

    stance_strength = max(positive_score, negative_score)
    confidence = round(min(0.92, max(0.2, best_score + (stance_strength * 0.5))), 3)
    if verdict == "UNVERIFIABLE":
        confidence = min(confidence, 0.45)

    return {
        "result": f"VERDICT: {verdict}\nREASON: {reason}",
        "verdict": verdict,
        "verifier_latency_ms": int((time.perf_counter() - started) * 1000),
        "matched_fact_id": None,
        "matched_fact": None,
        "citations": citations[:3],
        "confidence": confidence,
    }


def _extract_verdict_label(result_text: str) -> str:
    first = (result_text.splitlines()[:1] or [""])[0].strip().upper()
    if "VERDICT:" in first:
        return first.replace("VERDICT:", "").strip()
    if "TRUE" in first:
        return "TRUE"
    if "FALSE" in first:
        return "FALSE"
    return "UNVERIFIABLE"


def fact_check_pipeline(post: str, facts_records: list[dict[str, Any]], mode: str = "smart") -> dict[str, Any]:
    started = time.perf_counter()
    detected = detect_language(post)
    script_detected = _guess_language_by_script(post)
    language = script_detected if script_detected != "unknown" else detected

    if mode not in {"smart", "naive"}:
        raise ValueError("mode must be 'smart' or 'naive'")

    all_context = serialize_facts(facts_records)
    naive_context_tokens = _estimate_tokens(all_context)

    if mode == "smart":
        compact_post, post_metrics = _extract_core_claim(post)
        normalized_claim = _normalize_multilingual_claim(compact_post, language)
        candidate_facts = _retrieve_candidate_facts(
            facts_records,
            normalized_claim,
            top_k=RAG_TOP_K_SMART,
            include_inactive=False,
        )
        candidate_context = serialize_facts(candidate_facts)
        sd = _safe_scaledown(candidate_context, normalized_claim)
        relevant_facts_text = sd.get("compressed_prompt", candidate_context)
        ids = set(_extract_fact_ids(relevant_facts_text))

        if ids:
            selected_facts = [fact for fact in candidate_facts if fact.get("id") in ids]
        else:
            selected_facts = candidate_facts[:10]

        tokens_before = naive_context_tokens
        tokens_after = int(sd.get("compressed_prompt_tokens", _estimate_tokens(relevant_facts_text)))
        compression_latency_ms = int(sd.get("latency_ms", 0)) + int(post_metrics["post_compression_latency_ms"])
        compression_fallback = sd.get("fallback", "") or post_metrics.get("post_compression_fallback", "")
    else:
        compact_post = post
        normalized_claim = _normalize_multilingual_claim(compact_post, language)
        post_metrics = {
            "post_tokens_before": _estimate_tokens(post),
            "post_tokens_after": _estimate_tokens(post),
            "post_compression_latency_ms": 0,
            "post_compression_success": False,
            "post_compression_fallback": "naive_mode",
        }
        selected_facts = _retrieve_candidate_facts(
            facts_records,
            normalized_claim,
            top_k=RAG_TOP_K_NAIVE,
            include_inactive=True,
        )
        tokens_before = naive_context_tokens
        tokens_after = tokens_before
        compression_latency_ms = 0
        compression_fallback = "naive_mode"

    verifier_output = _verdict_from_facts(normalized_claim, selected_facts, language)

    live_evidence_used = False
    live_evidence_count = 0
    if mode == "smart" and verifier_output["verdict"] == "UNVERIFIABLE":
        live_rows = fetch_live_evidence(query=normalized_claim or compact_post, language=language, limit=8)
        live_evidence_count = len(live_rows)
        if live_rows:
            live_evidence_used = True
            live_verdict = _verdict_from_live_evidence(normalized_claim or compact_post, live_rows)
            verifier_output = live_verdict
    result = verifier_output["result"]
    verdict = _extract_verdict_label(result)
    total_latency_ms = int((time.perf_counter() - started) * 1000)

    naive_cost = _cost_for_tokens(tokens_before)
    actual_cost = _cost_for_tokens(tokens_after)

    return {
        "result": result,
        "verdict": verdict,
        "language": language,
        "mode": mode,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "token_savings_pct": round((1 - (tokens_after / max(tokens_before, 1))) * 100, 2),
        "cost_estimate_usd": round(actual_cost, 6),
        "cost_if_naive_usd": round(naive_cost, 6),
        "cost_saved_usd": round(max(0.0, naive_cost - actual_cost), 6),
        "latency_ms": total_latency_ms,
        "compression_latency_ms": compression_latency_ms,
        "verifier_latency_ms": verifier_output["verifier_latency_ms"],
        "verifier": "deterministic-evidence-engine",
        "compression_fallback": compression_fallback,
        "llm_fallback": "not_used",
        "matched_fact_id": verifier_output["matched_fact_id"],
        "matched_fact": verifier_output["matched_fact"],
        "confidence": verifier_output.get("confidence", 0.0),
        "citations": verifier_output.get("citations", []),
        "claim_compaction": {
            "original": post,
            "compact": compact_post,
            "normalized": normalized_claim,
            **post_metrics,
        },
        "selected_facts_count": len(selected_facts),
        "live_evidence_used": live_evidence_used,
        "live_evidence_count": live_evidence_count,
    }


def run_batch(
    posts: list[str],
    facts_records: list[dict[str, Any]],
    mode: str = "smart",
    max_workers: int = 20,
) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in executor.map(lambda p: fact_check_pipeline(p, facts_records, mode=mode), posts):
            results.append(item)

    elapsed = max(0.001, time.perf_counter() - started)
    throughput_per_min = int((len(posts) / elapsed) * 60)

    verdict_counts = {"TRUE": 0, "FALSE": 0, "UNVERIFIABLE": 0}
    tokens_before = 0
    tokens_after = 0
    total_cost = 0.0
    total_naive_cost = 0.0

    for item in results:
        verdict_counts[item["verdict"]] = verdict_counts.get(item["verdict"], 0) + 1
        tokens_before += item["tokens_before"]
        tokens_after += item["tokens_after"]
        total_cost += item["cost_estimate_usd"]
        total_naive_cost += item["cost_if_naive_usd"]

    return {
        "mode": mode,
        "count": len(posts),
        "workers": max_workers,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_per_min": throughput_per_min,
        "verdict_counts": verdict_counts,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "token_savings_pct": round((1 - (tokens_after / max(tokens_before, 1))) * 100, 2),
        "cost_estimate_usd": round(total_cost, 6),
        "cost_if_naive_usd": round(total_naive_cost, 6),
        "cost_saved_usd": round(max(0.0, total_naive_cost - total_cost), 6),
        "results": results,
    }