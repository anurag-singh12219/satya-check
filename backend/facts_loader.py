import json
from typing import Any

from config import DATA_DIR


def _normalize_fact(item: Any) -> str:
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        claim = item.get("claim", "")
        verdict = item.get("verdict", "")
        source = item.get("source", "")
        updated_at = item.get("updated_at", "")
        topic = item.get("topic", "")
        return (
            f"[TOPIC: {topic}] [UPDATED: {updated_at}] "
            f"CLAIM: {claim} | VERDICT: {verdict} | SOURCE: {source}"
        ).strip()

    return str(item)


def load_facts_records() -> list[dict[str, Any]]:
    facts_file = DATA_DIR / "facts_db.json"
    if not facts_file.exists():
        raise FileNotFoundError(f"Facts database not found at {facts_file}")

    with facts_file.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("facts_db.json must contain a JSON list of facts")

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if isinstance(item, dict):
            fact = dict(item)
            fact.setdefault("id", f"FACT_{idx:04d}")
            fact.setdefault("topic", "general")
            fact.setdefault("claim", "")
            fact.setdefault("verdict", "UNVERIFIABLE")
            fact.setdefault("source", "Unknown")
            fact.setdefault("date", "")
            fact.setdefault("status", "active")
            fact.setdefault("language", "en")
            fact.setdefault("tags", [])
            normalized.append(fact)
        else:
            normalized.append(
                {
                    "id": f"FACT_{idx:04d}",
                    "topic": "general",
                    "claim": str(item),
                    "verdict": "UNVERIFIABLE",
                    "source": "Unknown",
                    "date": "",
                    "status": "active",
                    "language": "en",
                    "tags": [],
                }
            )

    return normalized


def serialize_facts(facts: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in facts:
        tags = ", ".join(item.get("tags", []))
        lines.append(
            (
                f"[FACT_ID:{item.get('id')}] [TOPIC:{item.get('topic')}] "
                f"[DATE:{item.get('date')}] [STATUS:{item.get('status')}] "
                f"CLAIM: {item.get('claim')} | VERDICT: {item.get('verdict')} | "
                f"SOURCE: {item.get('source')} | TAGS: {tags}"
            )
        )

    return "\n".join(lines)


def load_facts() -> str:
    # Backward-compatible string view of facts database.
    return serialize_facts(load_facts_records())
