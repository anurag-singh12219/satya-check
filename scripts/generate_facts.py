import json
from pathlib import Path

REQUIRED_FIELDS = {
    "id",
    "topic",
    "claim",
    "verdict",
    "source",
    "source_url",
    "date",
    "status",
}


def validate_facts_db(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("facts_db.json must be a list of fact objects")

    missing_summary: list[str] = []
    for idx, fact in enumerate(data):
        if not isinstance(fact, dict):
            raise ValueError(f"Fact index {idx} is not an object")
        missing = REQUIRED_FIELDS - set(fact.keys())
        if missing:
            missing_summary.append(f"{idx}:{sorted(missing)}")

    if missing_summary:
        raise ValueError("Missing required fields in facts: " + "; ".join(missing_summary))

    print(f"Facts DB is valid. Records: {len(data)}")


if __name__ == "__main__":
    facts_path = Path(__file__).resolve().parents[1] / "backend" / "data" / "facts_db.json"
    validate_facts_db(facts_path)
