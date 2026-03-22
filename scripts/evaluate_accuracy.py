# pyright: reportMissingImports=false

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
import sys

sys.path.insert(0, str(BACKEND_DIR))

from facts_loader import load_facts_records  # noqa: E402
from pipeline import fact_check_pipeline  # noqa: E402


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[float, dict[str, dict[str, float]]]:
    per_label: dict[str, dict[str, float]] = {}
    f1_sum = 0.0

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        f1_sum += f1

    macro_f1 = safe_div(f1_sum, len(labels))
    return macro_f1, per_label


def evaluate(mode: str, claims_path: Path) -> dict:
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    facts = load_facts_records()

    y_true: list[str] = []
    y_pred: list[str] = []
    latency = []
    confidence = []
    by_lang_total = Counter()
    by_lang_correct = Counter()

    results = []
    for row in claims:
        text = row["text"]
        expected = row["expected"].upper()
        lang = row.get("language", "unknown")

        out = fact_check_pipeline(text, facts, mode=mode)
        pred = out.get("verdict", "UNVERIFIABLE").upper()

        y_true.append(expected)
        y_pred.append(pred)

        by_lang_total[lang] += 1
        if pred == expected:
            by_lang_correct[lang] += 1

        latency.append(out.get("latency_ms", 0))
        confidence.append(float(out.get("confidence", 0.0)))

        results.append(
            {
                "text": text,
                "language": lang,
                "expected": expected,
                "predicted": pred,
                "correct": pred == expected,
                "confidence": out.get("confidence", 0.0),
                "latency_ms": out.get("latency_ms", 0),
                "reason": out.get("result", ""),
            }
        )

    labels = ["TRUE", "FALSE", "UNVERIFIABLE"]
    macro_f1, per_label = compute_macro_f1(y_true, y_pred, labels)

    accuracy = safe_div(sum(1 for t, p in zip(y_true, y_pred) if t == p), len(y_true))

    confusion = defaultdict(lambda: Counter())
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    by_lang_accuracy = {
        lang: round(safe_div(by_lang_correct[lang], total), 4)
        for lang, total in by_lang_total.items()
    }

    report = {
        "mode": mode,
        "samples": len(claims),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "avg_latency_ms": round(sum(latency) / max(1, len(latency)), 2),
        "avg_confidence": round(sum(confidence) / max(1, len(confidence)), 3),
        "label_support": dict(Counter(y_true)),
        "prediction_distribution": dict(Counter(y_pred)),
        "by_language_accuracy": by_lang_accuracy,
        "per_label_metrics": per_label,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "results": results,
    }
    return report


if __name__ == "__main__":
    claims_file = ROOT / "scripts" / "eval_claims.json"
    out_dir = ROOT / "scripts"

    smart = evaluate("smart", claims_file)
    naive = evaluate("naive", claims_file)

    (out_dir / "evaluation_report_smart.json").write_text(
        json.dumps(smart, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "evaluation_report_naive.json").write_text(
        json.dumps(naive, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Smart accuracy:", smart["accuracy"], "Macro-F1:", smart["macro_f1"])
    print("Naive accuracy:", naive["accuracy"], "Macro-F1:", naive["macro_f1"])
    print("Reports written to scripts/evaluation_report_smart.json and scripts/evaluation_report_naive.json")
