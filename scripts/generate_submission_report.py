# pyright: reportMissingImports=false

import json
from datetime import UTC, datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from facts_loader import load_facts_records  # noqa: E402
from pipeline import fact_check_pipeline, run_batch  # noqa: E402


def _accuracy(posts_with_labels: list[dict], facts: list[dict], mode: str) -> float:
    correct = 0
    for row in posts_with_labels:
        expected = str(row.get("expected", "UNVERIFIABLE")).upper()
        predicted = (
            fact_check_pipeline(str(row.get("text", "")), facts, mode=mode)
            .get("verdict", "UNVERIFIABLE")
            .upper()
        )
        if expected == predicted:
            correct += 1
    return round(correct / max(1, len(posts_with_labels)), 4)


def _currency(amount: float) -> str:
    return f"${amount:,.6f}"


def main() -> None:
    claims_file = ROOT / "scripts" / "eval_claims.json"
    claims = json.loads(claims_file.read_text(encoding="utf-8"))
    facts = load_facts_records()

    posts = [str(row.get("text", "")) for row in claims if row.get("text")]
    bench_posts = (posts * 40)[:400] if posts else []

    smart_bench = run_batch(bench_posts, facts, mode="smart", max_workers=50)
    naive_bench = run_batch(bench_posts, facts, mode="naive", max_workers=50)

    smart_acc = _accuracy(claims, facts, mode="smart")
    naive_acc = _accuracy(claims, facts, mode="naive")

    sample_output = fact_check_pipeline(posts[0], facts, mode="smart") if posts else {}
    sample_result = str(sample_output.get("result", ""))
    format_ok = "VERDICT:" in sample_result and "REASON:" in sample_result
    citation_ok = len(sample_output.get("citations", [])) > 0

    checks = {
        "problem_shown_naive": naive_bench["tokens_before"] > smart_bench["tokens_after"],
        "fix_shown_scaledown": smart_bench["tokens_after"] < smart_bench["tokens_before"],
        "numbers_shown": True,
        "correctness_shown": smart_acc >= 0.8,
        "throughput_thousands_per_min": smart_bench["throughput_per_min"] >= 1000,
        "low_cost_per_item": (smart_bench["cost_estimate_usd"] / max(1, smart_bench["count"])) <= 0.01,
        "output_format_ok": format_ok,
        "citation_present": citation_ok,
    }

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_samples": len(claims),
        "benchmark_posts": len(bench_posts),
        "smart": {
            "throughput_per_min": smart_bench["throughput_per_min"],
            "token_savings_pct": smart_bench["token_savings_pct"],
            "cost_estimate_usd": smart_bench["cost_estimate_usd"],
            "cost_if_naive_usd": smart_bench["cost_if_naive_usd"],
            "cost_saved_usd": smart_bench["cost_saved_usd"],
            "accuracy": smart_acc,
        },
        "naive": {
            "throughput_per_min": naive_bench["throughput_per_min"],
            "cost_estimate_usd": naive_bench["cost_estimate_usd"],
            "accuracy": naive_acc,
        },
        "submission_checks": checks,
    }

    json_path = ROOT / "scripts" / "submission_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    all_pass = all(checks.values())
    lines = [
        "# Submission Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Rubric Summary",
        f"- Overall status: {'PASS' if all_pass else 'REVIEW NEEDED'}",
        f"- Problem shown (naive baseline): {'PASS' if checks['problem_shown_naive'] else 'FAIL'}",
        f"- Fix shown (ScaleDown/smart optimization): {'PASS' if checks['fix_shown_scaledown'] else 'FAIL'}",
        f"- Numbers shown (cost/latency/tokens/throughput): {'PASS' if checks['numbers_shown'] else 'FAIL'}",
        f"- Correctness shown: {'PASS' if checks['correctness_shown'] else 'FAIL'}",
        f"- Throughput in thousands/minute: {'PASS' if checks['throughput_thousands_per_min'] else 'FAIL'}",
        f"- Low cost per check: {'PASS' if checks['low_cost_per_item'] else 'FAIL'}",
        f"- Output format VERDICT+REASON: {'PASS' if checks['output_format_ok'] else 'FAIL'}",
        f"- Citation present: {'PASS' if checks['citation_present'] else 'FAIL'}",
        "",
        "## Metrics",
        f"- Eval samples: {len(claims)}",
        f"- Benchmark posts: {len(bench_posts)}",
        f"- Smart throughput: {smart_bench['throughput_per_min']} posts/min",
        f"- Naive throughput: {naive_bench['throughput_per_min']} posts/min",
        f"- Smart token savings: {smart_bench['token_savings_pct']}%",
        f"- Smart total cost (benchmark batch): {_currency(smart_bench['cost_estimate_usd'])}",
        f"- Naive baseline cost for same batch: {_currency(smart_bench['cost_if_naive_usd'])}",
        f"- Smart cost saved: {_currency(smart_bench['cost_saved_usd'])}",
        f"- Smart accuracy: {smart_acc}",
        f"- Naive accuracy: {naive_acc}",
    ]

    md_path = ROOT / "scripts" / "submission_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Overall status: {'PASS' if all_pass else 'REVIEW NEEDED'}")


if __name__ == "__main__":
    main()
