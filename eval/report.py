"""Router-quality report: per-item verdicts plus aggregate accuracy (specs/router-quality)."""

from __future__ import annotations

from typing import Any


def build_report(results: list[dict], router_mode: str, threshold: float) -> dict[str, Any]:
    """Build the report structure; callers serialize it to file/stdout."""
    hits = sum(1 for item in results if item["hit"])
    accuracy = hits / len(results) if results else 0.0
    return {
        "router_mode": router_mode,
        "threshold": threshold,
        "total": len(results),
        "hits": hits,
        "accuracy": round(accuracy, 4),
        "passed": accuracy >= threshold,
        "items": results,
    }


def format_report(report: dict) -> str:
    """Human-readable rendering of the report."""
    lines = [
        f"Router quality evaluation ({report['router_mode']})",
        f"Threshold: {report['threshold']:.3f} | "
        f"Accuracy: {report['accuracy']:.4f} ({report['hits']}/{report['total']}) | "
        f"Verdict: {'PASS' if report['passed'] else 'FAIL'}",
    ]
    for item in report["items"]:
        verdict = "hit" if item["hit"] else "MISS"
        lines.append(
            f"  [{verdict}] {item['id']}: expected={item['expected']} "
            f"actual={item['actual']} :: {item['question']}"
        )
    return "\n".join(lines)
