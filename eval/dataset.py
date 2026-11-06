"""Dataset loader: strict validation, fail-fast on unlabeled/bad items.

specs/router-quality: a dataset item lacking an expected decision or
carrying an unrecognized value must abort the run before any execution.
"""

from __future__ import annotations

import json

VALID_LABELS = ("search", "no_search")


def load_dataset(path: str) -> list[dict]:
    """Return validated dataset items, raising DatasetError on any violation."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict) or not isinstance(
        payload.get("items"), list
    ):
        raise DatasetError(f"Dataset '{path}' must be an object with 'items'.")

    items = []
    for position, raw in enumerate(payload["items"]):
        item_id = raw.get("id", f"#{position + 1}")
        question = raw.get("question")
        expected = raw.get("expected")
        if not isinstance(question, str) or not question.strip():
            raise DatasetError(f"Item {item_id}: missing question text.")
        if expected not in VALID_LABELS:
            raise DatasetError(
                f"Item {item_id}: expected label {expected!r} is missing or "
                f"invalid (must be one of {VALID_LABELS})."
            )
        items.append(
            {
                "id": str(item_id),
                "question": question.strip(),
                "expected": expected,
            }
        )

    if not items:
        raise DatasetError(f"Dataset '{path}' contains no items.")
    if not any(item["expected"] == "search" for item in items):
        raise DatasetError(
            f"Dataset '{path}' must contain at least one 'search' item."
        )
    if not any(item["expected"] == "no_search" for item in items):
        raise DatasetError(
            f"Dataset '{path}' must contain at least one 'no_search' item."
        )
    return items


class DatasetError(ValueError):
    """Raised for structurally invalid or mislabeled datasets."""
