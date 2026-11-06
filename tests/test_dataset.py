"""5.1: dataset loader strictness, incl. the shipped dataset file."""

import pytest
from eval.dataset import DatasetError, load_dataset


def test_shipped_dataset_is_valid_both_labels():
    items = load_dataset("eval/dataset.json")
    assert len(items) >= 10
    labels = {item["expected"] for item in items}
    assert labels == {"search", "no_search"}
    assert all(item["question"].strip() for item in items)


def test_unlabeled_item_fails_fast(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        '{"items": [{"id": "x", "question": "Вопрос?"}]}', encoding="utf-8"
    )
    with pytest.raises(DatasetError) as err:
        load_dataset(str(path))
    assert "x" in str(err.value)


def test_unknown_label_fails_fast(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        '{"items": [{"id": "x", "question": "Вопрос?", "expected": "maybe"}]}',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError):
        load_dataset(str(path))


def test_dataset_missing_question_fails_fast(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"items": [{"id": "x", "expected": "search"}]}', encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(str(path))


def test_dataset_needs_both_labels(tmp_path):
    path = tmp_path / "one-sided.json"
    path.write_text(
        '{"items": [{"id": "x", "question": "Вопрос?", "expected": "search"}]}',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError) as err:
        load_dataset(str(path))
    assert "no_search" in str(err.value)
