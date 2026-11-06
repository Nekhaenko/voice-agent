"""5.3: report content and the CLI exit/report contract."""

import json

import pytest
from eval.report import build_report, format_report
from eval.runner import run_eval
from eval.run import main as eval_main
from eval.dataset import DatasetError


def sample_items():
    return [
        {
            "id": "a",
            "question": "Курс доллара?",
            "expected": "search",
            "actual": "search",
            "hit": True,
        },
        {
            "id": "b",
            "question": "Привет",
            "expected": "no_search",
            "actual": "search",
            "hit": False,
        },
        {
            "id": "c",
            "question": "Кто написал «Война и мир»?",
            "expected": "no_search",
            "actual": "no_search",
            "hit": True,
        },
    ]


def test_report_aggregates_accuracy_and_threshold():
    report = build_report(sample_items(), router_mode="json_decision", threshold=0.8)
    assert report["total"] == 3
    assert report["hits"] == 2
    assert abs(report["accuracy"] - round(2 / 3, 4)) < 1e-9
    assert report["passed"] is False


def test_format_lists_misses():
    rendered = format_report(build_report(sample_items(), "web_tool", 0.5))
    assert "[MISS]" in rendered
    assert "[hit]" in rendered
    assert "0.5000" in rendered or "0.5" in rendered


def test_cli_exit_codes_and_report_file(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"

    def failing(*args, **kwargs):
        return build_report(sample_items(), "web_tool", 0.8)

    monkeypatch.setattr("eval.run.run_eval", failing)
    assert eval_main(["--report", str(report_path)]) == 1
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["passed"] is False and written["total"] == 3

    def passing(*args, **kwargs):
        good = build_report(sample_items(), "web_tool", 0.5)
        return good

    monkeypatch.setattr("eval.run.run_eval", passing)
    assert eval_main(["--report", str(report_path)]) == 0

    def bad_dataset(*args, **kwargs):
        raise DatasetError("unknown label")

    monkeypatch.setattr("eval.run.run_eval", bad_dataset)
    assert eval_main([]) == 2
