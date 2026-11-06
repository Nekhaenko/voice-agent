"""CLI entry: `python -m eval.run <dataset> [options]` (specs/router-quality).

Exit codes: 0 = pass, 1 = below threshold, 2 = configuration/dataset error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.dataset import DatasetError
from eval.report import format_report
from eval.runner import run_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Router quality evaluation")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(Path(__file__).parent / "dataset.json"),
        help="Path to the labeled dataset JSON.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--router-mode",
        choices=("web_tool", "json_decision"),
        default=None,
    )
    parser.add_argument("--report", default=None, help="Write JSON report here.")
    args = parser.parse_args(argv)

    try:
        report = run_eval(args.dataset, args.router_mode, args.threshold)
    except DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    print(format_report(report))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
