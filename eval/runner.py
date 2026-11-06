"""Router-quality eval runner (design D7).

Reuses the live routing path. The search node is swapped for a stub that
records the decision instead of touching the network, and the brave
DuckDuckGo guard is armed so any attempted real request fails loudly.
The run ends its role before any second LLM pass (no final answers).
"""

from __future__ import annotations

from typing import Any

import graph as agent_graph
from config import settings
from eval.dataset import DatasetError, load_dataset


def _stub_search_node(state: dict) -> dict[str, Any]:
    """Stand-in for the real search node: record the decision, never fetch."""
    return {
        "routed_search": True,
        "search_results": [],
        "messages": [],
    }


def _evaluate_item(item: dict, router_mode: str) -> dict:
    """Route one question through the real path without executing searches."""
    thread: dict[str, Any] = {
        "configurable": {"thread_id": f"eval-{item['id']}"}
    }
    if router_mode == "web_tool":
        graph = agent_graph.build_graph(
            router_mode=router_mode,
            checkpointer=agent_graph.MemorySaver(),
            interrupt_after=["search"],
            node_overrides={"search": _stub_search_node},
        )
    elif router_mode == "json_decision":
        graph = agent_graph.build_graph(
            router_mode=router_mode,
            checkpointer=agent_graph.MemorySaver(),
            interrupt_after=["decide"],
        )
    else:
        raise ValueError(f"Unknown router_mode: {router_mode!r}")

    agent_graph.arm_eval_search_guard()
    try:
        # invoke() finishes at the interrupt when one fires; without one the
        # graph simply reached END (no tool call / no search requested).
        final_state = graph.invoke(
            {"messages": [("user", item["question"])]}, config=thread
        )
    finally:
        agent_graph.release_eval_search_guard()

    actual = "search" if final_state.get("routed_search") else "no_search"
    return {
        "id": item["id"],
        "question": item["question"],
        "expected": item["expected"],
        "actual": actual,
        "hit": actual == item["expected"],
    }


def run_eval(
    dataset_path: str,
    router_mode: str | None = None,
    threshold: float | None = None,
) -> dict:
    """Evaluate every dataset item; returns the report dict."""
    from eval.report import build_report

    items = load_dataset(dataset_path)
    cfg = settings()
    mode = router_mode or cfg["router_mode"]
    minimum = threshold if threshold is not None else cfg["eval_threshold"]

    results = [_evaluate_item(item, mode) for item in items]
    return build_report(results, router_mode=mode, threshold=minimum)


__all__ = ["run_eval", "_evaluate_item", "DatasetError"]
