"""5.2: eval reuses the real routing path, never the network, no answers."""

import pytest
from langchain_core.messages import AIMessage

import graph
from eval.dataset import load_dataset
from eval.runner import run_eval
from fakes import FakeLLM, make_tool_call_message, patch_llm, patch_search

SEARCH_WORDS = ("курс", "погода", "новости", "чемпионата", "билет", "налоговый")


def joined_text(messages) -> str:
    parts = []
    for message in messages:
        content = getattr(message, "content", message)
        if isinstance(content, (list, tuple)):
            content = " ".join(
                part[-1] if isinstance(part, tuple) else str(part)
                for part in content
            )
        parts.append(str(content))
    return " ".join(parts)


def decider(web_tool: bool, search_bias=True):
    """Fake router: decide with keywords, as the local tests require."""

    def respond(messages):
        wants_search = search_bias and any(
            w in joined_text(messages).lower() for w in SEARCH_WORDS
        )
        if web_tool and wants_search:
            return make_tool_call_message("web_search", "ключевые слова")
        if web_tool:
            return AIMessage(content="Отвечаю напрямую.")
        return AIMessage(
            content=f'{{"search": {"true" if wants_search else "false"}}}'
        )

    return respond


def keyword_router_llm(web_tool: bool, search_bias=True) -> FakeLLM:
    fake = FakeLLM()

    def invoke(messages, *args, **kwargs):
        fake.invocations.append(list(messages))
        return decider(web_tool, search_bias)(messages)

    fake.invoke = invoke
    return fake


@pytest.fixture()
def search_tracker(monkeypatch):
    """Record real-search attempts through the live function."""
    real_search = graph._duckduckgo_search
    attempts = []

    def guarded(query, max_results):
        attempts.append(query)
        return real_search(query, max_results)

    monkeypatch.setattr(graph, "_duckduckgo_search", guarded)
    return attempts


def test_guard_blocks_real_search_when_armed():
    graph.arm_eval_search_guard()
    try:
        with pytest.raises(RuntimeError):
            graph._duckduckgo_search("anything", 3)
    finally:
        graph.release_eval_search_guard()


@pytest.mark.parametrize("router_mode", ["web_tool", "json_decision"])
def test_full_dataset_routes_correctly(router_mode, monkeypatch, search_tracker):
    """A correct keyword decider yields 100% hits without any real search."""
    patch_llm(monkeypatch, keyword_router_llm(web_tool=router_mode == "web_tool"))
    patch_search(monkeypatch, results=[{"title": "t", "url": "", "snippet": "s"}])

    graph.arm_eval_search_guard()
    try:
        report = run_eval("eval/dataset.json", router_mode=router_mode)
    finally:
        graph.release_eval_search_guard()

    assert search_tracker == [], "eval must not perform real searches"
    assert graph._eval_search_guard is False, "guard must be released after run"
    assert report["router_mode"] == router_mode
    assert report["accuracy"] == 1.0
    assert report["passed"] is True
    assert len(report["items"]) >= 10
    assert all(item["hit"] for item in report["items"])


@pytest.mark.parametrize("router_mode", ["web_tool", "json_decision"])
def test_eval_one_decision_pass_only(router_mode, monkeypatch, search_tracker):
    """Exactly one LLM invocation per item: the decision pass; no answers."""
    fake = patch_llm(
        monkeypatch, keyword_router_llm(web_tool=router_mode == "web_tool")
    )
    patch_search(monkeypatch, results=[])
    run_eval("eval/dataset.json", router_mode=router_mode)
    total_items = len(load_dataset("eval/dataset.json"))
    assert len(fake.invocations) == total_items


def test_never_router_yields_misses_and_failed_report(
    monkeypatch, search_tracker
):
    """A decider that always says no_search: search items miss, report FAILs."""
    patch_llm(monkeypatch, keyword_router_llm(web_tool=False, search_bias=False))
    patch_search(monkeypatch, results=[])
    report = run_eval("eval/dataset.json", router_mode="json_decision")

    misses = [item for item in report["items"] if not item["hit"]]
    assert misses, "search-label items must miss under a never-search router"
    assert report["accuracy"] < 1.0
    assert report["passed"] is False
