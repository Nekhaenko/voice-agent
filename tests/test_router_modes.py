"""3.3: router_mode switch produces decisions in both modes (dataset sample)."""

from langchain_core.messages import AIMessage

from fakes import FakeLLM, make_tool_call_message, patch_llm, patch_search
from graph import build_graph

SEARCH_QUESTION = "Какой сейчас курс доллара к рублю?"  # eval q01
PLAIN_QUESTION = "Привет, как дела?"  # eval q07


def test_web_tool_searches_when_tool_call(monkeypatch):
    patch_llm(
        monkeypatch,
        FakeLLM(
            responses=[
                make_tool_call_message("web_search", "курс"),
                AIMessage(content="Курс найден"),
            ]
        ),
    )
    patch_search(monkeypatch, results=[{"title": "x", "url": "", "snippet": "y"}])
    state = build_graph(router_mode="web_tool").invoke(
        {"messages": [("user", SEARCH_QUESTION)]},
        config={"configurable": {"thread_id": "m1"}},
    )
    assert state["routed_search"] is True


def test_json_decision_searches_on_true(monkeypatch):
    patch_llm(
        monkeypatch,
        FakeLLM(
            responses=[
                AIMessage(content='{"search": true}'),
                AIMessage(content="Курс найден"),
            ]
        ),
    )
    calls = patch_search(monkeypatch, results=[{"title": "x", "url": "", "snippet": "y"}])
    state = build_graph(router_mode="json_decision").invoke(
        {"messages": [("user", SEARCH_QUESTION)]},
        config={"configurable": {"thread_id": "m2"}},
    )
    assert calls, "json_decision 'search' must reach the search node"
    assert state["routed_search"] is True
    assert state["messages"][-1].content == "Курс найден"


def test_json_decision_skips_on_false(monkeypatch):
    patch_llm(
        monkeypatch,
        FakeLLM(responses=[AIMessage(content='{"search": false}')]),
    )
    calls = patch_search(monkeypatch, results=[])
    state = build_graph(router_mode="json_decision").invoke(
        {"messages": [("user", PLAIN_QUESTION)]},
        config={"configurable": {"thread_id": "m3"}},
    )
    assert not calls, "json_decision 'no_search' must not fetch anything"
    assert state["routed_search"] is False


def test_json_decision_unparsable_defaults_to_no_search(monkeypatch):
    patch_llm(
        monkeypatch,
        FakeLLM(
            responses=[
                AIMessage(content="I am not sure what to say..."),
                AIMessage(content="Прямой ответ"),
            ]
        ),
    )
    calls = patch_search(monkeypatch, results=[])
    state = build_graph(router_mode="json_decision").invoke(
        {"messages": [("user", PLAIN_QUESTION)]},
        config={"configurable": {"thread_id": "m4"}},
    )
    assert not calls
    assert state["routed_search"] is False
    assert state["messages"][-1].content == "Прямой ответ"


def test_unknown_router_mode_rejected():
    try:
        build_graph(router_mode="bogus")
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("unknown router_mode must raise")
