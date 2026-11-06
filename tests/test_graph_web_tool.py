"""Agent graph tests (tasks 3.1-3.4): fake LLM, fake search, no network."""

import json

from langchain_core.messages import AIMessage

import fakes
from fakes import FakeLLM, make_tool_call_message, patch_llm, patch_search
from graph import build_graph


def thread_config(item_id="t0"):
    return {"configurable": {"thread_id": item_id}}


def ask(question, extra=None):
    payload = {"messages": [("user", question)]}
    payload.update(extra or {})
    return payload


def test_search_runs_on_tool_call_and_results_return_to_llm(monkeypatch):
    """3.1: tool call -> search node -> results back to the LLM node."""
    fake = FakeLLM(
        responses=[
            make_tool_call_message("web_search", "погода Москва"),
            AIMessage(content="Ответ с результатами поиска"),
        ]
    )
    patch_llm(monkeypatch, fake)
    search_calls = patch_search(monkeypatch, results=fakes.canned_results())
    g = build_graph(router_mode="web_tool")
    state = g.invoke(ask("Какая погода в Москве?"), config=thread_config())

    assert search_calls == ["погода Москва"]  # search ran with the tool query
    assert state["routed_search"] is True
    assert state["search_results"][0]["snippet"] == "canned"
    tool_messages = [
        m for m in state["messages"] if type(m).__name__ == "ToolMessage"
    ]
    assert tool_messages, "results must flow back to the LLM as ToolMessage"
    assert "canned" in tool_messages[0].content
    assert fake.invocations[-1][-1].content == json.dumps(
        fakes.canned_results(), ensure_ascii=False
    ), "second LLM pass must receive the search tool result"
    assert state["messages"][-1].content == "Ответ с результатами поиска"


def test_no_tool_call_ends_without_search(monkeypatch):
    fake = FakeLLM(responses=[AIMessage(content="Привет! Чем помочь?")])
    patch_llm(monkeypatch, fake)
    search_calls = patch_search(monkeypatch, results=fakes.canned_results())
    g = build_graph(router_mode="web_tool")
    state = g.invoke(ask("Привет"), config=thread_config())

    assert search_calls == []  # DuckDuckGo untouched
    assert not state.get("search_results")
    assert state["messages"][-1].content == "Привет! Чем помочь?"
    assert len(fake.invocations) == 1  # no second pass


def test_soft_fail_keeps_answer_possible(monkeypatch):
    """3.2: failing search -> agent still answers with a could-not-retrieve note."""
    fake = FakeLLM(
        responses=[
            make_tool_call_message("web_search", "новости"),
            AIMessage(content="Не удалось найти свежие данные."),
        ]
    )
    patch_llm(monkeypatch, fake)
    from requests.exceptions import ConnectionError as ReqConnectionError

    patch_search(
        monkeypatch, error=ReqConnectionError("no internet")
    )
    g = build_graph(router_mode="web_tool")
    state = g.invoke(ask("Свежие новости"), config=thread_config())

    assert state["messages"][-1].content == "Не удалось найти свежие данные."
    tool_messages = [
        m for m in state["messages"] if type(m).__name__ == "ToolMessage"
    ]
    assert "Search could not be completed" in tool_messages[0].content
