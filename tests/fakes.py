"""Fake LLM plumbing for agent tests: no real orcarouter calls happen."""

from langchain_core.messages import AIMessage

import graph


class FakeLLM:
    """Records invocations, returns queued/callable responses."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.invocations = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, *args, **kwargs):
        self.invocations.append(list(messages))
        if not self.responses:
            return AIMessage(content="no answer configured")
        response = self.responses.pop(0)
        if callable(response):
            response = response(messages)
        return response

    @property
    def all_inputs(self):
        return self.invocations


def make_tool_call_message(tool_name: str, query: str, call_id: str = "c1"):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": tool_name, "args": {"query": query}, "id": call_id}
        ],
    )


def patch_llm(monkeypatch, fake: FakeLLM):
    monkeypatch.setattr(graph, "_make_llm", lambda cfg, **_: fake)
    return fake


def patch_search(monkeypatch, results=None, error=None):
    calls = []

    def _search(query, max_results):
        calls.append(query)
        if error is not None:
            raise error
        return results

    monkeypatch.setattr(graph, "_duckduckgo_search", _search)
    return calls


def canned_results():
    return [
        {"title": "News", "url": "https://example.com/1", "snippet": "canned"},
        {"title": "More", "url": "https://example.com/2", "snippet": "data"},
    ]
