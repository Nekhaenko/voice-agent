"""3.4: MemorySaver retains prior turns; fresh thread ids start fresh."""

from langchain_core.messages import AIMessage

from fakes import FakeLLM, patch_llm, patch_search
from graph import build_graph


def test_second_turn_sees_first_turn(monkeypatch):
    fake = patch_llm(monkeypatch, FakeLLM(responses=[
        AIMessage(content="Меня зовут Валера."),
        AIMessage(content="Ещё раз привет!"),
    ]))
    patch_search(monkeypatch, results=[])
    g = build_graph(router_mode="web_tool")
    cfg = {"configurable": {"thread_id": "turn-same"}}

    g.invoke({"messages": [("user", "Привет, меня зовут Валера")]}, config=cfg)
    g.invoke({"messages": [("user", "Как меня звали?")]}, config=cfg)

    second_call = fake.invocations[1]
    contents = [getattr(m, "content", m) for m in second_call]
    assert any("Валера" in str(c) for c in contents), (
        "second invocation must carry the first user turn"
    )
    assert any(
        isinstance(m, AIMessage) and m.content == "Меня зовут Валера."
        for m in second_call
    ), "second invocation must carry the first agent answer"


def test_fresh_thread_id_has_no_history(monkeypatch):
    fake = patch_llm(monkeypatch, FakeLLM(responses=[
        AIMessage(content="Первый ответ."),
        AIMessage(content="Отвечаю на новый вопрос."),
    ]))
    patch_search(monkeypatch, results=[])
    g = build_graph(router_mode="web_tool")

    g.invoke(
        {"messages": [("user", "секретный вопрос")]},
        config={"configurable": {"thread_id": "one"}},
    )
    fake.invocations.clear()
    g.invoke(
        {"messages": [("user", "обычный вопрос")]},
        config={"configurable": {"thread_id": "other"}},
    )

    contents = [getattr(m, "content", m) for m in fake.invocations[0]]
    assert not any("секретный" in str(c) for c in contents), (
        "a new thread must not see the other thread's messages"
    )
