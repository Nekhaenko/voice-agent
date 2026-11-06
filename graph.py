"""LangGraph conversation agent (design D3-D5).

Graph shape (router_mode="web_tool", the default):

    START -> llm_node --(conditional)--> (tool_calls? search_node -> llm_node : END)

router_mode="json_decision" routes on a structured yes/no decision instead
of tool calls: a decide node asks the model for {"search": true|false};
"yes" runs search_node then an answer pass; "no" answers directly.

The eval runner reuses this exact routing path by compiling with
interrupt_after so search/answer nodes never execute (design D7).

All endpoint specifics come from config.settings(); nothing is hard-coded
here (design risk: endpoint specifics stay in config).
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, TypedDict

from ddgs import DDGS
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from config import settings

SEARCH_TOOL_NAME = "web_search"

SYSTEM_PROMPT = (
    "You are a helpful voice assistant that answers in Russian. "
    f"Call the '{SEARCH_TOOL_NAME}' tool when the question needs fresh or "
    "specific information you cannot reliably supply from your own "
    "knowledge (news, prices, weather, facts likely to have changed). "
    "For greetings, introductions, and reasoning on general knowledge, "
    "answer directly without the tool."
)

#: Set by the router-quality eval runner. When armed, any attempt to
#: perform a real DuckDuckGo search raises immediately, guaranteeing no
#: web request can slip through during an eval run (specs/router-quality).
_eval_search_guard = False


def arm_eval_search_guard() -> None:
    global _eval_search_guard
    _eval_search_guard = True


def release_eval_search_guard() -> None:
    global _eval_search_guard
    _eval_search_guard = False


class AgentState(TypedDict):
    """Conversation state flowing through the graph."""

    messages: Annotated[list[BaseMessage], operator.add]
    # DuckDuckGo results fetched by the search node (title, url, snippet).
    search_results: Annotated[list[dict], operator.add]
    # Machine-readable routing outcome, consumed by eval and tests.
    routed_search: bool


def _make_llm(cfg: dict, bind_search_tool: bool) -> ChatOpenAI:
    llm = ChatOpenAI(
        model=cfg["model_name"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
    )
    if bind_search_tool:
        llm = llm.bind_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": SEARCH_TOOL_NAME,
                        "description": (
                            "Search DuckDuckGo for fresh information."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": (
                                        "Search query (Russian or English)."
                                    ),
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ]
        )
    return llm


def _duckduckgo_search(query: str, max_results: int) -> list[dict]:
    """Run one DuckDuckGo search, returning compact results (design D4)."""
    if _eval_search_guard:
        raise RuntimeError(
            "Eval network violation: a real DuckDuckGo search was attempted "
            "during an eval run."
        )
    with DDGS() as ddgs:
        hits = ddgs.text(query, max_results=max_results)
        return [
            {
                "title": hit.get("title", ""),
                "url": hit.get("href") or hit.get("url", ""),
                "snippet": hit.get("body", ""),
            }
            for hit in hits
        ]


def _unavailable_results(reason: str) -> list[dict]:
    return [{"title": "", "url": "", "snippet": reason}]


def _search_node(state: AgentState) -> dict[str, Any]:
    """Execute the DuckDuckGo search the router requested (soft-fail).

    A failing search must not crash the graph: the agent still answers,
    saying fresh information could not be retrieved (spec "Search fails").
    """
    last = state["messages"][-1]
    tool_call = None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        tool_call = last.tool_calls[0]

    try:
        cfg = settings()
        query = (
            tool_call["args"].get("query", "fresh information")
            if tool_call
            else _question_from_messages(state)
        )
        results = _duckduckgo_search(
            query, max_results=cfg.get("search_max_results", 5)
        )
        if not results:
            results = _unavailable_results("The search returned no results.")
    except RuntimeError:
        raise  # guard violations are fatal, not soft failures
    except Exception as exc:  # soft fail: keep the graph answerable
        results = _unavailable_results(
            f"Search could not be completed: {exc}"
        )

    updates: dict[str, Any] = {
        "search_results": results,
        "routed_search": True,
    }
    if tool_call is not None:
        updates["messages"] = [
            ToolMessage(
                content=json.dumps(results, ensure_ascii=False),
                tool_call_id=tool_call["id"],
            )
        ]
    return updates


def _question_from_messages(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if getattr(message, "content", None):
            return str(message.content)
    return ""


def _llm_node(state: AgentState) -> dict[str, Any]:
    """Answer (or emit a tool call) using the LLM with the search tool bound."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = _make_llm(settings(), bind_search_tool=True).invoke(messages)
    return {"messages": [response]}


def _extract_json_decision(content: Any) -> bool:
    """Parse the decision payload defensively; anything unparsable = no search."""
    if isinstance(content, (list, tuple)):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    text = str(content or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return False
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return False
    search = payload.get("search", False)
    return search if isinstance(search, bool) else False


def _decide_node(state: AgentState) -> dict[str, Any]:
    """json_decision mode: ask the model for a strict yes/no routing call."""
    llm = _make_llm(settings(), bind_search_tool=False)
    probing_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
        ("user", 'Reply ONLY with strict JSON: {"search": true|false}'),
    ]
    response = llm.invoke(probing_messages)
    return {"routed_search": _extract_json_decision(response.content)}


def _route_on_tool_calls(state: AgentState) -> str:
    """Conditional edge (web_tool): enter search when the model asked for it."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "search"
    return END


def _route_on_decision(state: AgentState) -> str:
    return "search" if state.get("routed_search") else "search_not_needed"


def _answer_node(state: AgentState) -> dict[str, Any]:
    """json_decision mode final answer (with search results if any)."""
    llm = _make_llm(settings(), bind_search_tool=False)
    if state.get("search_results"):
        results = json.dumps(state["search_results"], ensure_ascii=False)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            ("user", _question_from_messages(state)),
            (
                "user",
                "DuckDuckGo results to incorporate into the answer:\n"
                + results,
            ),
        ]
    else:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(
            state["messages"]
        )
    return {"messages": [llm.invoke(messages)]}


def build_graph(
    router_mode: str | None = None,
    checkpointer: MemorySaver | None = None,
    interrupt_after: list[str] | None = None,
    node_overrides: dict[str, Any] | None = None,
):
    """Build the conversation agent graph per the configured router_mode.

    interrupt_after + node_overrides let the router-quality eval stop right
    after the decision-carrying node, before any search or answer pass runs
    (design D7). Live code passes neither.
    """
    mode = router_mode or settings().get("router_mode", "web_tool")
    overrides = node_overrides or {}
    builder = StateGraph(AgentState)

    if mode == "web_tool":
        builder.add_node("llm", overrides.get("llm", _llm_node))
        builder.add_node("search", overrides.get("search", _search_node))
        builder.add_edge(START, "llm")
        builder.add_conditional_edges(
            "llm", _route_on_tool_calls, {"search": "search", END: END}
        )
        builder.add_edge("search", "llm")
    elif mode == "json_decision":
        builder.add_node("decide", overrides.get("decide", _decide_node))
        builder.add_node("search", overrides.get("search", _search_node))
        builder.add_node("answer", overrides.get("answer", _answer_node))
        builder.add_edge(START, "decide")
        builder.add_conditional_edges(
            "decide",
            _route_on_decision,
            {"search": "search", "search_not_needed": "answer"},
        )
        builder.add_edge("search", "answer")
        builder.add_edge("answer", END)
    else:
        raise ValueError(f"Unknown router_mode: {mode!r}")

    return builder.compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_after=interrupt_after,
    )
