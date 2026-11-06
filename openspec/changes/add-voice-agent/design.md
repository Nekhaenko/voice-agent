# Design

## Context

Greenfield: the repo holds only the OpenSpec scaffold, so every module is new. Constraints settled during exploration and binding here:

- Single-process Python app; the UI is a Gradio server and the agent runs in the same process.
- Audio capture happens in the browser (Gradio `gr.Audio` with the microphone source); the server receives the recorded clip as WAV bytes.
- Transcription is local and offline: a locally installed VOSK model, Russian (`vosk-model-small-ru-0.22` as the default, overridable by config/env because the exact model is an installation matter, not a code matter).
- No TTS in v1. One-shot transcription, no streaming partials.
- LLM via the OpenAI-compatible endpoint at `https://api.orcarouter.ai/v1`, model `orcarouter/free`, key from `ORCAROUTER_API_KEY` (the user's given access snippet).
- Search must go through DuckDuckGo inside a LangGraph node, gated by the LLM's own decision.
- Multi-turn memory via checkpointer, in-process for v1.

## Goals / Non-Goals

**Goals:**
- Working push-to-talk loop: record → transcribe → routed agent → display transcript + answer.
- Router decisions made by the model, honoring the tool-call burden on an unknown-quality free model with a deterministic fallback.
- An eval run that verifies the routing path is correct without touching the internet, and that gates on aggregate accuracy.

**Non-Goals:**
- Streaming partial transcripts while speaking.
- Text-to-speech or any spoken output.
- Persisting conversations across app restarts; Gradio session/thread management beyond one checkpointer thread id per session.
- Answer-quality evaluation; only router quality is measured in this change.

## Decisions

**D1 — Gradio audio delivery: non-streaming `gr.Audio` component.**
Because transcription is explicitly one-shot batch, the simplest delivery is a WAV blob received after the user presses stop. We use a state-backed toggle button around the `gr.Audio` component so that one UI control serves "start/stop recording" exactly as specified, and transcription is triggered on the component's change event.
*Alternative considered:* streaming chunks into a persistent VOSK recognizer. Rejected: contradict the decided UX and add lifecycle complexity out of scope for v1.

**D2 — WAV → 16 kHz mono PCM via `soundfile`, then VOSK in batch.**
VOSK requires 16 kHz mono 16-bit PCM, and browser recordings arrive in whatever rate/channels the browser produced, so decoding through `soundfile` lets us feed data straight to `vosk.Model/KaldiRecognizer` without pinning expectations of a particular audio backend. We read frames from the buffer fully before constructing the final text.
*Alternative considered:* ffmpeg CLI wrapper. Rejected: extra external binary dependency when `soundfile` suffices.

**D3 — Router mechanism: bound `web_search` tool on the LLM; structured-decision fallback behind a config switch.**
The standard conditional-edge route: bind the search tool, `add_conditional_edges` route to the search node when the last message contains `tool_calls`; `END` otherwise; search results go back to the LLM node for the second pass that weaves results into the answer. The function takes a `router_mode` toggled by `web_tool` or `json_decision`. Rationale: tool calling is cleaner if the free endpoint supports it; the JSON round trip is deterministic and safe on any endpoint if it doesn't.
*Alternative considered:* always-search. Rejected: contradicts the agreed behavior of answering simple questions directly without a web request.

**D4 — Search library: `ddgs` (the DuckDuckGo search package) called directly inside the search node.**
Direct call keeps the node's signature simple and does not pull the LangChain-community tool chain in just for one dependency wrapper. The node returns a compact list of (title, url, digest body) pairs into graph state.
*Alternative considered:* `langchain-community`'s `DuckDuckGoSearchRun`. Rejected: heavier dependency surface, no added function on top of direct queries.

**D5 — Checkpointing: `MemorySaver` keyed by per-interface-session thread id.**
`langgraph.checkpoint.memory.MemorySaver` gives multi-turn memory inside one process lifetime, and each interface session maps to its own thread id, matching "new session starts fresh".
*Alternative considered:* `SqliteSaver` from start. Rejected for v1: it buys persistence across restarts, needs no durable storage yet, and can be added later with an interface-layer change not a graph change.

**D6 — Code layout: small package, `app.py` entrypoint.**
`app.py` (Gradio wiring), `asr.py` (VOSK adapter), `graph.py` (LangGraph build incl. search node), `config.py` (env/config, incl. `router_mode`, model id, paths), `eval/` (dataset + runner + reporting for router quality). Layout matches the boundaries of the three capabilities, yet stays in one package as required.

**D7 — Eval mechanics: router-path reuse with the search node stubbed, dataset in a YAML file.**
The eval runner builds the same graph, but with the search node replaced by a no-internet stub that signals "search would have run" instead of requesting anything, and a final-answer stub that stops (no answer text requested) — satisfying a key requirement to verify the routing path end to end without sending web traffic or running the LLM output pass. The dataset (JSON) contains Russian questions labeled `search` / `no_search`, and the runner writes item-level including a clear pass/fail verdict per row plus aggregate accuracy against a configurable minimum, exiting nonzero on threshold failure, so a possible follow-up can gate on it.
*Alternative considered:* monkey-patching HTTP. Rejected: fragile; the graph-query separation is cleaner and verifiable.

## Risks / Trade-offs

- **[Free model may not emit tool calls reliably]** → D3 keeps the fallback `json_decision` mode a config switch away, and each mode can be verified against the same eval dataset.
- **[orcarouter endpoint availability/quantities change]** → everything endpoint-specific lives in config; no hard-coded model or URL inside the graph/msg-flow code.
- **[Russian VOSK small model lowers accuracy on noisy/slurring input]** → for v1 accuracy is accepted; the config exposes a bigger local VOSK model if the accuracy matters later.
- **[Gradio mic requires browser mic permission]** → interface surfaces a clear error at start time; we show a hint in the UI (e.g. microphone access blocked = nothing records).
- **[DuckDuckGo rate limiting/instability]** → the search node fails softly: the agent answers saying data retrieval failed, matching the spec scenario.
- **[MemorySaver loses history on restart]** → accepted v1 trade-off, matches spec of "new session starts fresh"; swapping Savers later is isolated to config/setup.

## Open Questions

- Exact installed VOSK model path/filename (env var, resolved on the machine where it's run) — no impact on graph or interface structure.
- Whether `orcarouter/free` supports tool calls reliably — answerable only by a first run against the real endpoint; D3 keeps either outcome workable.
