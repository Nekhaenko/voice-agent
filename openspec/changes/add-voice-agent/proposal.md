# Proposal

## Why

The project has no implementation yet. The user wants a local voice assistant: a Gradio interface that records speech on a button press, transcribes it with a locally installed VOSK model, and routes the transcript through a LangGraph agent (orcarouter LLM with a conditional DuckDuckGo search node) before showing the agent's answer. Because the agent's answer quality depends heavily on the LLM's "is a web search needed?" routing decision, the change must also deliver a way to measure how well that router performs.

## What Changes

- Add a Gradio web interface with a push-to-talk toggle button: first press starts recording, second press stops recording.
- Add offline speech recognition with a locally installed **Russian** VOSK model; the recorded utterance is batch-decoded once after recording stops (single-shot, no live partials).
- Add a LangGraph conversation agent backed by the OpenAI-compatible `orcarouter/free` model (base URL `https://api.orcarouter.ai/v1`, key from `ORCAROUTER_API_KEY`).
- Add a DuckDuckGo web-search node that runs only when the LLM decides a search is needed; the LLM decides via a bound search tool on a conditional edge.
- Persist conversation state with a LangGraph checkpointer so multi-turn dialogue is remembered (in-process for v1; no persistent store).
- Add no text-to-speech (out of scope for v1): transcript and answer appear as text in the interface.
- Add an evaluation harness that measures **router quality only**: whether the "search needed" decision is correct against a labeled question set.

## Capabilities

### New Capabilities

- `voice-assistant`: The user-facing interaction loop — push-to-talk recording via the Gradio toggle button, batch VOSK transcription, and display of the user's transcript and the agent's text response.
- `conversation-agent`: The LangGraph agent pipeline — LLM answering over conversation state, conditional DuckDuckGo search node when the LLM decides a search is needed, and checkpointer-backed multi-turn memory.
- `router-quality`: Evaluation of the agent's search-routing decisions — a labeled question set, a runner that skips real searches while reusing the real routing path, and pass/fail reporting.

### Modified Capabilities

<!-- none: greenfield project, openspec/specs/ is empty -->

## Impact

- New Python application code (none exists today); no downstream systems are affected.
- New dependencies: `gradio`, `vosk`, `langgraph`, `langchain-openai` (OpenAI-compatible, pointed at orcarouter), a DuckDuckGo search library (`ddgs`), and a Russian VOSK model downloaded locally.
- Requires `ORCAROUTER_API_KEY` in the environment and microphone access from the browser running the Gradio UI.
- New eval dataset file plus runner/reporting code owned by `router-quality`.
