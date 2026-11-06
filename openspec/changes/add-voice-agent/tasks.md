# Tasks

## 1. Project setup and dependencies

- [x] 1.1 Create package layout (`app.py`, `asr.py`, `graph.py`, `config.py`, `eval/`) and verify all files exist in place
- [x] 1.2 Add dependencies (`gradio`, `vosk`, `soundfile`, `langgraph`, `langchain-openai`, `ddgs`) and verify the project imports each, plus `ORCAROUTER_API_KEY` resolves from the environment in `config.py`
- [x] 1.3 Download the Russian VOSK model locally and record its filesystem path in configuration (env var override); verify a script loads the model successfully

## 2. Local speech recognition (voice-assistant backend)

- [x] 2.1 Implement WAV-to-16-kHz-mono-PCM conversion (`soundfile`) and test with a browser-mic WAV fixture of differing rate/channels
- [x] 2.2 Implement the batch VOSK transcription function and test it returns the expected Russian text for the fixture audio
- [x] 2.3 Test empty/non-speech audio returns an explicit "nothing recognized" result and is stubbed as such (verify no downstream agent call)

## 3. LangGraph agent (conversation-agent backend)

- [x] 3.1 Build the graph: LLM node with bound `web_search` tool, conditional edge on `tool_calls`, search node calling `ddgs`, `END` when no tool call; verify with a mocked tool-call message that the search node runs and results return to the LLM node
- [x] 3.2 Implement the search node with soft-fail behavior (network error keeps the graph on a "couldn't retrieve" note) and test with a simulated failing search
- [x] 3.3 Add the `router_mode` config switch (`web_tool` vs `json_decision`) and test that both modes produce a routing decision from the same dataset sample used in Group 5
- [x] 3.4 Attach `MemorySaver` and verify a scripted two-turn conversation retains prior exchanges while a fresh thread id does not

## 4. Gradio interface (voice-assistant UI)

- [ ] 4.1 Wire the toggle-button push-to-talk loop around `gr.Audio` (mic source, non-streaming) and verify recording starts on press 1 and hands the recorded WAV on press 2
- [x] 4.2 Display the transcript and agent response for each exchange and keep prior exchanges visible across turns (verify with a scripted turn sequence)
- [x] 4.3 Surface microphone-denied and missing/invalid API key as clear errors in the interface (verify each by triggering the failure path)
- [x] 4.4 Document how to run the app (env vars, model path, run command) in the README; verify the documented command starts the interface

## 5. Router quality evaluation (router-quality)

- [x] 5.1 Create the labeled dataset of Russian questions (`search` / `no_search`, both present, at least one per class + a meaningful spread; ≥ 10 total) and test a loader rejects unlabeled or unknown-label items fast
- [x] 5.2 Implement the eval runner: build a variant of the graph with the search node stubbed to record the decision and both nodes stopping before any network call; verify the stub swap throws if a request is attempted during eval
- [x] 5.3 Implement the report: per-item expected vs actual with hit/miss plus aggregate accuracy against a configurable threshold; verify the report file content on a small sample and that a below-threshold run exits non-zero

## 6. End-to-end integration

- [ ] 6.1 Run the full loop against the real endpoint: record → recognize Russian → routed agent answer (both a `no_search` question and a `search` question), verify both land in the interface and only one triggers a search
- [ ] 6.2 Run the router-quality eval on the full dataset with the real `router_mode`, verify all items report decision/hit/miss and the pass/fail verdict matches the configured threshold
