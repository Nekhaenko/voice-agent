# Voice Assistant

Push-to-talk voice assistant: a Gradio web interface records your speech
(single toggle button — press to start, press again to stop), transcribes it
locally and offline with a Russian VOSK model, then routes the transcript
through a LangGraph agent. The agent answers with the `orcarouter/free` LLM
and performs a DuckDuckGo web search only when the LLM decides the question
needs fresh information. No text-to-speech: transcript and answer appear as
text in the conversation.

## Requirements

- Python 3.12+ with `pip`
- `ORCAROUTER_API_KEY` in the environment
- The Russian VOSK model, downloaded once into `models/`
- A microphone-accessible browser for the interface

## Setup

```bash
pip install -r requirements.txt

# Russian VOSK model (small, offline ASR):
mkdir -p models
curl -fL -o models/vosk-model-small-ru-0.22.zip \
  https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
unzip models/vosk-model-small-ru-0.22.zip -d models
rm models/vosk-model-small-ru-0.22.zip
```

## Run

```bash
export ORCAROUTER_API_KEY=sk-...   # required
python app.py
```

Open the printed URL in a browser, allow microphone access for the tab
(microphone permission errors appear in the browser's record panel), and use
the toggle button: press to record, press again to finish and get the answer.

## Configuration (environment variables)

| Variable              | Default                              | Purpose                                        |
|-----------------------|--------------------------------------|------------------------------------------------|
| `ORCAROUTER_API_KEY`  | — (required)                         | API key for the LLM endpoint                   |
| `ORCAROUTER_BASE_URL` | `https://api.orcarouter.ai/v1`       | OpenAI-compatible base URL                     |
| `ORCAROUTER_MODEL`    | `orcarouter/free`                    | Model id                                       |
| `ROUTER_MODE`         | `web_tool`                           | `web_tool` (bound search tool) or `json_decision` (structured fallback for endpoints with unreliable tool calls) |
| `VOSK_MODEL_PATH`     | `models/vosk-model-small-ru-0.22`    | Local VOSK model path                          |
| `SEARCH_MAX_RESULTS`  | `5`                                  | DuckDuckGo results per query                   |
| `EVAL_THRESHOLD`      | `0.8`                                | Minimum router-quality accuracy to pass        |

## Router-quality evaluation

Run offline; no DuckDuckGo traffic and no final answers are generated during
an eval run (results never contact the search backend):

```bash
python -m eval.run                # uses eval/dataset.json and EVAL_THRESHOLD
python -m eval.run --router-mode json_decision --threshold 0.9 \
  --report eval-last.json
```

Exit codes: `0` pass, `1` below threshold, `2` configuration/dataset error.

## Layout

```
app.py       Gradio push-to-talk interface
asr.py       WAV conversion (16 kHz mono PCM) + batch VOSK transcription
graph.py     LangGraph agent: LLM node, conditional DuckDuckGo search, MemorySaver
config.py    Environment-driven settings (no endpoint specifics elsewhere)
eval/        Router-quality dataset, loader, runner, report, CLI
tests/       pytest suite (fake LLM/search; no network needed)
models/      Local VOSK model (downloaded, not committed)
```

## Tests

```bash
python -m pytest tests/ -q
```

Note: the Russian speech fixture `tests/fixtures/sample_ru.wav` was generated
with `gTTS` (a fixture-generation tool only; it is NOT a runtime dependency).
