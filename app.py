"""Gradio push-to-talk interface (specs/voice-assistant).

One toggle button starts/stops the mic recording (design D1: non-streaming
gr.Audio driven by the toggle's JS click on the component's own record/stop
button). When recording stops, the recorded clip is batch-transcribed with
VOSK and routed through the LangGraph agent; the transcript and the agent's
text answer join the chat history.

Browser-side state flows: the toggle's JS returns the new recording flag,
which becomes the first input to the Python handler (Gradio contract); the
DOM query targets the component's record/stop buttons directly.
"""

from __future__ import annotations

import io
import uuid

import gradio as gr
import numpy as np
from langgraph.checkpoint.memory import MemorySaver

from asr import ASREmptyResult, transcribe
from config import settings
from graph import build_graph

TOGGLE_START = "_ Record"
TOGGLE_STOP = "Stop recording"

TOGGLE_JS = """() => {
  const root = document.querySelector('#voice-mic');
  if (!root) return false;
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };
  // While recording a visible stop button appears; otherwise the record
  // button. The returned value ('was recording active?') becomes the toggle
  // handler's input, so Python state and DOM always agree.
  const stops = [...root.querySelectorAll('.stop-button')].filter(visible);
  if (stops.length) { stops[0].click(); return true; }
  const records = [...root.querySelectorAll('.record-button')].filter(visible);
  if (records.length) { records[0].click(); }
  return false;
}"""


def toggle_recording(recording: bool):
    """Flip the recording state and the toggle label (spec: single button)."""
    started = not bool(recording)
    return (
        started,
        TOGGLE_STOP if started else TOGGLE_START,
        "Recording... speak, then press the button again to finish."
        if started
        else "Recording stopped. Processing the audio...",
    )


def _audio_to_bytes(audio) -> bytes:
    """Accept bytes, a filepath, or a (rate, numpy) tuple from gr.Audio."""
    if audio is None:
        return b""
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    if isinstance(audio, str):
        with open(audio, "rb") as handle:
            return handle.read()
    if isinstance(audio, (tuple, list)):
        rate, data = audio
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(data), int(rate), format="WAV")
        return buffer.getvalue()
    raise TypeError(f"Unsupported audio payload: {type(audio)!r}")


def _agent_answer(thread_id: str, question: str) -> str:
    """Run the agent graph; raise on configuration errors (UI reports them)."""
    # settings() raises when ORCAROUTER_API_KEY is missing: process_exchange
    # surfaces that as a configuration error instead of answering (spec).
    cfg = settings()
    thread = {"configurable": {"thread_id": thread_id}}
    result = build_graph(
        router_mode=cfg["router_mode"],
        checkpointer=MemorySaver(),
    ).invoke({"messages": [("user", question)]}, config=thread)

    last = result["messages"][-1]
    content = getattr(last, "content", last)
    if isinstance(content, (list, tuple)):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def process_exchange(audio, history: list, thread_id: str | None):
    """Transcribe a finished recording and reply (spec conversation loop)."""
    history = list(history or [])
    if not thread_id:
        thread_id = uuid.uuid4().hex  # fresh browser session = fresh thread
    try:
        wav = _audio_to_bytes(audio)
        if not wav:
            raise ASREmptyResult("No audio was received from the recording.")

        text = transcribe(wav, settings()["vosk_model_path"])
        answer = _agent_answer(thread_id, text)
        history += [
            {"role": "user", "content": text},
            {"role": "assistant", "content": answer},
        ]
        status = "Answer ready. Press the button to record again."
    except ASREmptyResult as exc:
        # Nothing recognized: do NOT query the agent (spec, "Empty recording").
        status = f"Nothing recognized: {exc}. Press the button to try again."
    except RuntimeError as exc:
        status = f"Error: {exc}"
    except Exception as exc:  # keep the UI alive on any unexpected failure
        status = f"Unexpected error: {exc}"
    return history, status, thread_id


def build_block() -> gr.Blocks:
    with gr.Blocks(title="Russian Voice Assistant") as demo:
        thread_id = gr.State(value=None)
        recording = gr.State(False)
        chat = gr.Chatbot(label="Conversation", height=480)
        status = gr.Markdown("Press the button to record.")

        mic = gr.Audio(
            label="Push to talk",
            sources=["microphone"],
            elem_id="voice-mic",
        )
        toggle = gr.Button(TOGGLE_START, elem_id="voice-toggle")
        gr.Markdown(
            "If pressing the toggle does nothing, allow microphone access for "
            "this browser tab (spec: microphone permission errors are "
            "reported in the browser record panel and here in the status area)."
        )

        toggle.click(
            toggle_recording,
            inputs=[recording],
            outputs=[recording, toggle, status],
            js=TOGGLE_JS,
        )
        mic.stop_recording(
            process_exchange,
            inputs=[mic, chat, thread_id],
            outputs=[chat, status, thread_id],
        )

    return demo


def main() -> None:
    build_block().launch()


if __name__ == "__main__":
    main()
