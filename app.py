

import time
import uuid
import threading
from contextlib import asynccontextmanager
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, Query, Body
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from openai import OpenAI

import config
import voice
import llm_interface
import vision

def _warm_vision():
    t = time.time()
    try:
        vision.warm()
        # flush: stdout is block-buffered when uvicorn is redirected to a log
        # file, so without this the line can sit unseen for a long time.
        print(f"[vision] warm complete in {time.time() - t:.1f}s", flush=True)
    except Exception as e:
        # Warming is an optimisation; a failure here must not stop the server.
        # The first /observe will simply load the model the old way.
        print(f"[vision] warm failed after {time.time() - t:.1f}s: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm on a daemon thread so uvicorn reports ready immediately. vision._lock
    # makes an early /observe wait for this load instead of starting a second one.
    threading.Thread(target=_warm_vision, name="vision-warm", daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)
client = OpenAI()


@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "service": "rio-phase1"}


# Most recent completed talk turn, for the dashboard to pick up after playback.
# RIO's reply text only exists once the LLM stream has finished, which is after
# the response headers are already on the wire — so it cannot be a header.
# Single-driver assumption: this holds one turn for the whole process.
_last_talk = {
    "talk_id": None,
    "transcript": "",
    "reply": "",
    "session_id": None,
    "audio_bytes": 0,
    "latency_ms": 0.0,
    "t": 0.0,
}


@app.get("/last_talk")
def last_talk_endpoint():
    """Latest finished talk turn. The dashboard polls this for RIO's reply text."""
    return dict(_last_talk)


@app.post("/talk")
async def talk(audio: UploadFile, session_id: str = Query(default=None)):
    t0 = time.time()

    audio_bytes = await audio.read()

    input_path = "/tmp/rio_input.webm"
    with open(input_path, "wb") as f:
        f.write(audio_bytes)

    with open(input_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model=config.OPENAI_STT_MODEL,
            file=f,
        ).text

    t1 = time.time()

    print({
        "transcript": transcript,
        "whisper_seconds": round(t1 - t0, 3),
    })

    talk_id = uuid.uuid4().hex[:12]

    def streamer():
        buffer = ""
        reply_parts = []
        audio_len = 0

        for token in llm_interface.generate_stream(transcript):
            buffer += token
            reply_parts.append(token)

            if any(buffer.rstrip().endswith(p) for p in [".", "!", "?"]):
                for chunk in voice.synthesize_stream(buffer):
                    audio_len += len(chunk)
                    yield chunk
                buffer = ""

        if buffer.strip():
            for chunk in voice.synthesize_stream(buffer):
                audio_len += len(chunk)
                yield chunk

        # Stream is done: publish the turn for /last_talk and the session log.
        reply = "".join(reply_parts).strip()
        latency_ms = (time.time() - t0) * 1000
        _last_talk.update({
            "talk_id": talk_id,
            "transcript": transcript,
            "reply": reply,
            "session_id": session_id,
            "audio_bytes": audio_len,
            "latency_ms": round(latency_ms, 1),
            "t": time.time(),
        })
        sessions.log_talk(session_id, transcript, reply, audio_len, latency_ms)

    # The transcript is known before streaming starts, so it can ride on a header.
    # URL-encoded because headers are latin-1 only and transcripts are UTF-8.
    headers = {
        "X-Transcript": quote(transcript, safe=""),
        "X-Talk-Id": talk_id,
        "Access-Control-Expose-Headers": "X-Transcript, X-Talk-Id",
    }
    return StreamingResponse(streamer(), media_type="audio/mpeg", headers=headers)

from fastapi import File
import sessions

@app.post("/observe")
async def observe(image: UploadFile = File(...), session_id: str = Query(default=None)):
    import time as _t
    _t0 = _t.time()
    image_bytes = await image.read()
    observation = vision.observe(image_bytes)
    sessions.log_observe(session_id, len(image_bytes), observation, (_t.time() - _t0) * 1000)
    return {"observation": observation}

# --- Phase 2.5 session endpoints ---

@app.post("/session/start")
def session_start_endpoint(metadata: dict = Body(default=None, embed=True)):
    sid = sessions.start_session(metadata=metadata)
    return {"session_id": sid}


@app.post("/session/end")
def session_end_endpoint(session_id: str = Query(...)):
    closed = sessions.end_session(session_id)
    return {"closed": closed}


@app.post("/session/mark")
def session_mark_endpoint(session_id: str = Query(...), tag: str = Query(...), note: str = Query(default="")):
    sessions.mark(session_id, tag, {"note": note} if note else None)
    return {"marked": tag}


@app.get("/session/{session_id}")
def session_view_endpoint(session_id: str):
    import json as _json
    path = sessions.SESSIONS_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return {"error": "not found", "session_id": session_id}
    events = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                events.append(_json.loads(line))
            except Exception:
                pass
    return {"session_id": session_id, "events": events, "active": sessions.is_active(session_id)}
