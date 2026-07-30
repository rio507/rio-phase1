

import math
import time
import uuid
import threading
from contextlib import asynccontextmanager
from urllib.parse import quote
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool

load_dotenv()

from fastapi import FastAPI, UploadFile, Query, Body, Form
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

import config
import voice
import llm_interface
import vision
import perceive
# Import order matters: perceive lends vision's resident Qwen3-VL to
# headway.anchor at import time, so headway.live's anchor path finds a provider
# already installed and never pulls a second copy of the weights.
from headway import live as headway_live
from headway import live_policy
from headway import lanes as headway_lanes

def _warm_vision():
    t = time.time()
    try:
        vision.warm()
        # /perceive's second model (Depth Anything V2 Metric-Small). Warmed on
        # the same thread, after Qwen, so the overlay's first frame does not pay
        # the depth load while a drive is already running.
        perceive.warm()
        # UFLDv2 lane geometry. Same reasoning, and it matters more here: this
        # one runs on EVERY headway frame, so an unwarmed first call would put
        # a ~2 s weight load inside a live frame instead of a 2 ms forward.
        # A missing checkpoint must not take the server down -- headway falls
        # back to the static trapezoid and says so once.
        try:
            headway_lanes.warm()
            print("[vision] lane detection warm", flush=True)
        except Exception as e:
            print(f"[vision] lane detection unavailable: {e}", flush=True)
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

# The pre-rendered red-tier clips live in static/audio/ and must be reachable
# for the browser to preload them. "/" keeps serving index.html by hand so the
# mount does not shadow it.
app.mount("/static", StaticFiles(directory="static"), name="static")


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


@app.post("/perceive")
async def perceive_endpoint(image: UploadFile = File(...), session_id: str = Query(default=None),
                            debug: int = Query(default=0)):
    """Structured perception for the dashboard overlay. /observe is unchanged.

    Returns the same caption /observe would, plus boxes, per-object distance and
    the ego corridor, so the Camera panel can draw what RIO is looking at.
    """
    import time as _t
    _t0 = _t.time()
    image_bytes = await image.read()
    result = perceive.perceive(image_bytes, debug=bool(debug))
    sessions.log_perceive(session_id, len(image_bytes), result, (_t.time() - _t0) * 1000)
    return result

# --- Live headway monitoring (v3) ---

def _opt_float(v):
    """Form scalars arrive as strings; '', 'null' and 'NaN' all mean 'no value'.

    Distinguishing missing from zero matters here: 0 m/s is a real, meaningful
    speed (stopped) and None means the browser has no fix. Coercing one into
    the other would either coach in a parking lot or suppress at a red light.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    s = str(v).strip()
    if not s or s.lower() in ("null", "none", "nan", "undefined"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


@app.post("/headway_frame")
async def headway_frame_endpoint(
    image: UploadFile = File(...),
    session_id: str = Query(default=None),
    v_host: str = Form(default=None),
    v_host_age_s: str = Form(default=None),
    frame_t: str = Form(default=None),
):
    """One live headway frame: track -> depth -> filter -> band -> voice.

    NO Qwen call per frame. The lead box is carried by CSRT and re-anchored only
    on session start, track loss, tracker drift, or anchor staleness — see
    headway/live.py.

    Runs in a threadpool because the work is a blocking GPU pass, and a
    per-session non-blocking lock drops a frame rather than queueing it: at
    ~2 fps a queued frame would be measured against a dt that has already
    passed, which corrupts the very velocity estimate the warning rests on.
    """
    t0 = time.time()
    image_bytes = await image.read()
    key = session_id or "default"
    session = headway_live.get_session(key, use_qwen=config.VISION_ENABLED)

    if not session.lock.acquire(blocking=False):
        return {"ok": False, "skipped": "busy", "band": session.policy.band}
    try:
        result = await run_in_threadpool(
            session.process, image_bytes,
            _opt_float(v_host), _opt_float(v_host_age_s), _opt_float(frame_t),
        )
    except Exception as e:
        print(f"[headway] frame failed: {e}", flush=True)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        session.lock.release()

    sessions.log_headway(session_id, result, (time.time() - t0) * 1000)
    # Lane departure is logged, not spoken. It gets its own event kind so a
    # review pass can pull the handful of excursions out of a whole drive; see
    # sessions.log_lane_drift and the scope note in headway/lanes.py.
    if (result.get("lane_drift") or {}).get("drift"):
        sessions.log_lane_drift(session_id, result)
    return result


@app.post("/headway_reset")
def headway_reset_endpoint(session_id: str = Query(default=None)):
    """Drop a session's tracker/filter/policy — used when a video test restarts."""
    return {"reset": headway_live.reset_session(session_id or "default")}


@app.get("/headway_sessions")
def headway_sessions_endpoint():
    return headway_live.active_sessions()


@app.get("/headway_voice")
def headway_voice_endpoint(line: str = Query(...)):
    """Live TTS for the amber-tier lines.

    Only the calm tier comes through here. The red tier is served as static
    pre-rendered clips precisely so it never waits on this round-trip, and
    allowing an arbitrary `line` would turn a deterministic warning channel into
    a text-to-speech endpoint — so the key is looked up in the policy's own
    table and anything else is refused.
    """
    if live_policy.LINE_AUDIO.get(line) != "tts":
        return {"error": "unknown or non-TTS line", "line": line}
    text = live_policy.LINE_TEXT[line]
    return StreamingResponse(voice.synthesize_stream(text), media_type="audio/mpeg",
                             headers={"Cache-Control": "no-store"})


# --- Phase 2.5 session endpoints ---

@app.post("/session/start")
def session_start_endpoint(metadata: dict = Body(default=None, embed=True)):
    sid = sessions.start_session(metadata=metadata)
    return {"session_id": sid}


@app.post("/session/end")
def session_end_endpoint(session_id: str = Query(...)):
    closed = sessions.end_session(session_id)
    # Drop the live headway state with the session. Without this a tracker,
    # Kalman filter and cooldown table survive for every drive the process has
    # ever seen, and a re-used session id would resume mid-warning.
    headway_live.reset_session(session_id)
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
