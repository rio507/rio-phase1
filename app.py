

import math
import os
import sys
import time
import uuid
import threading
from contextlib import asynccontextmanager
from pathlib import Path
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
import nav
import tires
import telemetry
import insights
import framebuf
import router as request_router
import visual_qa
# Import order matters: perceive lends vision's resident Qwen3-VL to
# headway.anchor at import time, so headway.live's anchor path finds a provider
# already installed and never pulls a second copy of the weights.
from headway import live as headway_live
from headway import live_policy
from headway import lanes as headway_lanes
from headway import detect as headway_detect

# Set once the warm thread has finished, successfully or not.
#
# uvicorn opens the port before the models are loaded -- deliberately, so the
# pod reports ready immediately -- and a dashboard left open from a previous
# run starts POSTing /headway_frame at 4 fps the instant it can. Those requests
# lazily load RF-DETR, UFLDv2 and Depth Anything on the threadpool while Qwen
# is still being dispatched onto the same card.
#
# WHAT THIS IS NOT. It is not the fix for "Cannot copy out of meta tensor"
# during the Qwen load, which is what it was written to chase. That failure
# turned out to be TWO uvicorn processes started by mistake, both loading Qwen
# onto the same GPU at once; with a single server it does not happen, with or
# without traffic arriving during startup. (Worth knowing if it ever comes
# back: check `nvidia-smi --query-compute-apps` for a second server before
# suspecting anything subtler. `pkill -f "uvicorn app:app"` is a good way to
# create one, because the pattern also matches the shell running the pkill.)
#
# It is kept because refusing a frame while the models are still loading is the
# right answer to that request regardless. Processing one early means paying a
# ~3 s lazy model load inside a request that has a 250 ms budget, and getting a
# frame with no detector behind it -- which reports UNKNOWN anyway. The client
# simply sends the next frame, and /headway_frame already has a "skipped"
# contract for exactly this shape of thing.
_warm_done = threading.Event()


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
        # RF-DETR is now the headway candidate source and runs on EVERY frame,
        # so an unwarmed first call would put a ~3 s weight load inside a live
        # frame. Without it headway has no candidates at all and reports
        # UNKNOWN -- it does not fall back to Qwen, which is the point.
        try:
            headway_detect.warm()
            print("[vision] detector warm", flush=True)
        except Exception as e:
            print(f"[vision] detector unavailable: {e}", flush=True)
        # flush: stdout is block-buffered when uvicorn is redirected to a log
        # file, so without this the line can sit unseen for a long time.
        print(f"[vision] warm complete in {time.time() - t:.1f}s", flush=True)
    except Exception as e:
        # Warming is an optimisation; a failure here must not stop the server.
        # The first /observe will simply load the model the old way.
        #
        # The traceback goes to the log too. A one-line message is enough to
        # know the pod came up without vision and useless for knowing why:
        # this failure has been seen as "Cannot copy out of meta tensor", which
        # names a symptom several layers below the cause and is not something
        # anyone reconstructs from the message alone.
        import traceback

        print(f"[vision] warm failed after {time.time() - t:.1f}s: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        # ALWAYS, including on failure. A warm that died must not leave the
        # frame endpoints refusing work for the rest of the process's life --
        # degraded vision is recoverable, a server that never accepts a frame
        # is not.
        _warm_done.set()


# How often the reaper looks. Much shorter than SESSION_ABANDON_S, because the
# cost of looking is a dictionary scan and the cost of not looking is a drive's
# worth of state held for as long as the process lives.
REAP_INTERVAL_S = 20.0

_stop_reaper = threading.Event()


def _reap_abandoned_sessions():
    """End drives whose client has stopped reporting in.

    The other half of the heartbeat. /session/end covers the tidy exit; this
    covers the phone that went flat, the laptop that slept, and the tab closed
    while the page was hidden — none of which get to send anything.

    Runs on its own thread rather than piggybacking on request handling,
    precisely because the case it exists for is "no more requests are coming".
    """
    while not _stop_reaper.wait(REAP_INTERVAL_S):
        try:
            for sid in sessions.abandoned():
                idle = sessions.idle_for(sid)
                print(f"[sessions] reaping {sid[:8]} — no contact for "
                      f"{idle:.0f}s", flush=True)
                _teardown_session(sid, reason="abandoned")
        except Exception as e:
            # A reaper that dies takes the cleanup with it for the life of the
            # process, so it survives its own mistakes.
            print(f"[sessions] reaper error: {type(e).__name__}: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm on a daemon thread so uvicorn reports ready immediately. vision._lock
    # makes an early /observe wait for this load instead of starting a second one.
    threading.Thread(target=_warm_vision, name="vision-warm", daemon=True).start()
    threading.Thread(target=_reap_abandoned_sessions, name="session-reaper",
                     daemon=True).start()
    yield
    _stop_reaper.set()


app = FastAPI(lifespan=lifespan)
client = OpenAI()

# The pre-rendered red-tier clips live in static/audio/ and must be reachable
# for the browser to preload them. "/" keeps serving index.html by hand so the
# mount does not shadow it.
app.mount("/static", StaticFiles(directory="static"), name="static")


INDEX_PATH = Path("static/index.html")
MAPS_KEY_TOKEN = "__GOOGLE_MAPS_API_KEY__"


@app.get("/")
def index():
    """index.html with the Maps browser key injected at serve time.

    The Maps JavaScript API needs a key the browser can see, so "keep it out of
    the browser" is not available — but "keep it out of the repository" is, and
    that is the line that matters: the key lives in .env (gitignored) and enters
    the page here, on the way out. Nothing in static/ ever contains it.

    Consequence worth knowing: the page must be loaded from "/" and not from
    /static/index.html, which the StaticFiles mount will happily serve with the
    placeholder still in it — the map would be the only thing that fails, and it
    fails visibly.
    """
    html = INDEX_PATH.read_text()
    return HTMLResponse(html.replace(MAPS_KEY_TOKEN,
                                     os.getenv("GOOGLE_MAPS_API_KEY", "")))

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


def _visual_key(session_id):
    """Ring/visual-session key. One driver, so a keyless turn still has state."""
    return session_id or "default"


def _route_and_prepare(transcript: str, session_id: str):
    """Classify the utterance and, if it is visual, build the turn. -> (route, va).

    Blocking on purpose and always called through a threadpool: the router may
    ask a model, and preparing a visual turn decodes frames and can run a Qwen
    pass. Neither belongs on the event loop.

    Never fatal. A router or preparation failure returns no visual turn, and the
    ordinary conversation path answers instead — degraded, not broken.
    """
    key = _visual_key(session_id)
    try:
        if not config.VISUAL_QA_ENABLED:
            return None, None
        sess = visual_qa.get_session(key)
        # Both pieces of conversational state the router needs. Missing the
        # second one is invisible until it matters: RIO asks "the black one or
        # the white one?", the driver says "the black one", and without knowing
        # a question is outstanding the router reads that as an ordinary
        # utterance and the answer never lands.
        route = request_router.classify(
            transcript,
            has_referent=sess.active_referent() is not None,
            pending_clarification=sess.pending_clarification() is not None)
        if not request_router.is_visual(route["request_type"]):
            return route, None
        return route, visual_qa.answer(key, transcript, route)
    except Exception as e:
        print(f"[talk] visual routing failed: {type(e).__name__}: {e}", flush=True)
        return None, None


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

    # A question about what is out of the window goes down the visual path, and
    # everything else goes where it always did. The decision is made here, once,
    # on the server, so the voice path and the text path cannot diverge on it.
    route, va = await run_in_threadpool(_route_and_prepare, transcript, session_id)
    request_type = (route or {}).get("request_type", "non_visual_question")

    print({
        "transcript": transcript,
        "whisper_seconds": round(t1 - t0, 3),
        "request_type": request_type,
        "visual": va is not None,
    })

    talk_id = uuid.uuid4().hex[:12]

    def streamer():
        buffer = ""
        reply_parts = []
        audio_len = 0

        tokens = va.stream() if va is not None else llm_interface.generate_stream(transcript)

        for token in tokens:
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
            "request_type": request_type,
            "t": time.time(),
        })
        sessions.log_talk(session_id, transcript, reply, audio_len, latency_ms)
        if va is not None:
            # Every stage of the visual turn, with its own latencies. The STT
            # cost is added here because it is the one stage that happens
            # before the turn object exists.
            va.meta.setdefault("timing_ms", {})["stt"] = round((t1 - t0) * 1000, 1)
            va.meta["talk_id"] = talk_id
            sessions.log_visual_qa(session_id, va.meta)

    # The transcript is known before streaming starts, so it can ride on a header.
    # URL-encoded because headers are latin-1 only and transcripts are UTF-8.
    headers = {
        "X-Transcript": quote(transcript, safe=""),
        "X-Talk-Id": talk_id,
        "X-Request-Type": request_type,
        "Access-Control-Expose-Headers": "X-Transcript, X-Talk-Id, X-Request-Type",
    }
    return StreamingResponse(streamer(), media_type="audio/mpeg", headers=headers)


# --- Visual conversation (docs/visual_qa.md) --------------------------------


@app.get("/scene")
def scene_endpoint(session_id: str = Query(default=None)):
    """The live scene graph: what RIO can currently see, with stable track ids.

    Read-only view of state the 4 fps headway loop already produces. It runs no
    model and changes nothing — the dashboard draws it, the acceptance tests
    assert on it, and a driver never hears it.
    """
    return visual_qa.scene_graph(_visual_key(session_id))


@app.post("/ask")
async def ask_endpoint(body: dict = Body(...), session_id: str = Query(default=None)):
    """A visual question in text, answered in text. No microphone, no TTS.

    This is the same pipeline /talk uses, minus Whisper at the front and
    ElevenLabs at the back, which is what makes the acceptance tests runnable
    without a car. `meta` carries every stage's decision and latency.
    """
    question = str(body.get("question") or "").strip()
    if not question:
        return {"error": "no question"}

    route, va = await run_in_threadpool(_route_and_prepare, question, session_id)
    if va is None:
        # Not a visual question (or the visual path is unavailable): answer it
        # the ordinary way rather than refusing, so /ask is a complete
        # conversational endpoint and not a visual-only one.
        reply = await run_in_threadpool(
            lambda: "".join(llm_interface.generate_stream(question)).strip())
        return {"reply": reply, "request_type": (route or {}).get(
            "request_type", "non_visual_question"), "visual": False,
            "route": route}

    reply = await run_in_threadpool(va.text)
    sessions.log_visual_qa(session_id, va.meta)
    return {"reply": reply, "request_type": route["request_type"],
            "visual": True, "meta": va.meta}


@app.get("/visual_state")
def visual_state_endpoint(session_id: str = Query(default=None)):
    """Frame buffer and referent state. Diagnostics for the dashboard."""
    key = _visual_key(session_id)
    ring = framebuf.peek_ring(key)
    sess = visual_qa.get_session(key)
    ref = sess.active_referent()
    return {
        "session_id": key,
        "buffer": ring.stats() if ring else {"frames": 0},
        "retention": {"seconds": config.RING_SECONDS,
                      "max_frames": config.RING_MAX_FRAMES,
                      "persist_images": config.RING_PERSIST},
        "active_referent": ref.to_log() if ref else None,
        "turns": len(sess.turns),
        "enriched_tracks": sorted(sess.enrichment.all().keys()),
    }

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
    # Same gate as /headway_frame, same reason: this path runs Qwen and Depth
    # Anything, and doing that while Qwen is mid-load is what breaks the load.
    if not _warm_done.is_set():
        return {"boxes": [], "corridor": [], "caption": "", "observation": "",
                "skipped": "warming", "timing_ms": {"total": 0.0}}

    # ...and the same staleness rule. This one costs a full Qwen generate, so
    # it is the more expensive of the two to serve for nobody.
    if session_id and not sessions.touch(session_id):
        return {"boxes": [], "corridor": [], "caption": "", "observation": "",
                "stale": True, "reason": "unknown_session",
                "timing_ms": {"total": 0.0}}

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
    # Models still loading: refuse the frame rather than pull three more models
    # onto the card while Qwen is being dispatched onto it. See _warm_done.
    if not _warm_done.is_set():
        return {"ok": False, "skipped": "warming"}

    # A frame tagged with a session this process has never heard of is a tab
    # left open across a restart. Refusing costs nothing and stops the whole
    # cascade: no LiveSession minted on demand, no frame ring filled, no GPU
    # spent, no events spilled into stray-<id>.jsonl for a drive nobody is on.
    #
    # `stale` is the client's cue to stop. A frame with NO session id is a
    # different thing entirely and still welcome — that is the desk-testing
    # path, where a clip is run through the loop with no drive in progress.
    if session_id and not sessions.touch(session_id):
        return {"ok": False, "stale": True, "reason": "unknown_session"}

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

    # Retain the frame for the conversation path. AFTER processing, so nothing
    # here is inside the 250 ms frame budget, and outside the session lock,
    # which the frame no longer needs. The ring is RAM-only and six seconds
    # long — see framebuf.py.
    if config.VISUAL_QA_ENABLED:
        try:
            framebuf.get_ring(_visual_key(session_id)).push(image_bytes, result)
        except Exception as e:
            # Losing a frame from the buffer costs a better answer later. It
            # must never cost the headway frame that has already been computed.
            print(f"[framebuf] push failed: {type(e).__name__}: {e}", flush=True)

    sessions.log_headway(session_id, result, (time.time() - t0) * 1000)
    # Lane departure is logged, not spoken. It gets its own event kind so a
    # review pass can pull the handful of excursions out of a whole drive; see
    # sessions.log_lane_drift and the scope note in headway/lanes.py.
    if (result.get("lane_drift") or {}).get("drift"):
        sessions.log_lane_drift(session_id, result)
    # Merge promotions DO change behaviour (a promoted candidate can take the
    # lead lock), so unlike drift these are worth pulling out of the stream to
    # review against what the driver actually experienced.
    if result.get("merge_promotions"):
        sessions.log_merge_promotion(session_id, result)
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


# --- Navigation (WebNavProvider, step 1) ---
#
# Three endpoints and no more: ask Google for a route, look an announcement up,
# record what the progression engine did. The engine itself is client-side (see
# static/rio_navcore.js) — nothing here is in the timing path between a driver
# and a turn.


@app.get("/nav/suggest")
def nav_suggest_endpoint(q: str = Query(...), lat: float = Query(default=None),
                         lng: float = Query(default=None)):
    """Places autocomplete for the destination box. Never fatal — an empty list
    just means the driver types the address in full and it geocodes."""
    return {"suggestions": nav.suggest(q, lat, lng)}


@app.get("/nav/geocode")
def nav_geocode_endpoint(q: str = Query(...)):
    """Address -> lat/lng. Used for the desk-testing start-point override, and
    as the fallback when a destination is typed rather than picked."""
    g = nav.geocode(q)
    return g or {"error": "could not find that place", "q": q}


@app.post("/nav/route")
def nav_route_endpoint(body: dict = Body(...), session_id: str = Query(default=None)):
    """Compute a route and make it the active one.

    Google decides the route. The only judgement in this handler is refusing to
    pretend: a routing failure comes back as an error the panel shows, never as
    a stale or invented route.
    """
    try:
        lat = float(body.get("lat"))
        lng = float(body.get("lng"))
    except (TypeError, ValueError):
        return {"error": "need a current position (lat, lng) to route from"}
    try:
        route = nav.compute_route(
            lat, lng,
            destination=str(body.get("destination") or ""),
            place_id=str(body.get("place_id") or ""),
            label=str(body.get("label") or ""),
            reroute_of=body.get("reroute_of"),
        )
    except nav.NavError as e:
        sessions.log_nav(session_id, "route_failed", {
            "destination": body.get("destination"), "error": str(e)})
        return {"error": str(e)}
    # route_set is logged here rather than from the browser: the server is the
    # only place that holds the whole route, and the summary it writes is the
    # record a review reads to see what RIO *intended* to say on this drive.
    sessions.log_nav(session_id, "route_set", nav.summary(route))
    return route


@app.get("/nav/voice")
def nav_voice_endpoint(route_id: str = Query(...), m: int = Query(...),
                       tier: str = Query(...), dist_m: float = Query(default=None)):
    """TTS for one precomputed announcement, addressed by (route, maneuver, tier).

    Deliberately not a text-to-speech endpoint, for the same reason
    /headway_voice is not: the browser sends coordinates into a table, never a
    sentence. Anything that does not resolve to a maneuver on a live route is
    refused, so the set of things RIO's voice can ever say about navigation is
    bounded by what Google returned and this process precomputed.

    The exact sentence rides back on X-Nav-Text so the panel displays the words
    that are being spoken rather than its own reconstruction of them.
    """
    text = nav.announcement_text(route_id, m, tier, dist_m)
    if not text:
        return {"error": "unknown route, maneuver or tier",
                "route_id": route_id, "m": m, "tier": tier}
    return StreamingResponse(
        voice.synthesize_stream(text), media_type="audio/mpeg",
        headers={
            "X-Nav-Text": quote(text, safe=""),
            "Access-Control-Expose-Headers": "X-Nav-Text",
            "Cache-Control": "no-store",
        })


@app.post("/nav/event")
def nav_event_endpoint(body: dict = Body(...), session_id: str = Query(default=None)):
    """Record one progression event (kind "nav") in the session JSONL."""
    event = str(body.get("event") or "unknown")
    payload = body.get("payload")
    sessions.log_nav(session_id, event, payload if isinstance(payload, dict) else None)
    return {"logged": event}


# ---------------------------------------------------------------------------
# Vehicle health — tires
# ---------------------------------------------------------------------------
# Thin on purpose. Every threshold, every state name and every string the panel
# prints is decided in tires.py against config.py; these three handlers exist
# only to put that on the wire. When the mock provider is replaced by real TPMS
# hardware, nothing here changes.

@app.get("/vehicle/tires")
def vehicle_tires_endpoint():
    """Current tire state, already normalised and already worded.

    The dashboard polls this at the cadence the payload itself carries
    (`poll_ms`), and renders it verbatim — it does no arithmetic and holds no
    thresholds of its own.
    """
    return tires.snapshot()


@app.get("/vehicle/tires/scenario")
def vehicle_tires_scenario_endpoint():
    """Which mock scenario is live, and what else is on offer.

    Empty `scenarios` is the honest answer from a real provider: hardware has
    exactly one scenario, which is whatever the tires are actually doing.
    """
    return {"scenario": tires.current_scenario(),
            "scenarios": tires.scenarios(),
            "provider": tires.provider().name}


@app.post("/vehicle/tires/scenario")
def vehicle_tires_scenario_set_endpoint(name: str = Query(...)):
    """Switch the mock provider's scenario. Development only.

    Returns a full snapshot rather than an acknowledgement so the panel repaints
    on the same round trip instead of showing the old state until the next poll.
    An unknown name changes nothing and says so.

    Routed through telemetry.set_tire_scenario rather than tires.set_scenario
    because the tire channels are rows in the telemetry list now, and the trend
    window has to forget the pressures from the scenario being left behind.
    """
    if not telemetry.set_tire_scenario(name):
        return {"error": "unknown scenario", "name": name,
                "scenarios": tires.scenarios()}
    return tires.snapshot()


# ---------------------------------------------------------------------------
# Vehicle health — telemetry and insights
# ---------------------------------------------------------------------------
# Thin for the same reason the tire handlers above are thin. Every band, every
# status word, every trend arrow and every sentence in the insight log is
# decided in telemetry.py and insights.py against config.py; these handlers put
# that on the wire and nothing else.
#
# Neither of these paths can speak. There is no arbiter call in telemetry.py, in
# insights.py, or in static/rio_vehicle.js — a predictive observation is a line
# in a log, not an alert, and this whole column is display-only in this phase.

@app.get("/vehicle/telemetry")
def vehicle_telemetry_endpoint():
    """Every sensor on the car, already normalised and already worded.

    The dashboard polls this at the cadence the payload itself carries
    (`poll_ms`) and renders it verbatim — it does no arithmetic and holds no
    thresholds of its own. Folding the frame into the insight engine happens
    here as a side effect of the read, so the baselines see every sample rather
    than only the ones somebody was looking at the log for.
    """
    return telemetry.snapshot()


@app.get("/vehicle/insights")
def vehicle_insights_endpoint():
    """The VEHICLE INSIGHTS log, newest first.

    Slower cadence than telemetry by an order of magnitude: these entries are
    measured in hours and days, and an event log that repaints every second is
    a log nobody can read a line of.
    """
    return insights.snapshot()


@app.get("/vehicle/telemetry/scenario")
def vehicle_telemetry_scenario_endpoint():
    """Which mock scenario is live, and what else is on offer."""
    return {"scenario": telemetry.current_scenario(),
            "scenarios": telemetry.scenarios()}


@app.post("/vehicle/telemetry/scenario")
def vehicle_telemetry_scenario_set_endpoint(name: str = Query(...)):
    """Switch the mock ECU's scenario. Development only.

    Returns a full snapshot rather than an acknowledgement so the panel repaints
    on the same round trip instead of showing the old state until the next poll.
    An unknown name changes nothing and says so.
    """
    if not telemetry.set_scenario(name):
        return {"error": "unknown scenario", "name": name,
                "scenarios": telemetry.scenarios()}
    return telemetry.snapshot()


# --- Phase 2.5 session endpoints ---

@app.post("/session/start")
def session_start_endpoint(metadata: dict = Body(default=None, embed=True)):
    sid = sessions.start_session(metadata=metadata)
    return {"session_id": sid}


def _teardown_session(session_id: str, reason: str = "closed") -> bool:
    """End a drive and drop everything it was holding.

    One function, because there are now two ways in — the driver tapping End
    Drive, and the reaper noticing the client has gone — and a drive that ends
    the second way must release exactly as much as one that ends the first way.
    Anything freed on only one path is a leak that only shows up on the
    failure case, which is the worst place to find one.
    """
    closed = sessions.end_session(session_id, reason=reason)
    # Drop the live headway state with the session. Without this a tracker,
    # Kalman filter and cooldown table survive for every drive the process has
    # ever seen, and a re-used session id would resume mid-warning.
    headway_live.reset_session(session_id)
    # ...and the visual state with it. The frame ring is the one that matters:
    # it holds pictures of the road, and "the drive ended" is exactly when they
    # should stop existing. The referent goes too — a car discussed on the last
    # drive must not be what "what year is it?" attaches to on the next one.
    key = _visual_key(session_id)
    framebuf.drop_ring(key)
    visual_qa.drop_session(key)
    return closed


@app.post("/session/end")
def session_end_endpoint(session_id: str = Query(...)):
    return {"closed": _teardown_session(session_id)}


@app.post("/session/heartbeat")
def session_heartbeat_endpoint(session_id: str = Query(...)):
    """"I am still driving." Sent by the dashboard every few seconds.

    It answers a question the client cannot answer for itself: does this server
    still know about my drive? After a restart the answer is no, and the tab
    that has been cheerfully POSTing frames at 4 fps into a process that has
    never heard of it needs to be told so — see sessions.touch().

    `ok: false` is an instruction, not an error. The client stops the drive.
    """
    if sessions.touch(session_id):
        return {"ok": True, "session_id": session_id,
                "idle_s": round(sessions.idle_for(session_id) or 0.0, 2)}
    return {"ok": False, "unknown_session": True, "session_id": session_id}


@app.get("/sessions")
def sessions_endpoint():
    """Which drives this process thinks are live, and how long since each spoke."""
    return {"sessions": sessions.active_sessions(),
            "abandon_after_s": sessions.SESSION_ABANDON_S}


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
