

import json
import math
import re
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

from fastapi import FastAPI, UploadFile, Query, Body, Form, Header, Request
from fastapi.responses import (StreamingResponse, HTMLResponse, FileResponse,
                               JSONResponse)
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

import config
import voice
import llm_interface
import vision
import perceive
import realtime
from navigation import events as navevents
from navigation import service as navservice
from navigation import speech as navspeech
from navigation import verify as navverify
import tires
import telemetry
import insights
import vehicle_health
import vehicle_health_policy
from tire_diag import engine as tire_diag
from tire_diag import codes as tire_codes
from tire_diag import monitors as tire_monitors
from powertrain_diag import engine as powertrain_diag
from powertrain_diag import codes as powertrain_codes
from powertrain_diag import monitors as powertrain_monitors
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
from headway import depth as headway_depth

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
        # Depth Anything V2 Metric-Small. Warmed on the same thread, after
        # Qwen, so a live frame does not pay the load while a drive is already
        # running. It used to be warmed through perceive.warm(), back when
        # /perceive measured a distance inside every box Qwen drew; that path
        # is gone and depth belongs to the headway loop, which is the only
        # thing that ranges anything now.
        try:
            headway_depth.warm()
            print("[vision] depth warm", flush=True)
        except Exception as e:
            print(f"[vision] depth unavailable: {e}", flush=True)
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
# The replay presentation buffer, injected the same way and for the same reason
# the Maps key is: the value has to exist in one place. config.py is that place,
# the browser is where it is acted on, and a constant copied into the JavaScript
# would drift from the harness that asserts against it.
REPLAY_LEAD_TOKEN = "__HEADWAY_REPLAY_LEAD_S__"

# Every local asset the page pulls in, so its URL can be stamped with the
# file's own mtime on the way out.
_ASSET_REF = re.compile(r'(src|href)="(/static/[^"?#]+)"')


def _stamp_assets(html: str) -> str:
    """/static/rio_nav.js -> /static/rio_nav.js?v=<mtime>

    A browser holding yesterday's JavaScript against today's endpoints is not a
    stale cache, it is two different programs talking to each other, and it
    fails in ways that look like the server is broken: this repo has already
    spent an afternoon on a dropdown that rendered blank rows because the panel
    and the endpoint disagreed about field names across exactly that gap.

    Stamping the URL with the file's modification time means a changed file is
    a changed URL and there is nothing for the browser to serve from cache. An
    unchanged file keeps its URL and stays cached, which is the whole point of
    doing this rather than disabling caching.

    A file that cannot be stat'd is left alone: a missing asset should 404
    visibly, not be hidden behind a rewrite that failed quietly.
    """
    def stamp(m):
        attr, path = m.group(1), m.group(2)
        try:
            mtime = int(Path(path.lstrip("/")).stat().st_mtime)
        except OSError:
            return m.group(0)
        return f'{attr}="{path}?v={mtime}"'
    return _ASSET_REF.sub(stamp, html)


@app.get("/")
def index():
    """index.html with the Maps browser key injected and the assets stamped.

    The Maps JavaScript API needs a key the browser can see, so "keep it out of
    the browser" is not available — but "keep it out of the repository" is, and
    that is the line that matters: the key lives in .env (gitignored) and enters
    the page here, on the way out. Nothing in static/ ever contains it.

    Consequence worth knowing: the page must be loaded from "/" and not from
    /static/index.html, which the StaticFiles mount will happily serve with the
    placeholder still in it AND with unstamped script tags — the map would fail
    visibly, and the JavaScript would go stale invisibly, which is worse.

    The page itself is served no-cache so the stamps are always read fresh; it
    is a few tens of kilobytes and it is the one document that decides which
    version of everything else the browser runs.
    """
    html = _stamp_assets(INDEX_PATH.read_text())
    html = html.replace(MAPS_KEY_TOKEN, os.getenv("GOOGLE_MAPS_API_KEY", ""))
    html = html.replace(REPLAY_LEAD_TOKEN, f"{config.HEADWAY_REPLAY_LEAD_S:g}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

@app.get("/health")
def health():
    """Is the server up, and is it whole?

    `status: ok` used to mean only "the process is answering", which is a
    weaker claim than it reads as. Several capabilities are deliberately
    non-fatal when missing — a pod that comes up without a detector is better
    than a pod that does not come up — and the cost of that choice is that a
    degraded pod looks identical to a healthy one from here.

    It is not hypothetical. On 2026-08-02 this pod ran for a day with no
    RF-DETR: boot.sh had aborted before installing it, headway had no candidate
    source and reported UNKNOWN for every frame, and /health said "ok"
    throughout. `degraded` is the field that would have said so.

    Cheap: every check below is a `in sys.modules` test or an attribute lookup
    on something already imported. Nothing is loaded to answer this.
    """
    degraded = []

    if not headway_detect.available():
        degraded.append({
            "component": "detector",
            "detail": "RF-DETR is not loaded — headway has no candidates and "
                      "the visual scene graph is empty",
            "fix": "python -m tools.preflight --fix"})
    if not headway_lanes.available():
        degraded.append({
            "component": "lanes",
            "detail": "UFLDv2 is not loaded — headway uses the static trapezoid "
                      "corridor",
            "fix": "python -m tools.fetch_lane_weights"})
    if not _warm_done.is_set():
        degraded.append({
            "component": "warm",
            "detail": "models are still loading — frames are refused until this "
                      "clears (normally ~45s from start)",
            "fix": ""})

    return {"status": "ok", "service": "rio-phase1",
            # "ok" the process is up; "degraded" it is up and missing something
            # it is supposed to have. Never "error" — if this handler runs at
            # all, the server is answering.
            "readiness": "ok" if not degraded else "degraded",
            "degraded": degraded}


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
        if request_router.is_vehicle_health(route["request_type"]):
            # The driver asked how the car is. One of the spec's cooldown
            # resets, and it belongs here rather than in the health layer: this
            # is the only place that knows a question was asked, and the policy
            # is deliberately incapable of finding out on its own.
            _health_policy.note_status_request(time.time())
        if request_router.is_navigation(route["request_type"]):
            # The driver named a destination. Resolved HERE, on the server,
            # through the same provider the panel uses, so a spoken request and
            # a typed one cannot land on different places. Nothing is routed
            # yet: the tracker is client-side and the browser decides when to
            # attach a route, exactly as it does when the destination is typed.
            route["navigation"] = _resolve_spoken_destination(transcript, session_id)
        if request_router.is_diagnostic_report(route["request_type"]):
            # ...and this one asked RIO to go and interrogate the car, which is
            # a different thing and is why it is a different intent. Run here,
            # on the threadpool this function is already on, so the answer is
            # about a scan that just happened rather than one from four minutes
            # ago.
            #
            # Never fatal: a report that fails leaves RIO answering from the
            # ordinary health context, which is degraded rather than broken.
            try:
                route["diagnostic_report"] = _run_report(session_id).get("report")
            except Exception as e:
                print(f"[talk] diagnostic report failed: "
                      f"{type(e).__name__}: {e}", flush=True)
        if not request_router.is_visual(route["request_type"]):
            return route, None
        return route, visual_qa.answer(key, transcript, route)
    except Exception as e:
        print(f"[talk] visual routing failed: {type(e).__name__}: {e}", flush=True)
        return None, None


def _resolve_spoken_destination(transcript: str, session_id: str = None) -> dict:
    """"Take me to LAX" -> a destination, or a question about which one.

    Returns what the browser needs to act on plus the exact line RIO says. The
    line is a template (navigation/speech.py): a model composing "Routing to
    LAX" is a model that can compose "Routing to LAS", and the destination is
    the one part of this sentence that has to be the provider's own word.
    """
    try:
        res = navservice.resolve_destination(transcript)
    except Exception as e:
        print(f"[talk] destination resolution failed: {type(e).__name__}: {e}",
              flush=True)
        return {"status": "not_found", "query": transcript,
                "spoken": navspeech.destination_reply("not_found")}
    status = res["status"]
    query = res.get("query", "")
    if status == "resolved":
        dest = res["destination"].to_dict()
        return {"status": status, "query": query, "destination": dest,
                "spoken": navspeech.destination_reply(status,
                                                      name=dest["display_name"])}
    if status == "ambiguous":
        sessions.log_nav(session_id, navevents.DESTINATION_AMBIGUOUS,
                         {"query": query, "spoken": True,
                          "candidates": [c["display_name"] for c in res["candidates"]]})
        return {"status": status, "query": query, "candidates": res["candidates"],
                "spoken": navspeech.destination_reply(status,
                                                      candidates=res["candidates"])}
    return {"status": "not_found", "query": query,
            "spoken": navspeech.destination_reply("not_found", query=query)}


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

        nav_action = (route or {}).get("navigation")
        if nav_action:
            # A destination request is answered from a template, not a model.
            # One "token", so the sentence reaches TTS in one piece.
            tokens = iter([nav_action["spoken"]])
        elif va is not None:
            tokens = va.stream()
        else:
            tokens = llm_interface.generate_stream(transcript, route)

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
    # A resolved (or ambiguous) destination rides back on a header so the panel
    # can set the route — or offer the choice — while RIO is still saying so.
    # The browser gets a place id and a label, never a sentence to speak.
    nav_action = (route or {}).get("navigation")
    if nav_action:
        headers["X-Nav-Action"] = quote(json.dumps(nav_action), safe="")
        headers["Access-Control-Expose-Headers"] += ", X-Nav-Action"
    return StreamingResponse(streamer(), media_type="audio/mpeg", headers=headers)


# --- RIO live: speech-to-speech conversation (docs/realtime_conversation.md) -
#
# Two endpoints, and neither is in an audio path. The browser holds the audio
# connection to the model directly — that is what makes it a conversation
# rather than a series of round trips through here — so the server's whole job
# is to mint a short-lived credential for it and to answer the one tool the
# session can call.
#
# Nothing about safety, vehicle health or navigation speech passes through
# either. Those are generated by policy code from fixed tables and always will
# be; the live model is conversation, and conversation is the tier that yields.


@app.post("/realtime/session")
def realtime_session_endpoint(session_id: str = Query(default=None)):
    """An ephemeral credential for one live conversation.

    The browser cannot be given the account key, so it is given a secret that
    is scoped to a single session and expires in a minute — the same discipline
    every other key on this page follows.

    A failure here is not fatal to the drive: the panel falls back to
    hold-to-talk through Whisper, which is the path that was there before.
    """
    if not config.REALTIME_ENABLED:
        return {"error": "live conversation is switched off", "enabled": False,
                "status": realtime.status()}
    try:
        out = realtime.mint_client_secret()
    except Exception as e:
        print(f"[realtime] session mint failed: {type(e).__name__}: {e}", flush=True)
        sessions.log_live(session_id, "session_failed",
                          {"error": f"{type(e).__name__}"})
        return {"error": f"could not start a live session: {type(e).__name__}",
                "enabled": True, "status": realtime.status()}
    sessions.log_live(session_id, "session_started",
                      {"model": out["model"], "voice": out["voice"]})
    # Start describing the road NOW, not when she is first asked about it.
    # A live conversation is exactly the context where "what do you see" is
    # asked, and the observer needs a second's head start to have an answer
    # ready — see observer.py, and the measurement in tools/visual_latency.py.
    try:
        import observer

        observer.start(_visual_key(session_id))
    except Exception as e:
        # A missing observation costs speed, never the conversation.
        print(f"[realtime] observer not started: {type(e).__name__}: {e}",
              flush=True)
    out["enabled"] = True
    return out


@app.get("/realtime/status")
def realtime_status_endpoint():
    """What the live path is doing, read-only.

    Added while diagnosing a slow "what do you see": the fast path either has a
    fresh observation or it does not, and until this existed there was no way
    to ask which — the answer was inferred from latency, which is how a
    contended GPU and a stopped observer look identical from outside.
    """
    return realtime.status()


@app.post("/realtime/tool")
def realtime_tool_endpoint(body: dict = Body(...), session_id: str = Query(default=None)):
    """The one tool the live session can call: think harder, or look it up.

    Runs here rather than in the browser for the obvious reason — the key — and
    for a less obvious one: this is the only place that can write what RIO
    reached for into the drive's log. A session that goes quiet for six seconds
    is a question about escalation, and the audio stream cannot answer it.

    Every failure comes back as `ok: false` with a short note. The session's
    instructions tell RIO what to do with that, which is to carry on.
    """
    name = str(body.get("name") or "")
    args = body.get("arguments")
    # The car's last GPS fix, attached by the panel to every tool call. The
    # browser is the only thing that knows where the car is -- the same reason
    # nav_status is answered there -- and find_places needs it to make "near me"
    # mean anything. Absent or stale, the tool asks the driver for an area
    # rather than searching somewhere plausible.
    where = body.get("where")
    # The visual tool needs the session's own frame ring — the same key the
    # hold-to-talk path uses, so a live question and a recorded one look at the
    # same few seconds of road.
    result = realtime.run_tool(name, args, session_key=_visual_key(session_id),
                               where=where)
    logged = {"tool": name, "ok": bool(result.get("ok")),
              "took_ms": result.get("took_ms")}
    if isinstance(args, dict):
        logged["question"] = str(args.get("question") or "")[:400]
        # A place search is worth reviewing afterwards: what was asked, where it
        # was searched, and what came back that RIO was then speaking from.
        if args.get("query"):
            logged["query"] = str(args["query"])[:200]
            logged["near"] = str(args.get("near") or "")[:120]
            logged["open_now"] = bool(args.get("open_now"))
    if result.get("results") is not None:
        logged["n_results"] = len(result.get("results") or [])
        logged["names"] = [r.get("name") for r in (result.get("results") or [])][:5]
    if not result.get("ok"):
        logged["note"] = result.get("note")
    sessions.log_live(session_id, "tool_call", logged)
    # A visual turn records its whole decision chain in the same kind the
    # hold-to-talk path uses: which frame, which track, what went to the model.
    # Without this a live drive's visual answers would be unarguable.
    meta = result.pop("meta", None)
    if meta:
        sessions.log_visual_qa(session_id, meta)
    return result


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
    nav_action = (route or {}).get("navigation")
    if nav_action:
        # Same answer /talk speaks, in text: the destination path is fully
        # testable without a microphone or a speaker.
        return {"reply": nav_action["spoken"], "request_type": route["request_type"],
                "visual": False, "navigation": nav_action}
    if va is None:
        # Not a visual question (or the visual path is unavailable): answer it
        # the ordinary way rather than refusing, so /ask is a complete
        # conversational endpoint and not a visual-only one.
        reply = await run_in_threadpool(
            lambda: "".join(llm_interface.generate_stream(question, route)).strip())
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
    """One frame -> one sentence. RIO's cheapest look at the road.

    In a threadpool for the same reason /perceive and /headway_frame are: the
    work is a blocking Qwen generate, and run directly in this coroutine it
    holds the event loop for its whole duration while every other request --
    the 4 fps frame stream included -- waits behind it.

    Measured on this pod, four runs each, a headway loop pushing frames for the
    length of one observe: on the loop, exactly ONE frame completed per observe
    and it took 208-366 ms, which is that frame sitting out the generate. Off
    it, three to six frames complete inside the same window at ~95 ms each.
    Observe itself gets slower (360-410 ms to 560-770 ms) because it now shares
    the card instead of owning it, which is the trade and the right way round:
    a caption may wait, a following-distance warning may not.

    The numbers here are smaller than /perceive's because OBSERVER_MAX_SIDE_PX
    keeps this generate short. That is a property of the current prompt and not
    a reason to leave it on the loop.
    """
    import time as _t
    _t0 = _t.time()
    image_bytes = await image.read()
    observation = await run_in_threadpool(vision.observe, image_bytes)
    sessions.log_observe(session_id, len(image_bytes), observation, (_t.time() - _t0) * 1000)
    return {"observation": observation}


@app.post("/perceive")
async def perceive_endpoint(image: UploadFile = File(...), session_id: str = Query(default=None),
                            debug: int = Query(default=0)):
    """The caption and the deterministic geometry under it. /observe unchanged.

    Returns the same caption /observe would, plus the ego corridor and the
    detected lane lines, so the Camera panel can draw the geometry RIO is
    reasoning over. It does NOT return boxes to draw -- see perceive.py.

    In a threadpool, like /headway_frame and for a sharper version of the same
    reason. The work is a blocking Qwen generate of one to four seconds, and it
    used to run directly in this coroutine, which meant it blocked the event
    loop and therefore every other request the server had -- including the
    4 fps /headway_frame stream. That was survivable while a caption and a
    headway run never overlapped on a clip. They do now: playing a clip starts
    detection, and the caption keeps ticking underneath it, so a blocking
    generate here would stall the detection loop for seconds at a time and put
    the stutter squarely on the feature that made them overlap.
    """
    # Same gate as /headway_frame, same reason: this path runs Qwen and Depth
    # Anything, and doing that while Qwen is mid-load is what breaks the load.
    if not _warm_done.is_set():
        return {"qwen_boxes": [], "corridor": [], "caption": "", "observation": "",
                "skipped": "warming", "timing_ms": {"total": 0.0}}

    # ...and the same staleness rule. This one costs a full Qwen generate, so
    # it is the more expensive of the two to serve for nobody.
    if session_id and not sessions.touch(session_id):
        return {"qwen_boxes": [], "corridor": [], "caption": "", "observation": "",
                "stale": True, "reason": "unknown_session",
                "timing_ms": {"total": 0.0}}

    import time as _t
    _t0 = _t.time()
    image_bytes = await image.read()
    result = await run_in_threadpool(perceive.perceive, image_bytes, bool(debug))
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


# --- Navigation (docs/navigation_v1.md) ---
#
# Six endpoints, none of them in the timing path between a driver and a turn.
# The tracker is client-side (static/rio_navcore.js) and the speech planner
# with it (static/rio_navplan.js); the server resolves destinations, computes
# routes, holds the one table of sentences RIO may say about them, verifies a
# landmark when asked, and writes the log.
#
# The authority firewall (§2) is visible in the shapes here: nothing a client
# POSTs can change a route's maneuvers, and /nav/voice will not synthesize a
# sentence that is not already in the route's own table.


@app.get("/nav/suggest")
def nav_suggest_endpoint(q: str = Query(...), lat: float = Query(default=None),
                         lng: float = Query(default=None),
                         session: str = Query(default=None)):
    """Destination autocomplete, proxied so the key stays on this side.

    Never fatal: an empty list just means the driver types the address in full
    and submits it, which resolves through the same provider. Autocomplete is a
    convenience on top of that path, never a gate in front of it.

    `session` is RIO's opaque id for one typing session, minted by the panel and
    sent on every keystroke and again on the selection. What a provider makes of
    it stops at the provider boundary — Google bills the whole session once; a
    provider with no such concept ignores it.
    """
    return {"suggestions": [c.to_dict() for c in
                            navservice.get_provider().suggest(q, lat, lng,
                                                              session=session)]}


@app.get("/nav/geocode")
def nav_geocode_endpoint(q: str = Query(...)):
    """Address -> lat/lng, for the desk-testing start-point override."""
    provider = navservice.get_provider()
    g = provider.geocode_point(q) if hasattr(provider, "geocode_point") else None
    return g or {"error": "could not find that place", "q": q}


@app.post("/nav/destination")
def nav_destination_endpoint(body: dict = Body(...),
                             session_id: str = Query(default=None)):
    """Resolve what the driver asked for — or ask them which one they meant.

    "Take me to LAX", "Navigate to Griffith Observatory", "Directions to 123
    Main Street", "Let's go to the Getty" all arrive here as free text. The
    last of those is two museums eight miles apart, and the honest answer to it
    is a question. RIO never silently picks one (§4).
    """
    res = navservice.resolve_destination(str(body.get("q") or ""),
                                         body.get("lat"), body.get("lng"),
                                         session=str(body.get("session") or "") or None)
    if res["status"] == "ambiguous":
        sessions.log_nav(session_id, navevents.DESTINATION_AMBIGUOUS,
                         {"query": res["query"],
                          "candidates": [c["display_name"] for c in res["candidates"]]})
        return {"status": "ambiguous", "query": res["query"],
                "candidates": res["candidates"]}
    if res["status"] != "resolved":
        return {"status": "not_found", "query": res.get("query", "")}
    return {"status": "resolved", "reason": res.get("reason"),
            "destination": res["destination"].to_dict()}


@app.post("/nav/route")
def nav_route_endpoint(body: dict = Body(...), session_id: str = Query(default=None)):
    """Compute a route and make it the active generation.

    The provider decides the route. The only judgement in this handler is
    refusing to pretend: a routing failure comes back as an error the panel
    shows, never as a stale or invented route.

    `reroute_of` carries the previous route id, which is what makes the new
    route the NEXT GENERATION of the same journey rather than an unrelated
    drive — and what lets every announcement queued against the old one be
    invalidated the moment it would otherwise be spoken.
    """
    try:
        lat = float(body.get("lat"))
        lng = float(body.get("lng"))
    except (TypeError, ValueError):
        return {"error": "need a current position (lat, lng) to route from"}

    previous = None
    reroute_of = body.get("reroute_of")
    if reroute_of:
        previous = navservice.get_route(str(reroute_of))
        if previous is not None and not navservice.reroute_allowed(previous.journey_id):
            # Anti-flap. A journey rerouting this often is not being rerouted,
            # it is oscillating, and each attempt costs a routing call.
            sessions.log_nav(session_id, navevents.REROUTE_FAILED, {
                "route_id": previous.route_id, "journey_id": previous.journey_id,
                "reason": "reroute_limit"})
            return {"error": "too many reroutes on this journey"}

    place_id = str(body.get("place_id") or "")
    label = str(body.get("label") or "")
    query = str(body.get("destination") or "")
    session = str(body.get("session") or "") or None
    provider = navservice.get_provider()
    if previous is not None:
        # A reroute goes back to the SAME destination object, never to a
        # re-resolution of its label: re-geocoding "Starbucks" from a different
        # part of town lands on a different Starbucks.
        destination = previous.destination
    elif place_id:
        # Picking a suggestion is what ends the typing session, so the session
        # id rides along with the selection — this lookup is the call the whole
        # autocomplete session is billed as.
        destination = provider.destination(place_id=place_id, label=label,
                                           session=session)
    else:
        destination = provider.destination(query=query, label=label,
                                           session=session)
    if destination is None:
        sessions.log_nav(session_id, navevents.ROUTE_FAILED,
                         {"destination": query or label, "error": "unresolved destination"})
        return {"error": "could not find that place"}

    if previous is not None:
        sessions.log_nav(session_id, navevents.REROUTE_STARTED, {
            "route_id": previous.route_id, "journey_id": previous.journey_id,
            "generation_id": previous.generation_id,
            "reason": str(body.get("reason") or "off_route")})
    try:
        route = navservice.build_route(lat, lng, destination, previous=previous)
    except navservice.NavError as e:
        sessions.log_nav(session_id, navevents.ROUTE_FAILED,
                         {"destination": destination.display_name, "error": str(e)})
        return {"error": str(e)}

    # Logged here rather than from the browser: the server is the only place
    # that holds the whole route, and this summary is the record a review reads
    # to see what RIO *intended* to be able to say on this drive.
    sessions.log_nav(session_id,
                     navevents.REROUTE_COMPLETE if previous is not None
                     else navevents.ROUTE_STARTED,
                     navservice.summary(route))
    return navservice.wire(route)


@app.get("/nav/voice")
def nav_voice_endpoint(route_id: str = Query(...), m: str = Query(...),
                       call: str = Query(...), anchor: str = Query(default=None)):
    """TTS for one precomputed line, addressed by (route, maneuver, call, anchor).

    Deliberately not a text-to-speech endpoint, for the same reason
    /headway_voice is not: the browser sends coordinates into a table, never a
    sentence. Anything that does not resolve to a line on a live route is
    refused, so the set of things RIO's voice can ever say about navigation is
    bounded by what the provider returned and this process precomputed — an
    anchor id included, which is why a landmark cannot be spoken about unless
    the map put it on the route in the first place.

    The exact sentence rides back on X-Nav-Text so the panel shows the words
    being spoken rather than its own reconstruction of them.
    """
    route = navservice.get_route(route_id)
    text = navspeech.text_for(route, m, call, anchor) if route else None
    if not text:
        return {"error": "unknown route, maneuver, call or anchor",
                "route_id": route_id, "m": m, "call": call}
    return StreamingResponse(
        voice.synthesize_stream(text), media_type="audio/mpeg",
        headers={
            "X-Nav-Text": quote(text, safe=""),
            "Access-Control-Expose-Headers": "X-Nav-Text",
            "Cache-Control": "no-store",
        })


@app.post("/nav/anchor/verify")
def nav_anchor_verify_endpoint(body: dict = Body(...),
                               session_id: str = Query(default=None)):
    """Is one of this maneuver's expected landmarks visible right now?

    The camera's entire role in navigation. The candidate list is taken from
    the ROUTE, not from the request body — a client cannot introduce a landmark
    RIO never looked up, and cannot alter the relation the map computed for it.

    Returns `{"anchor": null, "reason": ...}` freely and without apology: no
    camera, no recent frames, nothing visible, an uncertain identity, a
    duplicate in view, an unstable track, a stale look or an implausible depth
    all end here, and all of them mean RIO speaks the canonical instruction.
    """
    route = navservice.get_route(str(body.get("route_id") or ""))
    if route is None:
        return {"anchor": None, "reason": "unknown_route"}
    if int(body.get("generation_id") or route.generation_id) != route.generation_id:
        return {"anchor": None, "reason": "stale_generation"}
    man = route.maneuver(str(body.get("maneuver_id") or ""))
    if man is None:
        return {"anchor": None, "reason": "unknown_maneuver"}
    if not man.anchors:
        return {"anchor": None, "reason": "no_candidates"}

    result = navverify.verify(_visual_key(session_id), man.anchors)
    if result.get("anchor"):
        sessions.log_nav(session_id, navevents.ANCHOR_VERIFIED,
                         {"route_id": route.route_id,
                          "generation_id": route.generation_id,
                          "maneuver_id": man.id, "anchor": result["anchor"],
                          "observation": result.get("observation")})
    else:
        sessions.log_nav(session_id, navevents.ANCHOR_REJECTED,
                         {"route_id": route.route_id,
                          "generation_id": route.generation_id,
                          "maneuver_id": man.id, "reason": result.get("reason"),
                          "rejections": result.get("rejections")})
    return result


@app.post("/nav/event")
def nav_event_endpoint(body: dict = Body(...), session_id: str = Query(default=None)):
    """Record one navigation event (kind "nav") in the session JSONL."""
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

    This is also one of exactly two places the diagnostic engine is fed. The
    other is /vehicle/health/announcement. A conversation turn deliberately does
    NOT feed it: reports are deduplicated by their own timestamp so extra polls
    add no evidence, but calling it from a turn would still tie the drive-cycle
    and monitor-run bookkeeping to how talkative the driver is.
    """
    snap = tires.snapshot()
    _feed_diagnostics(snap)
    return snap


def _feed_diagnostics(snap: dict) -> None:
    """Hand a tire snapshot to the diagnostic monitors. Never fatal.

    Motion goes in the other direction first: direct TPMS sensors wake on
    rotation, so the provider has to be told whether the wheels are turning, and
    tires.py deliberately cannot find that out for itself.
    """
    try:
        moving, speed = _vehicle_motion()
        tires.set_motion(moving)
        tire_diag.observe(snap, moving=moving, speed_mph=speed)
    except Exception as e:
        print(f"[tire_diag] observe failed: {type(e).__name__}: {e}", flush=True)


def _dtc_view() -> dict:
    """What the engine's DTC monitor is told. A view, never a decision.

    Assembled here rather than fetched by powertrain_diag, so that domain cannot
    reach sideways into the DTC service — the same reason its monitors are
    handed their baselines instead of reading them.
    """
    try:
        svc = dtc_service.service()
        active = svc.registry.active_codes()
        stats = svc.stats()
        worst = None
        for rec in active:
            cand = dtc_catalog.health_severity(rec.get("severity", ""))
            if worst is None or (vehicle_health_policy.SEVERITY_RANK.get(cand, 0)
                                 > vehicle_health_policy.SEVERITY_RANK.get(worst, 0)):
                worst = cand
        return {"scanned": stats["scans"] > 0,
                "responding": stats["ecu_responding"],
                "codes": [r["code"] for r in active],
                "mil": svc.registry.mil(),
                "count": svc.registry.meta().get("dtc_count_reported"),
                "worst_health_severity": worst}
    except Exception as e:
        print(f"[dtc] view failed: {type(e).__name__}: {e}", flush=True)
        return {}


def _link_view() -> dict:
    """What the transport currently looks like, for the connection monitor."""
    try:
        gws = gateway_auth.gateways(config.VEHICLE_ID)
        live = next((g for g in gws if g["link"] == "connected"), None)
        hb = (live or {}).get("heartbeat") or {}
        return {"source": telemetry.source(),
                "can_state": hb.get("can_state"),
                "network_state": hb.get("network_state"),
                "outbox_pending": hb.get("outbox_pending"),
                "gateways": len(gws)}
    except Exception:
        return {"source": telemetry.source()}


def _feed_engine_diagnostics() -> None:
    """Hand the current engine picture to the powertrain monitors. Never fatal.

    Fed from the announcement poll for the reason the tire monitors are: this is
    the loop that keeps running when nobody has the dashboard open, and a
    monitor that only advanced while somebody was watching would have its
    confirmation counts measure attention rather than persistence.

    record=False, deliberately. This read must not add a sample to the trend
    ring or advance the simulator — see telemetry.snapshot's docstring.
    """
    try:
        moving, speed = _vehicle_motion()
        powertrain_diag.observe(telemetry.snapshot(record=False),
                                moving=moving, speed_mph=speed,
                                dtc=_dtc_view(), link=_link_view())
    except Exception as e:
        print(f"[powertrain_diag] observe failed: {type(e).__name__}: {e}",
              flush=True)


def _poll_dtc() -> None:
    """Let the DTC scheduler ask for whatever is due. Never fatal.

    Driven from the announcement poll rather than from the telemetry poll, for
    the reason the diagnostic engine is: this is the loop that keeps running
    when nobody has the dashboard open, and a diagnostic scan that only happened
    while somebody was watching would find every pending code late.

    The scheduler decides what is actually due — see vehicle/dtc/service.py on
    why it is not "ask for everything every time".
    """
    try:
        dtc_service.service().poll(time.time())
    except Exception as e:
        print(f"[dtc] poll failed: {type(e).__name__}: {e}", flush=True)


def _vehicle_motion():
    """-> (moving, speed_mph). Conservative: unknown speed is not moving.

    The only thing this flag can do on its own is promote a quiet sensor to an
    urgent condition, so guessing that the car is moving when nobody knows would
    mean interrupting a parked driver about sensors that are asleep by design.
    """
    try:
        snap = telemetry.snapshot(record=False)
        for row in snap.get("rows", []):
            if row["id"] == "vehicle_speed" and row.get("value") is not None:
                speed = float(row["value"])
                return speed >= config.TIRE_DIAG_DRIVE_START_MPH, speed
    except Exception:
        pass
    return False, None


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


@app.get("/vehicle/telemetry/source")
def vehicle_telemetry_source_endpoint():
    """Which producer the pipeline is listening to, and what else it could.

    The spec's source selector. What it switches is the PROVIDER and nothing
    else: the bands, the trend window, the staleness rule, the insight engine,
    the diagnostic monitors, the conversation layer and the browser are
    identical for every option in this list. That is the whole claim of the
    canonical pipeline, and it is worth being able to check by hand.
    """
    return {"source": telemetry.source(), "sources": telemetry.sources()}


@app.post("/vehicle/telemetry/source")
def vehicle_telemetry_source_set_endpoint(name: str = Query(...)):
    """Switch producers. No restart, and no state is discarded but the trend ring.

    The ring goes because a slope fitted across the moment the source changed is
    a slope across a discontinuity. Diagnostic issues, monitor counters, the
    insight baselines and the announcement ledger all survive — switching what
    you are listening to is not evidence that anything was repaired.
    """
    if not telemetry.set_source(name):
        return {"error": "unknown source", "name": name,
                "sources": telemetry.sources()}
    return telemetry.snapshot(record=False)


# --- the in-process producers, and the link they talk over ------------------
# Development controls. The simulator and the replay both build canonical events
# and hand them to the same ingestion path a bridge posts to — they are not a
# shortcut into the pipeline, which is the entire reason they are worth having.
#
# The fault injector sits between building and ingesting, because that is where
# a link lives. It can delay, drop, duplicate, reorder and skew; it can never
# fabricate a reading.

from vehicle import faults as vehicle_faults              # noqa: E402
from vehicle import producers as vehicle_producers        # noqa: E402
from vehicle.producers import replay as vehicle_replay    # noqa: E402
from vehicle.producers import simulator as vehicle_sim    # noqa: E402


@app.get("/vehicle/producers")
def vehicle_producers_endpoint():
    """What the in-process producers are doing, and what the link is doing to
    them. Diagnostics for the §27.9 scenario list."""
    return vehicle_producers.stats()


@app.get("/vehicle/simulator/scenario")
def vehicle_simulator_scenario_endpoint():
    sim = vehicle_sim.simulator()
    return {"scenario": sim.scenario, "scenarios": sim.scenarios(),
            "stats": sim.stats()}


@app.post("/vehicle/simulator/scenario")
def vehicle_simulator_scenario_set_endpoint(name: str = Query(...)):
    """Switch the simulated vehicle's condition. Development only.

    The clock restarts with the scenario: every one of them is a function of
    elapsed time, and an overheat that remembered where it got to an hour ago
    would jump 60°F between two consecutive readings — which the range check
    would then correctly refuse as an impossible step.
    """
    sim = vehicle_sim.simulator()
    if not sim.set_scenario(name):
        return {"error": "unknown scenario", "name": name,
                "scenarios": sim.scenarios()}
    return {"scenario": sim.scenario, "scenarios": sim.scenarios()}


@app.get("/vehicle/faults")
def vehicle_faults_endpoint():
    return vehicle_faults.injector().stats()


@app.post("/vehicle/faults")
def vehicle_faults_set_endpoint(mode: str = Query(...)):
    """Break the link on purpose. Development only.

    Clearing a cloud disconnect RELEASES what it held rather than discarding it,
    because that is what a reconnecting bridge does — the outbox empties. That
    release is also how the out-of-order path gets exercised for real.
    """
    inj = vehicle_faults.injector()
    if not inj.set_mode(mode):
        return {"error": "unknown fault mode", "mode": mode,
                "modes": [m["name"] for m in inj.stats()["modes"]]}
    return inj.stats()


@app.post("/vehicle/replay/load")
def vehicle_replay_load_endpoint(path: str = Query(...)):
    """Load a recorded canonical log. Nothing is sent until it is started."""
    return vehicle_replay.replay().load(path)


@app.post("/vehicle/replay/start")
def vehicle_replay_start_endpoint(speed: float = Query(default=1.0),
                                  loop: bool = Query(default=False),
                                  session_id: str = Query(default=None)):
    """Play it back. `speed` compresses the relative timing only.

    A two-hour drive at 120x is a minute and every monitor sees the same values
    in the same order, which is what makes an hour-long slow-leak window
    testable. What it does not do is change the observed_at spacing: a monitor
    asking how long something took gets the answer the drive gave.
    """
    return vehicle_replay.replay().start(time.time(), speed=speed, loop=loop,
                                         session_id=session_id)


@app.post("/vehicle/replay/stop")
def vehicle_replay_stop_endpoint():
    return vehicle_replay.replay().stop()


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


# ---------------------------------------------------------------------------
# Vehicle health — the conversation context and the announcement channel
# ---------------------------------------------------------------------------
# The decision to speak is made HERE, on the server, in deterministic code. The
# browser polls, and when an announcement is due it submits it to the speech
# arbiter it already owns. That split is deliberate and it is the same one
# navigation uses: the policy that decides is server-side and testable, the
# mouth is client-side because that is where the audio device is.
#
# One policy instance for the process — the same single-driver assumption
# `_last_talk` and nav's route registry already make.

_health_policy = vehicle_health_policy.VehicleHealthPolicy()


@app.get("/vehicle/health")
def vehicle_health_endpoint(full: bool = Query(default=True)):
    """The normalized vehicle health context.

    The same object the conversation layer injects, exposed for the dashboard,
    the acceptance tests and anyone debugging what RIO actually knows. Read-only
    and side-effect free: it does not tick the announcement policy, so opening
    this in a browser tab cannot consume an announcement the driver should have
    heard.
    """
    return vehicle_health.context(full=full)


@app.get("/vehicle/health/announcement")
def vehicle_health_announcement_endpoint():
    """Is there something the driver has to be told right now?

    This is the tick. The policy is pure and clockless, so time is supplied
    here, and everything it decides — including every silence and the reason for
    it — comes back on the wire where the panel and the log can see it.

    `announce` is null on almost every poll, which is the point: an announcement
    is what a check engine light does, not what a dashboard does.
    """
    if not getattr(config, "VEHICLE_HEALTH_ENABLED", True):
        return {"announce": None, "reason": "disabled",
                "poll_ms": config.HEALTH_POLL_MS}

    # Feed the monitors first, then ask. This poll and /vehicle/tires are the
    # only two things that advance the diagnostic engine, and this one is the
    # one that keeps running when nobody has the dashboard open.
    _feed_diagnostics(tires.snapshot())
    _poll_dtc()
    _feed_engine_diagnostics()

    now = time.time()
    issues = vehicle_health.issues()
    out = _health_policy.tick(issues, now)

    # The communication ledger is the ENGINE's, not the policy's, because it has
    # to survive a restart: a driver who has already been told about a tire must
    # not be told again because the process bounced. The policy stays pure and
    # writes nothing — it decides, and the write happens here.
    ann = out.get("announce")
    if ann and ann.get("issue_id"):
        tire_diag.engine().note_announced(ann["issue_id"], ann.get("severity"), now)
    proposal = out.get("proposal")
    if proposal and proposal.get("issue_id"):
        tire_diag.engine().note_shadow_proposal(
            proposal["issue_id"], proposal.get("text", ""),
            proposal.get("severity"), proposal.get("would_have_fired_because", ""),
            now)

    out["poll_ms"] = config.HEALTH_POLL_MS
    out["shadow_mode"] = bool(config.TIRE_DIAG_SHADOW_MODE)
    # Enough state for the panel to stay in step with what RIO believes, without
    # it having to compute anything: same contract as every other payload here.
    out["overall_status"] = vehicle_health.overall_status(issues)
    out["issue_count"] = len(issues)
    return out


@app.get("/vehicle/health/voice")
def vehicle_health_voice_endpoint(id: str = Query(...)):
    """TTS for one announcement, addressed by the id the policy issued.

    Deliberately not a text-to-speech endpoint, for exactly the reason
    /nav/voice and /headway_voice are not: the browser sends an id into a table,
    never a sentence. An id the policy never issued — or one it has since
    retired — is refused, so the set of things RIO's voice can ever say
    unprompted about the car is bounded by vehicle_health_policy.LINE and
    nothing else can add to it.

    The words ride back on X-Health-Text so the Voice Layer column shows what is
    being spoken rather than its own reconstruction of it.
    """
    text = _health_policy.text_for(id)
    if not text:
        return {"error": "unknown or expired announcement", "id": id}
    return StreamingResponse(
        voice.synthesize_stream(text), media_type="audio/mpeg",
        headers={
            "X-Health-Text": quote(text, safe=""),
            "Access-Control-Expose-Headers": "X-Health-Text",
            "Cache-Control": "no-store",
        })


@app.get("/vehicle/health/policy")
def vehicle_health_policy_endpoint():
    """What the announcement policy is holding, and why it has been quiet.

    Diagnostics. A silence with a reason attached is as informative as an
    utterance — the same argument headway/live_policy.py's voice_log makes — and
    without this the only way to know why RIO said nothing is to reproduce it.
    """
    return {"state": _health_policy.state(), "log": _health_policy.log[-40:]}


# ---------------------------------------------------------------------------
# Tire diagnostics (tire_diag/) — OBD-inspired, not OBD-II
# ---------------------------------------------------------------------------
# RIO Tire Health is not an OBD-II system and emits no SAE powertrain codes.
# These endpoints are the service view: what each monitor could and could not
# evaluate, what it found, what has been confirmed, and the frozen evidence
# behind each confirmation. Nothing here is shown to the driver — the
# conversation layer gets driver_term and nothing else.

@app.get("/vehicle/diagnostics")
def vehicle_diagnostics_endpoint():
    """Monitor readiness, issues and drive cycle. The whole diagnostic state.

    `status` and `last_result` are separate fields on every monitor, and that is
    the point of the view: a monitor that has not run has no result, and
    reporting either as a pass would be the system claiming more certainty than
    its evidence supports.
    """
    return tire_diag.state()


@app.get("/vehicle/diagnostics/events")
def vehicle_diagnostics_events_endpoint(limit: int = Query(default=100),
                                        issue_id: str = Query(default=None)):
    """The append-only record: candidates, confirmations, freeze frames,
    resolutions, recurrences, relearns, and every announcement — including the
    ones shadow mode only proposed."""
    from tire_diag import store as tire_store
    return {"events": tire_store.read_events(limit=limit, issue_id=issue_id),
            "paths": tire_store.paths()}


@app.get("/vehicle/diagnostics/engine")
def vehicle_diagnostics_engine_endpoint():
    """The powertrain domain's diagnostic state.

    The same shape /vehicle/diagnostics returns for the tires, because it is
    literally the same code — powertrain_diag is an instance of diag/, not a
    second diagnostic system. `status` and `last_result` are separate fields on
    every monitor here for the same reason they are there.
    """
    return powertrain_diag.state()


@app.get("/vehicle/diagnostics/catalogue")
def vehicle_diagnostics_catalogue_endpoint():
    """Every diagnostic code and every monitor definition, fully described.

    A service view, deliberately verbose. These identifiers never reach the
    driver: RIO says "your rear-left tire may have a slow leak", not
    "RIO-TIRE-POSSIBLE-LEAK-RL is active".

    Both domains, side by side, with their own shadow flags. Those flags are
    separate because the tire monitors have shadow logs from real drives behind
    them and the engine monitors have never seen a vehicle — and one boolean
    would clear both in the same edit.
    """
    from diag import shadow as diag_shadow
    return {"domains": {
        "tires": {"codes": tire_codes.service_view(),
                  "monitors": tire_monitors.definitions_view(),
                  "shadow_mode": diag_shadow.is_shadowed("tires")},
        "powertrain": {"codes": powertrain_codes.service_view(),
                       "monitors": powertrain_monitors.definitions_view(),
                       "shadow_mode": diag_shadow.is_shadowed("powertrain")},
    },
        "shadow_by_domain": diag_shadow.registered(),
        # The original shape, for anything already reading it.
        "codes": tire_codes.service_view(),
        "monitors": tire_monitors.definitions_view(),
        "shadow_mode": bool(config.TIRE_DIAG_SHADOW_MODE)}


@app.post("/vehicle/diagnostics/relearn")
def vehicle_diagnostics_relearn_endpoint(corner: str = Query(default=None),
                                         reason: str = Query(default=""),
                                         by: str = Query(default="driver")):
    """Sensors replaced, tires rotated, or a baseline deliberately reset.

    Deletes nothing. Trend monitors go NOT_READY because they genuinely are;
    absolute pressure monitoring stays live the moment a reliable reading
    exists, and a relearn can never suppress a validated critical condition.
    """
    return tire_diag.engine().relearn(corner=corner, reason=reason, by=by)


# ---------------------------------------------------------------------------
# The canonical vehicle data API (§25)
# ---------------------------------------------------------------------------
# Versioned and vehicle-scoped, unlike everything above it, and deliberately so.
# These are the routes a device outside this process posts to — today a laptop
# bridge with a CANable in it, later a Jetson — and the whole migration story
# rests on the contract not changing when the hardware does. The unversioned
# /vehicle/* routes stay exactly as they are: they are the dashboard's, they are
# read-only, and nothing outside this pod calls them.
#
# SINGLE VEHICLE, MULTI-VEHICLE CONTRACT. Every route takes a vehicle_id and
# every payload carries one, so the contract is already right. The STATE behind
# it is not: the announcement policy, both diagnostic engines and the ingestion
# buffer are one-per-process, exactly as _last_talk and nav's route registry
# have always been. Saying so here is cheaper than discovering it later.
#
# READ-ONLY POSTURE. There is no route below that asks a vehicle to do anything.
# No code clearing, no readiness reset, no actuator test, no ECU write, no
# Holley transmit — and no command channel a bridge could poll, which is the
# part that would be easy to add by accident. vehicle/selftest.py asserts their
# absence by parsing this file rather than by trusting this comment.

import vehicle.ingest as vehicle_ingest                    # noqa: E402
from vehicle.gateway import auth as gateway_auth           # noqa: E402
from vehicle.providers import ingested as vehicle_ingested  # noqa: E402
from vehicle.signals import registry as signal_registry    # noqa: E402


def _gateway_credentials(gateway_id: str, token: str) -> tuple:
    """Credentials from headers. One place, so no route invents its own scheme."""
    return (gateway_id or "").strip(), (token or "").strip()


@app.post("/api/v1/vehicle-gateways/register")
def gateway_register_endpoint(body: dict = Body(...)):
    """Admit a gateway and issue it a token.

    The token comes back exactly once and is stored hashed — see
    vehicle/gateway/auth.py. A bridge that loses it re-registers.

    Registration is REFUSED when no bootstrap key is configured, rather than
    allowed. An unconfigured deployment that accepts any device is worse than
    one that accepts none, because the first failure is silent.
    """
    try:
        out = gateway_auth.register(
            device_name=str(body.get("device_name") or ""),
            vehicle_id=str(body.get("vehicle_id") or config.VEHICLE_ID),
            registration_key=str(body.get("registration_key") or ""),
            hardware_type=str(body.get("hardware_type") or "unknown"),
            firmware_type=str(body.get("firmware_type") or "unknown"),
            bridge_version=str(body.get("bridge_version") or "0.0.0"),
            gateway_id=body.get("gateway_id"))
    except gateway_auth.AuthError as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    return out


@app.post("/api/v1/vehicle-gateways/heartbeat")
def gateway_heartbeat_endpoint(body: dict = Body(...),
                               x_gateway_id: str = Header(default=""),
                               x_gateway_token: str = Header(default="")):
    """"I am still here, and this is what I think of myself."

    The bridge reports its own view — CAN state, network state, how many events
    are stuck in its outbox. The cloud records that verbatim and forms its own
    opinion of the link from when it last heard, because a bridge that believes
    its network is fine and has not been heard from in five minutes is exactly
    the disagreement worth surfacing.
    """
    gid, token = _gateway_credentials(x_gateway_id, x_gateway_token)
    try:
        rec = gateway_auth.authenticate(gid, token)
        gateway_auth.authorize_vehicle(rec, str(body.get("vehicle_id") or ""))
        return gateway_auth.heartbeat(gid, body)
    except gateway_auth.AuthError as e:
        return JSONResponse({"error": str(e)}, status_code=401)


@app.get("/api/v1/vehicle-gateways")
def gateway_list_endpoint(vehicle_id: str = Query(default=None)):
    """Every gateway and what the cloud believes about it. Never a credential."""
    return {"gateways": gateway_auth.gateways(vehicle_id),
            "registration_configured": gateway_auth.registration_enabled(),
            "stale_after_s": config.VEHICLE_GATEWAY_STALE_S}


@app.post("/api/v1/vehicle-telemetry/batches")
async def telemetry_batch_endpoint(request: Request,
                                   x_gateway_id: str = Header(default=""),
                                   x_gateway_token: str = Header(default="")):
    """Canonical telemetry, in batches. The one door vehicle data comes through.

    Read as raw bytes rather than a parsed body so the size ceiling is applied
    BEFORE the JSON is materialised — a payload ten times the limit should cost
    a length check, not a parse.

    Per-event results, not a single verdict: one undecodable frame in two
    hundred good readings must not cost the two hundred, and a bridge needs to
    know which ids to stop resending without throwing away its backlog.
    """
    raw = await request.body()
    if len(raw) > config.VEHICLE_INGEST_MAX_BYTES:
        return JSONResponse(
            {"error": f"payload is {len(raw)} bytes, over the "
                      f"{config.VEHICLE_INGEST_MAX_BYTES} limit"},
            status_code=413)
    try:
        import json as _json
        body = _json.loads(raw or b"{}")
    except Exception as e:
        return JSONResponse({"error": f"malformed JSON: {e}"}, status_code=400)

    gid, token = _gateway_credentials(x_gateway_id, x_gateway_token)
    try:
        return await run_in_threadpool(vehicle_ingest.ingest_batch, body, gid, token)
    except vehicle_ingest.IngestError as e:
        return JSONResponse({"error": str(e)}, status_code=e.status)


@app.get("/api/v1/vehicle-telemetry/stats")
def telemetry_stats_endpoint():
    """What the ingestion layer has seen. Diagnostics for a bridge under test."""
    return vehicle_ingest.stats()


# --- diagnostic trouble codes ----------------------------------------------
# Modes 01, 02, 03, 07, 09 and 0A. There is no route here that changes anything
# about the vehicle, and specifically no Mode 04: the prototype must not expose
# or call the code-clearing service, and the way to guarantee that is for the
# capability not to exist rather than for it to be guarded.

from vehicle.dtc import catalog as dtc_catalog             # noqa: E402
from vehicle.dtc import service as dtc_service             # noqa: E402
from vehicle.producers import ecu as vehicle_ecu           # noqa: E402


@app.get("/vehicle/dtc")
def vehicle_dtc_endpoint():
    """The Flagged Error Codes section (§17), already grouped and worded.

    Rendered verbatim by the browser, like every other payload here. The
    grouping, the status labels, the severity order and the empty-state sentence
    are all decided server-side — a status label computed in JavaScript is a
    status label that will one day disagree with the lifecycle that produced it.
    """
    return dtc_service.service().section()


@app.get("/vehicle/conditions")
def vehicle_conditions_endpoint():
    """RIO-Observed Conditions (§27.5), already worded and already ordered.

    Everything RIO has inferred, and nothing the vehicle reported — the codes
    have their own section and mixing them would undo the one distinction this
    whole feature rests on.

    Each condition carries what §27.5 asks for: the observation, its provenance,
    a confidence, the signals behind it, when it was first seen, and whether a
    cause has been confirmed. That last field is always "not confirmed" unless a
    person put something there, and the browser prints it rather than deciding
    it.
    """
    from vehicle.signals import provenance as prov

    issues = vehicle_health.issues()
    conditions, gaps = [], []
    for i in issues:
        if i["domain"] == "diagnostics":
            continue
        itype = i.get("type") or ""
        row = {
            "observation": i["message"],
            "domain": i["domain"],
            "type": itype,
            "severity": i["severity"],
            "severity_rank": i["severity_rank"],
            "state": i["severity"],
            "where": i["location"] or None,
            "observation_window": i["observation_window"],
            "provenance": prov.RIO_OBSERVED_PATTERN,
            "provenance_label": prov.display(prov.RIO_OBSERVED_PATTERN),
            "confidence": (i.get("evidence") or {}).get("confidence"),
            "supporting_signals": sorted(
                (i.get("evidence") or {}).get("measurement", {}).keys()),
            "first_observed_at": (i.get("evidence") or {}).get("confirmed_at"),
            "possible_explanation": i["suggested_action"] or None,
            "cause_confirmed": False,
        }
        if itype.endswith("_unavailable") or "not_ready" in itype \
                or "not_evaluated" in itype:
            gaps.append(row)
        else:
            conditions.append(row)

    return {
        "conditions": conditions,
        "count": len(conditions),
        # What RIO could not look at, kept apart from what it found. A panel
        # that listed them together would make an unmonitored car look like a
        # troubled one — and hiding them would be worse still.
        "coverage_gaps": gaps,
        "gap_count": len(gaps),
        "empty_state": "RIO has not observed anything unusual.",
        "empty_state_caveat": ("This covers only what RIO can currently see. "
                               "It is not a statement about the whole vehicle."),
        "poll_ms": config.HEALTH_POLL_MS,
        # The panel addresses the report endpoint with this, rather than holding
        # its own copy of config.VEHICLE_ID — a constant duplicated into the
        # browser is a constant that will one day be stale.
        "vehicle_id": config.VEHICLE_ID,
        # §27.8's progress states, in order, so the report button can show the
        # list of questions it is about to ask before the first answer lands.
        "report_stages": list(vehicle_report.STAGES),
    }


@app.get("/vehicle/dtc/catalogue")
def vehicle_dtc_catalogue_endpoint():
    """Every code RIO has a definition for. A service view.

    Deliberately finite and deliberately incomplete. A code that is not in here
    is still read, stored, displayed and reported — it simply says so rather
    than guessing, which is the whole of §15's "unknown and manufacturer-specific
    codes must still be stored and displayed".
    """
    return {"codes": dtc_catalog.view(),
            "severity_labels": dtc_catalog.SEVERITY_LABEL,
            "systems": dtc_catalog.SYSTEM_LABEL}


@app.get("/vehicle/dtc/snapshot")
def vehicle_dtc_snapshot_endpoint(id: str = Query(...)):
    """RIO's own recording of the minute either side of a code appearing.

    Labelled `rio_recorded_history` in the payload and NEVER merged with the
    ECU's freeze frame, which is on the code's own record. One is what the
    vehicle chose to preserve; the other is what RIO happened to be watching,
    and presenting them alike would give RIO's observations the vehicle's
    authority.
    """
    snap = dtc_service.service().snapshots.get(id)
    if snap is None:
        return {"error": "unknown snapshot", "id": id}
    return snap


@app.get("/vehicle/ecu/scenario")
def vehicle_ecu_scenario_endpoint():
    return vehicle_ecu.ecu().stats()


@app.post("/vehicle/ecu/scenario")
def vehicle_ecu_scenario_set_endpoint(name: str = Query(...)):
    """Switch what the simulated ECU reports. Development only."""
    e = vehicle_ecu.ecu()
    if not e.set_scenario(name, now=time.time()):
        return {"error": "unknown scenario", "name": name,
                "scenarios": e.scenarios()}
    return e.stats()


@app.get("/api/v1/vehicles/{vehicle_id}/dtcs")
def vehicle_dtcs_endpoint(vehicle_id: str):
    """Codes the vehicle is currently reporting, worst first (§25.6)."""
    svc = dtc_service.service()
    return {"vehicle_id": vehicle_id,
            "dtcs": [svc.card(r) for r in svc.registry.active_codes()],
            "mil_commanded_on": svc.registry.mil(),
            "stats": svc.stats()}


@app.get("/api/v1/vehicles/{vehicle_id}/dtcs/history")
def vehicle_dtcs_history_endpoint(vehicle_id: str):
    """Every code this vehicle has ever reported (§25.7).

    Including the ones that stopped being reported. A code that went away is
    still the vehicle's history, and "has this happened before" is only
    answerable because nothing here deletes it.
    """
    svc = dtc_service.service()
    return {"vehicle_id": vehicle_id,
            "dtcs": [svc.card(r) for r in svc.registry.records()],
            "events": svc.events(limit=200)}


@app.get("/api/v1/vehicles/{vehicle_id}/dtcs/{code}")
def vehicle_dtc_detail_endpoint(vehicle_id: str, code: str):
    """One code in full (§25.8)."""
    svc = dtc_service.service()
    rec = svc.registry.get(code)
    if rec is None:
        return JSONResponse({"error": "unknown code for this vehicle",
                             "code": code}, status_code=404)
    card = svc.card(rec)
    card["rio_snapshot"] = svc.snapshots.get(rec.get("snapshot_id") or "")
    return card


@app.post("/api/v1/vehicles/{vehicle_id}/dtc-scans")
def vehicle_dtc_scan_endpoint(vehicle_id: str,
                              session_id: str = Query(default=None),
                              reason: str = Query(default="driver_requested")):
    """Ask the vehicle for its codes now (§25.5).

    A READ. Modes 03, 07 and 0A return what the ECU has stored; none of them
    changes it. §16.4 lists the moments that justify asking out of turn — the
    drive started, the lamp changed, the driver asked — and this is how they are
    honoured, because a cadence alone would miss every one of them.
    """
    return dtc_service.service().scan(time.time(), session_id, reason=reason)


@app.post("/api/v1/vehicle-diagnostics/dtcs")
def vehicle_dtc_ingest_endpoint(body: dict = Body(...),
                                x_gateway_id: str = Header(default=""),
                                x_gateway_token: str = Header(default="")):
    """A scan performed by a gateway and uploaded (§25.4).

    The bridge does the talking to the vehicle; the cloud does the deciding. The
    payload is the same shape the simulated ECU produces, and it goes through
    the same registry — so a code's lifecycle is identical whether it was found
    by a CANable in a car or by a scenario on a laptop.
    """
    gid, token = _gateway_credentials(x_gateway_id, x_gateway_token)
    try:
        rec = gateway_auth.authenticate(gid, token)
        gateway_auth.authorize_vehicle(rec, str(body.get("vehicle_id") or ""))
    except gateway_auth.AuthError as e:
        return JSONResponse({"error": str(e)}, status_code=401)
    svc = dtc_service.service()
    return svc._ingest_direct(body, time.time())


# --- the diagnostic report --------------------------------------------------

from vehicle import report as vehicle_report                # noqa: E402
from vehicle import state as vehicle_state_mod              # noqa: E402


def _vehicle_state_view(snap: dict = None) -> dict:
    """§21's state, DERIVED from what already knows. See vehicle/state.py."""
    snap = telemetry.snapshot(record=False) if snap is None else snap
    try:
        cycle = powertrain_diag.engine().cycles.state().get("current")
    except Exception:
        cycle = None
    open_sessions = bool(sessions.active_sessions())
    return vehicle_state_mod.derive(snap, open_sessions, cycle)


def _run_report(session_id: str = None) -> dict:
    """Build a report and run it. Everything it needs is handed in.

    The report module imports nothing from the conversation layer, the
    announcement policy or a model, and this is where that is arranged — so the
    whole document is testable without a server, and cannot grow a dependency
    on the thing that narrates it.
    """
    report = vehicle_report.store().create(config.VEHICLE_ID, session_id)
    return report.run(
        dtc_service=dtc_service.service(),
        telemetry_snapshot=lambda: telemetry.snapshot(record=False),
        vehicle_state=lambda snap: _vehicle_state_view(snap),
        health_issues=vehicle_health.issues,
        insight_feed=lambda: insights.snapshot().get("entries", []),
        gateways=lambda: gateway_auth.gateways(config.VEHICLE_ID),
        capability=lambda: vehicle_ingested.buffer().capability(),
    )


@app.post("/api/v1/vehicles/{vehicle_id}/diagnostic-reports")
async def diagnostic_report_request_endpoint(vehicle_id: str,
                                             session_id: str = Query(default=None)):
    """Run a full diagnostic report (§25.9).

    Synchronous today because the whole thing takes milliseconds against a
    simulated ECU and a bridge's scan is a single upload. The PROGRESS contract
    is here regardless — stage, stage_index and the full stage list come back on
    every read — because the moment a real vehicle is on the other end this
    becomes seconds of a driver watching a button, and "Checking Pending Codes"
    is a more honest thing to show them than a spinner.
    """
    return await run_in_threadpool(_run_report, session_id)


@app.get("/api/v1/diagnostic-reports/{report_id}")
def diagnostic_report_status_endpoint(report_id: str):
    """One report, with its progress (§25.10)."""
    report = vehicle_report.store().get(report_id)
    if report is None:
        return JSONResponse({"error": "unknown report", "report_id": report_id},
                            status_code=404)
    return report.view()


@app.get("/api/v1/vehicles/{vehicle_id}/diagnostic-reports")
def diagnostic_report_list_endpoint(vehicle_id: str):
    """Reports are vehicle history (§29.1) and are kept as such."""
    return {"vehicle_id": vehicle_id, "reports": vehicle_report.store().list()}


@app.get("/api/v1/vehicles/{vehicle_id}/health")
def vehicle_health_api_endpoint(vehicle_id: str):
    """Current vehicle health (§25.11), in the canonical API's shape.

    The same object /vehicle/health returns — the conversation layer's context —
    plus the derived vehicle state and the gateway view. One health picture, two
    front doors, and deliberately not two computations of it.
    """
    ctx = vehicle_health.context(full=True)
    ctx["vehicle_id"] = vehicle_id
    ctx["vehicle_state"] = _vehicle_state_view()
    ctx["gateways"] = gateway_auth.gateways(vehicle_id)
    return ctx


@app.get("/vehicle/state")
def vehicle_state_endpoint():
    """§21's eight states, derived rather than tracked. See vehicle/state.py."""
    return _vehicle_state_view()


@app.get("/api/v1/vehicles/{vehicle_id}/signals")
def vehicle_signals_endpoint(vehicle_id: str):
    """The canonical registry, and which of it this vehicle has ever produced.

    The capability report §32.7 asks for. A signal that has never arrived is
    NOT a fault — most vehicles do not expose most PIDs — and separating
    "unsupported" from "missing" here is what stops the panel showing an empty
    row that looks exactly like a sensor that has died.
    """
    cap = vehicle_ingested.buffer().capability()
    return {"vehicle_id": vehicle_id,
            "signals": signal_registry.view(),
            "supported": cap["supported"],
            "unsupported": cap["unsupported"]}


# --- Phase 2.5 session endpoints ---

@app.post("/session/start")
def session_start_endpoint(metadata: dict = Body(default=None, embed=True)):
    sid = sessions.start_session(metadata=metadata)
    # A drive cycle rides on the session rather than inventing its own notion of
    # a drive. sessions.py already knows when one starts and ends, including the
    # untidy ending where the client simply vanishes — see _teardown_session.
    try:
        eng = tire_diag.engine()
        eng.cycles.note_session_start(sid, active_issue_ids=eng.active_issue_ids())
    except Exception as e:
        print(f"[tire_diag] drive cycle start failed: {type(e).__name__}: {e}",
              flush=True)
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
    # Stop describing a road nobody is driving on. It holds the same GPU the
    # detector and the depth model want, and its whole justification is that a
    # conversation is open.
    try:
        import observer

        observer.stop(key)
    except Exception:
        pass
    framebuf.drop_ring(key)
    visual_qa.drop_session(key)
    # The drive cycle ends with the drive. Note what does NOT happen here: no
    # diagnostic issue is cleared, no monitor counter is reset and no history is
    # dropped. A drive ending is not evidence that a tire was fixed, and a
    # system where ending a session repaired the car would be one where the way
    # to clear a fault is to close the tab.
    try:
        tire_diag.engine().cycles.note_session_end(session_id)
    except Exception as e:
        print(f"[tire_diag] drive cycle end failed: {type(e).__name__}: {e}",
              flush=True)
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
