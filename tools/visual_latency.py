"""visual_latency.py — why "what do you see?" takes so long.

    python -m tools.visual_latency                      # the local half
    python -m tools.visual_latency --live --n 10        # + a real session

"Slow" is a feeling, and a feeling cannot be fixed. This turns it into a
distribution over the stages a visual turn actually has, so the thing that is
slow can be found rather than guessed at -- and so that a change can be shown
to have moved it.

WHAT A VISUAL TURN IS MADE OF
-----------------------------
    route          classify the question                       (µs, local)
    select_frame   pick the clearest recent frame + graph       (ms, local)
    resolve        which object is "the black one"              (ms, local)
    crop           the frame that best shows THAT object        (ms, local)
    enrich         colour/body-style per object -- QWEN, local  (~1 s each,
                   cached for ENRICH_TTL_S)
    gpt            the multimodal answer -- OPENAI, remote, with a reasoning
                   pass; `gpt_first_token` is when the first word could be
                   spoken, which is what the driver actually waits for

The local stages are all in-RAM: the frame ring is memory, the graph is
arithmetic. Anything slow is therefore one of the two model calls, and they are
different models in different places for different reasons.

WHAT THIS DOES NOT MEASURE
--------------------------
The turn-taking around it: the live model deciding to call the tool, the
round trip to this server, and the time RIO then takes to start speaking. That
is what --live is for, and it is honest about which part is which -- the
`took_ms` the tool reports is the same number this file's local half breaks
down, so the difference between them IS the round trip.
"""
import argparse
import asyncio
import contextlib
import glob
import json
import os
import re
import statistics
import sys
import threading
import time

sys.path.insert(0, "/workspace/rio-phase1")

from dotenv import load_dotenv                    # noqa: E402

load_dotenv("/workspace/rio-phase1/.env")

import config                                     # noqa: E402
import framebuf                                   # noqa: E402
import realtime                                   # noqa: E402
import visual_qa                                  # noqa: E402
from headway import live as headway_live          # noqa: E402

SESSION = "latency_probe"

# The questions a driver actually asks, in the proportion they ask them: mostly
# "what is out there", sometimes about one thing in particular.
QUESTIONS = [
    "what do you see",
    "what's around us right now",
    "describe the road ahead",
    "what's that car ahead",
    "what do you see out there",
    "anything I should know about ahead",
    "what's on the left",
    "what do you see",
    "what colour is the car in front",
    "what's the road like",
]


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def fmt(values, unit="ms"):
    if not values:
        return "     --"
    return (f"p50 {pct(values, 0.5):7.0f} {unit}   p95 {pct(values, 0.95):7.0f} {unit}"
            f"   max {max(values):7.0f} {unit}   n={len(values)}")


def load_frames(video, dt, limit):
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * dt)))
    out, i = [], 0
    while len(out) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok2:
                out.append(buf.tobytes())
        i += 1
    cap.release()
    return out


def warm():
    """Every model loaded off the measured path, as the server does at startup.

    Without this the first question pays for a 16 GB checkpoint load and the
    p95 is a number about start-up that looks like a number about answering.
    """
    from headway import depth as depth_mod
    from headway import detect as detect_mod
    from headway import lanes as lanes_mod
    import vision

    for name, fn in (("lanes", lanes_mod.warm), ("depth", depth_mod.warm),
                     ("detector", detect_mod.warm), ("qwen", vision.warm)):
        t0 = time.time()
        try:
            fn()
            print(f"  warm {name:9s} {(time.time() - t0) * 1000:7.0f} ms")
        except Exception as e:
            print(f"  warm {name:9s} FAILED: {type(e).__name__}: {e}")


def feed(session, ring, frames, start, n, dt):
    for i in range(start, min(start + n, len(frames))):
        rec = session.process(frames[i], 13.0, 0.1, frame_t=i * dt)
        ring.push(frames[i], rec)
    return min(start + n, len(frames))


class Drive:
    """Frames arriving the way they do on a road: continuously, in the
    background, while the driver is talking.

    Feeding in bursts between questions -- which is what this file did first --
    measures a system nobody is using. The ring is trimmed by wall clock and
    the observer only has something to say about a frame that exists, so a
    burst-fed harness reports the fast path as never hitting and the reason is
    the harness. On the road frames do not stop arriving because a question was
    asked.
    """

    def __init__(self, session, ring, frames, dt):
        self.session, self.ring, self.frames, self.dt = session, ring, frames, dt
        self.i = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            t0 = time.time()
            jpeg = self.frames[self.i % len(self.frames)]
            try:
                rec = self.session.process(jpeg, 13.0, 0.1,
                                           frame_t=self.i * self.dt)
                self.ring.push(jpeg, rec)
            except Exception as e:
                print(f"  (drive frame failed: {type(e).__name__}: {e})")
            self.i += 1
            self.stop.wait(max(0.01, self.dt - (time.time() - t0)))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.thread.join(timeout=3)


def run_local(video, n, dt, keep_going):
    print("=" * 78)
    print("LOCAL — one visual turn, broken into its stages")
    print("=" * 78)
    warm()

    frames = load_frames(video, dt, 12 + n * 3)
    if not frames:
        raise SystemExit("no frames decoded")
    headway_live.reset_session(SESSION)
    framebuf.drop_ring(SESSION)
    visual_qa.drop_session(SESSION)
    session = headway_live.get_session(SESSION, use_qwen=config.VISION_ENABLED)
    ring = framebuf.get_ring(SESSION)
    idx = feed(session, ring, frames, 0, 12, dt)
    print(f"  ring primed: {json.dumps(ring.stats())}\n")

    rows = []
    drive = Drive(session, ring, frames[idx:], dt) if keep_going else None
    if drive:
        drive.thread.start()
        # One cadence of frames before the first question, so the observer has
        # had the chance a live session gives it.
        time.sleep(1.2)
    for i in range(n):
        q = QUESTIONS[i % len(QUESTIONS)]
        t0 = time.perf_counter()
        out = realtime.run_tool(realtime.LOOK_TOOL_NAME, {"question": q},
                               session_key=SESSION)
        wall = (time.perf_counter() - t0) * 1000
        meta = out.get("meta") or {}
        timing = dict(meta.get("timing_ms") or {})
        timing["look_total"] = wall
        timing["_q"] = q
        timing["_ok"] = bool(out.get("ok"))
        timing["_answer"] = (out.get("answer") or out.get("note") or "")[:90]
        timing["_fast"] = bool(out.get("fast_path"))
        rows.append(timing)
        print(f"  {i + 1:2d}. {q:38s} {wall:7.0f} ms   "
              f"{'FAST' if out.get('fast_path') else '    '}   "
              f"gpt {timing.get('gpt', 0):6.0f}   "
              f"first_token {timing.get('gpt_first_token', 0):6.0f}   "
              f"enrich {timing.get('enrich', 0):6.0f}")

    if drive:
        drive.stop.set()
        drive.thread.join(timeout=3)

    print("\n  stage breakdown")
    keys = ["route", "select_frame", "resolve", "crop", "enrich",
            "prepare_total", "gpt_first_token", "gpt", "look_total"]
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        print(f"    {k:18s} {fmt(vals)}")

    fast = [r["look_total"] for r in rows if r.get("_fast")]
    slow = [r["look_total"] for r in rows if not r.get("_fast")]
    if fast:
        print(f"\n    {'served from cache':18s} {fmt(fast)}")
        print(f"    {'full turn':18s} {fmt(slow)}")
        print(f"    {len(fast)}/{len(rows)} answered from the running observation")

    local = [sum(r.get(k, 0) or 0 for k in ("route", "select_frame", "resolve",
                                            "crop", "enrich")) for r in rows]
    remote = [r.get("gpt", 0) or 0 for r in rows]
    print(f"\n    {'local stages':18s} {fmt(local)}")
    print(f"    {'remote (openai)':18s} {fmt(remote)}")
    tot = sum(pct(local, 0.5) or 0 for _ in [0]) + (pct(remote, 0.5) or 0)
    if tot:
        print(f"\n    at p50 the answer is {100 * (pct(remote, 0.5) or 0) / tot:.0f}% "
              f"one remote call and {100 * (pct(local, 0.5) or 0) / tot:.0f}% "
              f"everything this pod does")
    return rows


class HttpDrive:
    """A drive, over HTTP, into the SERVER's frame ring.

    The live half is meaningless without one: /realtime/tool answers from the
    ring belonging to a session id, and an empty ring makes every visual
    question come back "nothing to see" -- which measures the round trip and
    nothing else. So this feeds the real endpoint at the real cadence, exactly
    as the dashboard does, and the questions are asked against the same
    session id.
    """

    def __init__(self, base, session_id, video, dt):
        import httpx

        self.base, self.dt = base, dt
        self.frames = load_frames(video, dt, 200)
        self.client = httpx.Client(timeout=30)
        self.stop = threading.Event()
        self.n = 0
        self.rejected = 0
        # A REAL session, started the way the dashboard starts one.
        # /headway_frame refuses frames for a session id the server has never
        # heard of -- correctly: that is the check that stops a tab left open
        # after a restart from streaming into nothing. The first version of
        # this harness invented an id, got 200s that all said `stale`, and
        # measured an empty ring answering "I can't see anything right now" in
        # 674 ms, which is a real answer to a question nobody asked.
        r = self.client.post(f"{base}/session/start",
                             json={"metadata": {"source": "visual_latency"}})
        self.session_id = (r.json() or {}).get("session_id") or session_id
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        i = 0
        while not self.stop.is_set():
            t0 = time.time()
            jpeg = self.frames[i % len(self.frames)]
            try:
                out = self.client.post(
                    f"{self.base}/headway_frame",
                    params={"session_id": self.session_id},
                    files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                    data={"v_host": "13", "v_host_age_s": "0.1",
                          "frame_t": str(i * self.dt)}).json()
                if out.get("ok") is False:
                    self.rejected += 1
                else:
                    self.n += 1
            except Exception as e:
                print(f"  (frame post failed: {type(e).__name__}: {e})")
            i += 1
            self.stop.wait(max(0.01, self.dt - (time.time() - t0)))

    def __enter__(self):
        self.thread.start()
        # A second of frames before the first question, so the observer has had
        # the head start a live session gives it.
        time.sleep(1.5)
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.thread.join(timeout=5)
        try:
            self.client.post(f"{self.base}/session/end",
                             params={"session_id": self.session_id})
        except Exception:
            pass
        self.client.close()


# ---------------------------------------------------------------------------
# THE LIVE PATH, AS IT IS NOW
# ---------------------------------------------------------------------------
# The old version of this measured a session it configured itself with
# `output_modalities: ["audio"]`, which stopped being the path the car uses the
# day the voice moved to ElevenLabs. It reported honest numbers about a
# configuration nobody was running.
#
# This one takes the session exactly as realtime.session_config() builds it —
# text mode, ElevenLabs behind it — feeds a real spoken question in at real
# time so the server's own detector ends the turn, drives frames into the
# SERVER's ring over HTTP at the rate the panel does, and calls the tool over
# HTTP the way the panel does. Everything in the path is the thing itself.
#
# The stages, and why each one is worth its own column:
#
#   turn end -> look()      the model deciding this is a visual question and
#                           asking for the camera. Nothing local can make this
#                           faster; a big number here is the model.
#   look() -> tool result   the server answering. FAST means the running
#                           observer had a sentence ready; a miss means a Qwen
#                           pass, or worse, the full remote visual turn.
#   result -> first token   the model composing the spoken answer from it.
#   token -> first phrase   the chunker deciding enough has arrived to speak.
#   phrase -> first audio   the synthesiser.
#
# The last three are the ones this system owns, and they are the three that
# changed when the voice did.
VISUAL_STAGES = [
    ("turn_end_to_look", "turn end -> look() called"),
    ("look_to_result", "look() -> tool result"),
    ("result_to_token", "tool result -> first text token"),
    ("token_to_phrase", "first token -> first phrase out"),
    ("phrase_to_audio", "first phrase -> FIRST AUDIO"),
    ("felt", "turn end -> FIRST AUDIO (felt)"),
]


def observer_state(base, session_id):
    """What /realtime/status says the observer is doing for this session."""
    import httpx

    try:
        st = httpx.get(f"{base}/realtime/status", timeout=10).json()
        obs = (st.get("observer") or {})
        return obs.get("sessions", {}).get(session_id or "default", {}), obs
    except Exception as e:
        return {"error": f"{type(e).__name__}"}, {}


async def _one_visual_turn(conn, sink, http, base, session_id, question_pcm,
                           marks, tool_url, params):
    """One spoken question, answered, with a timestamp at every seam."""
    import json as _json

    rid = {"id": None}
    feeder = asyncio.create_task(_feed_audio(conn, question_pcm))
    async for event in conn:
        t = event.type
        if t == "input_audio_buffer.speech_stopped":
            marks["turn_end"] = time.perf_counter()
        elif t == "response.created":
            rid["id"] = event.response.id
            if sink:
                await sink.begin(rid["id"])
        elif t == "response.function_call_arguments.done":
            marks.setdefault("look", time.perf_counter())
            marks["tool"] = event.name
            marks["asked"] = (_json.loads(event.arguments or "{}")
                              .get("question") or "")
            r = await http.post(tool_url, params=params, json={
                "name": event.name,
                "arguments": _json.loads(event.arguments or "{}"),
                "where": {"lat": 34.0195, "lng": -118.4912, "age_s": 2},
            })
            res = r.json()
            marks["result"] = time.perf_counter()
            # From here, sound belongs to the ANSWER. Anything before it was
            # the holding line, which the driver hears quickly and which is not
            # what this is measuring.
            marks["phrase_mark"]["armed"] = True
            marks["fast_path"] = bool(res.get("fast_path"))
            marks["path"] = res.get("path")
            marks["on_demand"] = bool(res.get("on_demand"))
            marks["server_ms"] = res.get("took_ms")
            marks["seen_s_ago"] = res.get("seen_s_ago")
            await conn.conversation.item.create(item={
                "type": "function_call_output",
                "call_id": event.call_id, "output": _json.dumps(res)})
            # THE DIRECT PATH, exactly as the panel does it: the observer's own
            # sentence is spoken and no model is asked to compose one. A probe
            # that still called response.create here would be measuring the
            # path that was just removed.
            if res.get("speak_directly") and res.get("speech") and sink:
                await conn.conversation.item.create(item={
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text",
                                 "text": res["speech"]}]})
                marks["first_token"] = time.perf_counter()   # nothing to wait for
                marks["answer"] = res["speech"]
                await sink.begin(rid["id"] + ":direct")
                await sink.delta(rid["id"] + ":direct", res["speech"])
                await sink.end(rid["id"] + ":direct")
                pm = marks["phrase_mark"]
                deadline = time.perf_counter() + 8
                while ("first_audio" not in pm
                       and time.perf_counter() < deadline):
                    await asyncio.sleep(0.02)
                break
            await conn.response.create()
        elif t == "response.output_text.delta":
            # Only AFTER the tool result: the text before it is the holding
            # line ("let me look"), which the driver hears quickly and which is
            # not the answer. Counting it as one is how a slow turn measures
            # fast.
            if marks.get("result") and "first_token" not in marks:
                marks["first_token"] = time.perf_counter()
            marks["answer"] = marks.get("answer", "") + (event.delta or "")
            if sink:
                await sink.delta(rid["id"], event.delta or "")
        elif t == "response.done":
            # The account's per-minute token cap arrives as an ordinary
            # `response.done` with no output. Unhandled it looks exactly like a
            # model that decided to say nothing, and the turn then sits until
            # the timeout — which is what two rows of the first run were.
            pause = _rate_limit_pause(event)
            if pause:
                # The account's per-minute cap. Waiting it out and carrying on
                # would fold the wait into whichever stage is open, so the turn
                # is marked and thrown away instead.
                marks["capped"] = True
                await asyncio.sleep(pause)
                if marks.get("result"):
                    await conn.response.create()
                    continue
                break
            if marks.get("result") and marks.get("first_token"):
                if sink:
                    await sink.end(rid["id"])
                    # The model finishing is not the driver hearing anything.
                    # Breaking here — which the first version did — ends the
                    # turn before the synthesiser has answered, and reports the
                    # one stage this whole exercise is about as missing.
                    pm = marks["phrase_mark"]
                    deadline = time.perf_counter() + 8
                    while ("first_audio" not in pm
                           and time.perf_counter() < deadline):
                        await asyncio.sleep(0.02)
                break
        elif t == "error":
            marks["error"] = str(getattr(event, "error", ""))[:120]
            break
        if time.perf_counter() - marks["t0"] > 60:
            marks["error"] = "timeout"
            break
    feeder.cancel()
    return marks


def _rate_limit_pause(event):
    """Seconds to wait, if this response failed on the realtime token cap."""
    details = getattr(getattr(event, "response", None), "status_details", None)
    if details is None or "rate_limit" not in str(details):
        return None
    m = re.search(r"try again in ([0-9.]+)\s*s", str(details))
    return min(30.0, float(m.group(1)) + 1.0) if m else 5.0


async def _feed_audio(conn, pcm):
    """The driver asking, at the speed a driver asks."""
    import base64 as _b64

    rate, frame_ms = 24000, 20
    payload = pcm + b"\x00\x00" * int(rate * (int(config.REALTIME_VAD_SILENCE_MS) + 400) / 1000)
    step = int(rate * frame_ms / 1000) * 2
    for i in range(0, len(payload), step):
        await conn.input_audio_buffer.append(
            audio=_b64.b64encode(payload[i:i + step]).decode("ascii"))
        await asyncio.sleep(frame_ms / 1000.0)


async def run_live_async(base, n, session_id, drive=None, gap=1.5,
                         reset_every=4):
    import httpx
    from openai import AsyncOpenAI

    import voice_dialogue as vd

    print("\n" + "=" * 78)
    print("LIVE — the current path, end to end, with a seam at every stage")
    print("=" * 78)

    cfg = realtime.session_config()
    text_mode = cfg["output_modalities"] == ["text"]
    print(f"  session modality : {cfg['output_modalities']}")
    print(f"  conversation     : "
          + (f"{config.ELEVENLABS_CONVERSATION_MODEL} "
             f"({type(vd.dialect_for('v', config.ELEVENLABS_CONVERSATION_MODEL, 'p')).__name__})"
             if text_mode else f"cedar ({config.OPENAI_REALTIME_VOICE})"))

    http = httpx.AsyncClient(timeout=90.0)
    tool_url = f"{base}/realtime/tool"
    params = {"session_id": session_id} if session_id else None

    # The question, spoken. One recording per question, so the detector sees
    # the same thing every turn and the turn-end mark means the same thing
    # every turn.
    from tools.voice_latency import driver_audio_for

    rows = []
    # The synthesiser, instrumented. `first phrase out` is the moment the
    # chunker decided a piece of the answer was worth speaking, which is the
    # stage between the model and the sound and the one nothing else can see.
    phrase_mark = {}

    async def on_audio(rid, pcm, text):
        if phrase_mark.get("armed"):
            phrase_mark.setdefault("first_audio", time.perf_counter())

    sink = None
    if text_mode:
        sink = vd.DialogueSession(on_audio=on_audio,
                                  on_event=lambda k, d: asyncio.sleep(0))
        real_speak = sink._speak

        async def timed_speak(utt, phrase):
            if phrase_mark.get("armed"):
                phrase_mark.setdefault("first_phrase", time.perf_counter())
            return await real_speak(utt, phrase)

        sink._speak = timed_speak
        await sink.start()

    # The observer only starts when a live session is minted, which is what
    # the panel does — so the probe does it too, rather than relying on look()
    # to start one cold on the first question.
    minted = await http.post(f"{base}/realtime/session", params=params)
    print(f"  minted           : {minted.status_code} "
          f"(this is what starts the observer)")
    await asyncio.sleep(3)              # let the ring fill and the loop tick

    # ONE CONNECTION FOR THE WHOLE DRIVE, which is both what a car does and
    # what the account can afford. A session per question re-sends the
    # instructions and the whole tool list every time, and eight of those walk
    # straight into the realtime token cap — after which every stage timing is
    # really a measurement of a rate limiter. That is what the first run of
    # this measured, at 12 s a stage, and it is why cap-hit turns are now
    # DISCARDED rather than reported: a number that is mostly somebody else's
    # queue is worse than no number.
    # ONE CONNECTION, BUT NOT FOREVER. History accumulates on a realtime
    # connection and every later turn re-sends all of it, so a long probe walks
    # into the per-minute token cap by growing rather than by asking. Recycling
    # every few turns keeps each turn's context the size a real question has,
    # and `gap` paces the run so the cap is not reached by speed either.
    #
    # Neither is a property of the car — a driver does not ask ten scene
    # questions in ninety seconds — they are what it costs to measure one.
    client = AsyncOpenAI()
    conn_mgr = None
    conn = None
    for i in range(n):
        if conn is None or (reset_every and i and i % reset_every == 0):
            if conn_mgr is not None:
                await conn_mgr.__aexit__(None, None, None)
            conn_mgr = client.realtime.connect(
                model=config.OPENAI_REALTIME_MODEL)
            conn = await conn_mgr.__aenter__()
            await conn.session.update(session=cfg)
        if True:
            q = QUESTIONS[i % len(QUESTIONS)]
            pcm = driver_audio_for(q)
            phrase_mark.clear()
            marks = {"t0": time.perf_counter(), "q": q,
                     "phrase_mark": phrase_mark}
            try:
                await asyncio.wait_for(
                    _one_visual_turn(conn, sink, http, base, session_id,
                                     pcm, marks, tool_url, params),
                    timeout=90)
            except asyncio.TimeoutError:
                marks["error"] = "turn timed out"
            except Exception as e:
                marks["error"] = f"{type(e).__name__}: {str(e)[:100]}"

            obs, _ = observer_state(base, session_id)
            row = _stage_row(marks, phrase_mark, obs)
            rows.append(row)
            _print_row(i + 1, row)
            await asyncio.sleep(gap)
    if conn_mgr is not None:
        await conn_mgr.__aexit__(None, None, None)

    if sink:
        await sink.close()
    await http.aclose()
    _report_stages(rows, base, session_id, drive)
    return rows


def _stage_row(marks, phrase_mark, obs):
    def gap(a, b):
        if marks.get(a) and marks.get(b):
            return (marks[b] - marks[a]) * 1000
        return None

    fa = phrase_mark.get("first_audio")
    fp = phrase_mark.get("first_phrase")
    row = {
        "q": marks["q"], "tool": marks.get("tool"),
        "fast_path": marks.get("fast_path"), "on_demand": marks.get("on_demand"),
        "server_ms": marks.get("server_ms"), "seen_s_ago": marks.get("seen_s_ago"),
        "answer": (marks.get("answer") or "").strip(),
        "asked": marks.get("asked"),
        "path": marks.get("path"),
        "error": marks.get("error"),
        "capped": bool(marks.get("capped")),
        "observations": obs.get("observations"),
        "turn_end_to_look": gap("turn_end", "look"),
        "look_to_result": gap("look", "result"),
        "result_to_token": gap("result", "first_token"),
    }
    row["answer_words"] = len(row["answer"].split())
    if fp and marks.get("first_token"):
        row["token_to_phrase"] = (fp - marks["first_token"]) * 1000
    if fa and fp:
        row["phrase_to_audio"] = (fa - fp) * 1000
    if fa and marks.get("turn_end"):
        row["felt"] = (fa - marks["turn_end"]) * 1000
    return row


def _print_row(i, r):
    def ms(k):
        v = r.get(k)
        return f"{v:6.0f}" if v is not None else "     -"

    flag = {"observer_direct": "SPEAK", "observer_composed": "cache",
            "full_visual": "FULL "}.get(r.get("path"), "     ")
    asked = (r.get("asked") or "")
    rewrite = ""
    if asked and asked.strip().lower() != r["q"].strip().lower():
        rewrite = f"  model asked: {asked[:44]!r}"
    if r.get("capped"):
        flag = "CAP  "
    print(f"  {i:2d}. {r['q'][:26]:26s} {flag} "
          f"look {ms('turn_end_to_look')} "
          f"tool {ms('look_to_result')} "
          f"tok {ms('result_to_token')} "
          f"phr {ms('token_to_phrase')} "
          f"aud {ms('phrase_to_audio')} "
          f"= {ms('felt')} ms  {r['answer_words']}w"
          + (f"  [{r['error']}]" if r.get("error") else "") + rewrite)


def _report_stages(rows, base, session_id, drive=None):
    print("\n  " + "-" * 74)
    capped = [r for r in rows if r.get("capped")]
    rows = [r for r in rows if not r.get("capped")]
    if capped:
        print(f"    ({len(capped)} turn(s) discarded: the realtime token cap "
              "fired, and a stage that contains somebody else's queue is not a "
              "measurement)")
    for key, label in VISUAL_STAGES:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            print(f"    {label:32s} —")
            continue
        mark = "  <<<" if key == "felt" else ""
        print(f"    {label:32s} p50 {pct(vals, 0.5):6.0f}  "
              f"p95 {pct(vals, 0.95):6.0f} ms   n={len(vals)}{mark}")

    # THE TARGET IS ABOUT SCENE QUESTIONS. An object question — "what colour
    # is the car in front" — crops a frame and asks a multimodal model, and it
    # is supposed to take seconds; averaging it in hides the number that is
    # being aimed at behind one that is not.
    scene = [r["felt"] for r in rows
             if r.get("fast_path") and r.get("felt") is not None]
    obj = [r["felt"] for r in rows
           if r.get("tool") and not r.get("fast_path") and r.get("felt") is not None]
    if scene:
        print(f"\n    {'SCENE questions only':32s} p50 {pct(scene, 0.5):6.0f}  "
              f"p95 {pct(scene, 0.95):6.0f} ms   n={len(scene)}")
    if obj:
        print(f"    {'object questions (full turn)':32s} p50 {pct(obj, 0.5):6.0f}  "
              f"p95 {pct(obj, 0.95):6.0f} ms   n={len(obj)}   "
              "— these are meant to be slow")

    cache = sum(1 for r in rows if r.get("path") == "observer_direct")
    demand = sum(1 for r in rows if r.get("path") == "observer_composed")
    full = sum(1 for r in rows if r.get("path") == "full_visual")
    words = [r["answer_words"] for r in rows if r["answer_words"]]
    print(f"\n    spoken from the observer  {cache}/{len(rows)}   "
          "(no model composed it)")
    print(f"    observer, composed by her {demand}/{len(rows)}   "
          "(the line did not pass the lint)")
    print(f"    full visual turns         {full}/{len(rows)}   "
          "(the remote multimodal path)")
    if words:
        print(f"    answer length         p50 {pct(words, 0.5)} words, "
              f"p95 {pct(words, 0.95)}")
    obs, whole = observer_state(base, session_id)
    print(f"    observer for this session: {obs or 'NOT RUNNING'}")
    if drive is not None:
        print(f"    frames posted: {drive.n} ({drive.rejected} refused)")


def run_live(base, n, session_id=None, drive=None, gap=1.5, reset_every=4):
    return asyncio.run(run_live_async(base, n, session_id, drive, gap,
                                      reset_every))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", default="/workspace/ufldv2/example.mp4")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--live", action="store_true",
                    help="also drive a real session (needs the server up)")
    ap.add_argument("--no-local", action="store_true")
    ap.add_argument("--static", action="store_true",
                    help="do not keep feeding frames between questions")
    ap.add_argument("--session", default=None,
                    help="feed this session id with frames over HTTP during the "
                         "live run, and ask the questions against it")
    ap.add_argument("--gap", type=float, default=1.5,
                    help="seconds between questions. Raise it when the "
                         "realtime token cap starts discarding turns — a "
                         "drive does not ask ten questions in a minute.")
    ap.add_argument("--reset-every", type=int, default=4,
                    help="recycle the realtime connection every N turns, so "
                         "accumulated history does not grow each turn's cost")
    ap.add_argument("--out", default=None, help="write the rows as JSON")
    args = ap.parse_args()

    out = {}
    if not args.no_local:
        out["local"] = run_local(args.video, args.n, args.dt, not args.static)
    if args.live:
        if args.session:
            with HttpDrive(args.base, args.session, args.video, args.dt) as drive:
                print(f"  drive session {drive.session_id[:8]} — "
                      f"{drive.n} frames in, {drive.rejected} refused")
                out["live"] = run_live(args.base, args.n, drive.session_id,
                                       drive, args.gap, args.reset_every)
                print(f"    ({drive.n} frames accepted, {drive.rejected} refused)")
        else:
            out["live"] = run_live(args.base, args.n, None, None, args.gap,
                                   args.reset_every)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
