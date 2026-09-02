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
import glob
import json
import os
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


def run_live(base, n, session_id=None):
    """Ten visual questions through a real session, timed as the driver feels it.

    Reports which tools were called -- deep_dive on a visual question is a
    misroute and doubles the wait -- and the gap between the question landing
    and the first audio delta, which is the felt latency.
    """
    import httpx
    from openai import OpenAI

    from tools.realtime_selftest import _rate_limit_wait as rate_limit_wait

    print("\n" + "=" * 78)
    print("LIVE — a real session, visual questions, felt latency")
    print("=" * 78)

    client = OpenAI()
    cfg = realtime.session_config()
    rows = []
    tool_url = f"{base}/realtime/tool"
    params = {"session_id": session_id} if session_id else None
    with client.realtime.connect(model=config.OPENAI_REALTIME_MODEL) as conn:
        conn.session.update(session={
            "type": "realtime",
            "instructions": cfg["instructions"],
            "tools": cfg["tools"],
            "output_modalities": ["audio"],
            "audio": {"output": {"voice": config.OPENAI_REALTIME_VOICE,
                                 "format": {"type": "audio/pcm", "rate": 24000}}},
        })
        for i in range(n):
            q = QUESTIONS[i % len(QUESTIONS)]
            conn.conversation.item.create(item={
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": q}]})
            conn.response.create()
            t_ask = time.perf_counter()
            row = {"q": q, "tools": [], "tool_ms": None, "first_audio_ms": None,
                   "answer_audio_ms": None, "took_ms": None}
            need_followup, active, started, waited = False, 0, 0, 0
            # Audio arriving BEFORE the tool result is the holding line -- "let
            # me look" -- which is not the answer and must not be counted as
            # one. The driver hears it quickly and then waits, and it is that
            # second wait this whole exercise is about.
            tool_done = False
            for event in conn:
                if time.perf_counter() - t_ask > 90:
                    break
                if event.type == "response.function_call_arguments.done":
                    row["tools"].append(event.name)
                    t_tool = time.perf_counter()
                    r = httpx.post(tool_url, timeout=90, params=params, json={
                        "name": event.name,
                        "arguments": json.loads(event.arguments or "{}"),
                        "where": {"lat": 34.0195, "lng": -118.4912, "age_s": 2},
                    }).json()
                    row["fast"] = bool(r.get("fast_path"))
                    row["tool_ms"] = (time.perf_counter() - t_tool) * 1000
                    row["took_ms"] = r.get("took_ms")
                    conn.conversation.item.create(item={
                        "type": "function_call_output",
                        "call_id": event.call_id, "output": json.dumps(r)})
                    need_followup = True
                    tool_done = True
                elif event.type == "response.output_audio.delta":
                    now = (time.perf_counter() - t_ask) * 1000
                    if row["first_audio_ms"] is None:
                        row["first_audio_ms"] = now
                    if tool_done and row["answer_audio_ms"] is None:
                        row["answer_audio_ms"] = now
                elif event.type == "response.created":
                    started += 1
                    active += 1
                elif event.type == "response.done":
                    active -= 1
                    # The realtime token cap. Without this the response comes
                    # back empty, the loop breaks, and every LATER question
                    # returns instantly with nothing because there is an unread
                    # `done` sitting in the stream -- which is exactly what the
                    # first version of this file measured: two answers and eight
                    # blanks. tools/realtime_selftest.py hit the same wall and
                    # this is its handling, imported rather than re-derived.
                    pause = rate_limit_wait(event)
                    if pause is not None and waited < 4:
                        waited += 1
                        print(f"    (token cap — waiting {pause:.0f}s)")
                        time.sleep(pause)
                        conn.response.create()
                        continue
                    if need_followup:
                        need_followup = False
                        conn.response.create()
                        continue
                    if started and active <= 0:
                        break
                elif event.type == "error":
                    row["error"] = str(getattr(event, "error", ""))
                    break
            rows.append(row)
            print(f"  {i + 1:2d}. {q:30s} {'FAST' if row.get('fast') else '    '} "
                  f"{','.join(row['tools']) or '-':<12s} "
                  f"tool_rt {row['tool_ms'] or 0:6.0f}  "
                  f"server {row['took_ms'] or 0:6.0f}  "
                  f"holding {row['first_audio_ms'] or 0:6.0f}  "
                  f"ANSWER {row['answer_audio_ms'] or 0:7.0f} ms")

    # THE number: ask -> the first word of the ANSWER, past any holding line.
    felt = [r["answer_audio_ms"] for r in rows if r["answer_audio_ms"]]
    fast_felt = [r["answer_audio_ms"] for r in rows
                 if r.get("fast") and r["answer_audio_ms"]]
    slow_felt = [r["answer_audio_ms"] for r in rows
                 if not r.get("fast") and r["answer_audio_ms"]]
    holding = [r["first_audio_ms"] for r in rows if r["first_audio_ms"]]
    rt = [r["tool_ms"] for r in rows if r["tool_ms"]]
    srv = [r["took_ms"] for r in rows if r["took_ms"]]
    deep = sum(1 for r in rows if realtime.TOOL_NAME in r["tools"])
    looked = sum(1 for r in rows if realtime.LOOK_TOOL_NAME in r["tools"])
    print(f"\n    {'felt (ask -> ANSWER)':22s} {fmt(felt)}")
    print(f"    {'  ...served from cache':22s} {fmt(fast_felt)}")
    print(f"    {'  ...full turn':22s} {fmt(slow_felt)}")
    print(f"    {'first sound at all':22s} {fmt(holding)}   "
          f"(the holding line, not the answer)")
    print(f"    {'tool round trip':22s} {fmt(rt)}")
    print(f"    {'of which server':22s} {fmt(srv)}")
    print(f"\n    look called on {looked}/{len(rows)} visual questions")
    print(f"    deep_dive called on {deep}/{len(rows)} — "
          + ("MISROUTE: a visual question must never go there" if deep
             else "none, which is correct"))
    return rows


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
                out["live"] = run_live(args.base, args.n, drive.session_id)
                print(f"    ({drive.n} frames accepted, {drive.rejected} refused)")
        else:
            out["live"] = run_live(args.base, args.n)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
