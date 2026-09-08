#!/usr/bin/env python3
"""frame_transport_bench.py — how old is the picture the detector runs on?

    python3 tools/frame_transport_bench.py --seconds 40

WHY THIS EXISTS AT ALL. The first real drive (session 06af3214, iPhone,
607.8 s, 1949 frames) could not answer that question about a single one of its
frames. The phone sent no capture timestamp, so the server knew only when a
frame ARRIVED, and "frame age at detection" -- the number that decides whether
a gap warning is about the road the car is on or the road it was on -- was not
a measurement anybody had. What the log did have:

    inter-frame arrival    p50 258 ms   p90 406 ms   p99 1503 ms
    server processing      p50  23.6 ms  p90 33.5 ms
    frame bytes            71-90 KB (from the /perceive events of the same feed)

The middle line says the server was never the bottleneck. The other two are
what this bench reproduces.

THE LINK IS DERIVED FROM THAT LOG, NOT INVENTED. A frame of ~80 KB whose round
trip reached 406 ms at p90, of which 24 ms was the server, leaves ~322 ms for
80 KB of uplink plus a base round trip -- about 2.0 Mbit/s up with ~60 ms of
RTT, which is an ordinary LTE uplink with a voice session already on it. Those
are the defaults below, and --uplink-mbit / --rtt-ms move them.

Both transports are driven from the SAME frames of the same clip against the
SAME running server, and both are shaped by the same link, so the difference
in the table is the transport and nothing else.

    before   the shipped POST path: one multipart request per frame, camera
             resolution, JPEG q0.8, 250 ms cadence, next frame after the last
             result lands.
    after    the socket: one connection, 640 px long side, adaptive quality,
             adaptive 8-15 fps, and one frame in flight with the newest
             evicting the waiting one.

Run it with the server up (boot.sh leaves one on :8888). It needs no camera and
no phone: --clip is any road video, and runs/road_clip.mp4 is the default.
"""
import argparse
import asyncio
import io
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_CLIP = REPO / "runs" / "road_clip.mp4"


# ---------------------------------------------------------------------------
# The link. Shaping happens in the client, which is the honest place for it:
# the server is the thing under test and must not know it is on a bad radio.
# ---------------------------------------------------------------------------
class Link:
    """A one-way uplink with a bandwidth and a latency, and a return path.

    Serialisation is modelled as a shared resource: two frames cannot be on
    the radio at once, which is exactly the property that makes queueing
    frames fatal and is the whole reason the socket path drops them.
    """

    def __init__(self, uplink_mbit: float, rtt_ms: float):
        self.bps = uplink_mbit * 1e6
        self.owd = (rtt_ms / 1000.0) / 2.0
        self._free_at = 0.0
        self._lock = asyncio.Lock()

    async def send(self, nbytes: int):
        """Block until `nbytes` have been clocked out, then wait out latency."""
        async with self._lock:
            now = time.perf_counter()
            start = max(now, self._free_at)
            serialise = nbytes * 8 / self.bps
            self._free_at = start + serialise
            wait = self._free_at - now
        if wait > 0:
            await asyncio.sleep(wait)
        await asyncio.sleep(self.owd)

    async def receive(self, nbytes: int = 800):
        # The downlink is not the constrained direction on a phone; a result is
        # a few hundred bytes and only the latency is worth modelling.
        await asyncio.sleep(self.owd)


class Unshaped(Link):
    def __init__(self):
        super().__init__(uplink_mbit=1e6, rtt_ms=0.0)

    async def send(self, nbytes: int):
        return

    async def receive(self, nbytes: int = 800):
        return


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def load_frames(clip: Path, limit: int):
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while len(frames) < limit:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {clip}")
    return frames


def encode(frame, max_side: int | None, quality: float) -> bytes:
    img = frame
    if max_side:
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (max(2, int(w * scale)), max(2, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY),
                                         int(round(quality * 100))])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def start_session(base, label):
    """A real session, because a bench that skips one is not on the real path.

    /headway_frame refuses a frame tagged with a session the process has never
    heard of, and both transports go through the session's frame ring, its
    JSONL and its reaper. Measuring without one would measure a shorter path
    than the drive takes.
    """
    import httpx
    r = httpx.post(f"{base}/session/start",
                   json={"metadata": {"source": f"transport-bench:{label}"}},
                   timeout=15.0)
    return r.json()["session_id"]


def end_session(base, sid):
    import httpx
    try:
        httpx.post(f"{base}/session/end?session_id={sid}", timeout=15.0)
    except Exception:
        pass


def pct(values, p):
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round((len(v) - 1) * p)))], 1)


def summarise(name, ages, sent, results, dropped, byte_list, wall):
    return {
        "transport": name,
        "frames_sent": sent,
        "results": results,
        "dropped_or_skipped": dropped,
        "fps": round(results / wall, 2) if wall else None,
        "mean_frame_kb": round(statistics.mean(byte_list) / 1024, 1) if byte_list else None,
        "uplink_mbit": round(sum(byte_list) * 8 / wall / 1e6, 2) if wall and byte_list else None,
        "frame_age_ms": {"p50": pct(ages, 0.5), "p90": pct(ages, 0.9),
                         "p99": pct(ages, 0.99), "max": pct(ages, 1.0),
                         "n": len(ages)},
    }


# ---------------------------------------------------------------------------
# BEFORE: the POST path, exactly as the page drove it on the first real drive
# ---------------------------------------------------------------------------
async def run_post(base, session, frames, link, seconds, cadence_ms=250,
                   quality=0.8, max_side=None):
    import httpx  # local: only this path needs an async HTTP client

    ages, byte_list = [], []
    sent = results = skipped = 0
    t_end = time.perf_counter() + seconds
    i = 0
    url = f"{base}/headway_frame?session_id={session}"
    async with httpx.AsyncClient(timeout=30.0) as http:
        while time.perf_counter() < t_end:
            t_tick = time.perf_counter()
            jpeg = encode(frames[i % len(frames)], max_side, quality)
            i += 1
            cap_t = time.time()
            await link.send(len(jpeg))
            files = {"image": ("frame.jpg", jpeg, "image/jpeg")}
            data = {"v_host": "18.0", "v_host_age_s": "0.1",
                    "cap_t": str(cap_t), "source": "clip"}
            try:
                r = await http.post(url, files=files, data=data)
                j = r.json()
            except Exception:
                skipped += 1
                continue
            await link.receive()
            sent += 1
            byte_list.append(len(jpeg))
            if j.get("ok") is False:
                skipped += 1
                continue
            results += 1
            # Measured here rather than trusted from the server: on the POST
            # path the browser waits for the response, so the age the DRIVER
            # experiences includes the reply coming back.
            ages.append((time.time() - cap_t) * 1000.0)
            spent = time.perf_counter() - t_tick
            await asyncio.sleep(max(0.0, cadence_ms / 1000.0 - spent))
    return ages, sent, results, skipped, byte_list


# ---------------------------------------------------------------------------
# AFTER: the socket, with the page's own pacing and drop rule
# ---------------------------------------------------------------------------
async def run_ws(base, session, frames, link, seconds, tuning_override=None):
    import websockets

    url = base.replace("http://", "ws://").replace("https://", "wss://")
    url = f"{url}/headway_ws?session_id={session}"

    ages, byte_list = [], []
    state = {"sent": 0, "results": 0, "skipped": 0, "dropped": 0,
             "inflight": 0, "fps": 10.0, "quality": 0.62, "tuning": None,
             "stop": False, "max_inflight": 1, "age_ema": None, "starved": 0}

    async with websockets.connect(url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        tuning = ready.get("tuning") or {}
        tuning.update(tuning_override or {})
        state["tuning"] = tuning
        state["fps"] = tuning.get("start_fps", 10.0)
        state["quality"] = tuning.get("quality", 0.62)

        async def reader():
            while not state["stop"]:
                try:
                    msg = json.loads(await ws.recv())
                except Exception:
                    return
                op = msg.get("op")
                if op in ("skip", "error"):
                    state["inflight"] = max(0, state["inflight"] - 1)
                    state["skipped"] += 1
                    continue
                if op:
                    continue
                await link.receive()
                state["inflight"] = max(0, state["inflight"] - 1)
                state["results"] += 1
                state["dropped"] = msg.get("dropped") or state["dropped"]
                age = msg.get("frame_age_ms")
                if age is not None:
                    ages.append(age)
                    # The same controller rio_frames.js runs, in the same
                    # direction and with the same numbers: rate answers to
                    # frame age, quality answers to the byte budget.
                    ema = state["age_ema"]
                    state["age_ema"] = age if ema is None else ema * 0.8 + age * 0.2
                    ema = state["age_ema"]
                    if age > tuning["max_age_ms"] or ema > tuning["max_age_ms"]:
                        state["max_inflight"] = 1
                        state["starved"] = 0
                        state["fps"] = max(tuning["min_fps"], state["fps"] * 0.8)
                        if state["fps"] <= tuning["min_fps"] + 0.01:
                            state["quality"] = max(tuning["quality_min"],
                                                   state["quality"] - 0.06)
                    elif ema < tuning["target_age_ms"]:
                        state["fps"] = min(tuning["max_fps"], state["fps"] + 0.5)
                        if (state["starved"] >= 2
                                and ema < tuning["target_age_ms"] * 0.75):
                            state["max_inflight"] = min(
                                tuning.get("max_inflight", 1),
                                state["max_inflight"] + 1)
                            state["starved"] = 0

        rtask = asyncio.create_task(reader())
        t_end = time.perf_counter() + seconds
        i = 0
        while time.perf_counter() < t_end:
            t_tick = time.perf_counter()
            # DROP, NEVER QUEUE. No picture is taken at all while one is out.
            if state["inflight"] >= state["max_inflight"]:
                state["skipped"] += 1
                state["starved"] += 1
            else:
                jpeg = encode(frames[i % len(frames)],
                              int(tuning["max_side_px"]), state["quality"])
                i += 1
                over = len(jpeg) / tuning["target_bytes"]
                if over > 1.25:
                    state["quality"] = max(tuning["quality_min"], state["quality"] - 0.04)
                elif over < 0.75:
                    state["quality"] = min(tuning["quality_max"], state["quality"] + 0.02)
                cap_t = time.time()
                head = json.dumps({"seq": state["sent"], "cap_t": cap_t,
                                   "v": 18.0, "va": 0.1, "src": "clip"}).encode()
                payload = len(head).to_bytes(4, "big") + head + jpeg
                await link.send(len(payload))
                state["inflight"] += 1
                state["sent"] += 1
                byte_list.append(len(jpeg))
                try:
                    await ws.send(payload)
                except Exception:
                    break
            spent = time.perf_counter() - t_tick
            await asyncio.sleep(max(0.0, 1.0 / state["fps"] - spent))

        # Let the last frames land before the socket goes.
        await asyncio.sleep(0.6)
        state["stop"] = True
        rtask.cancel()

    return (ages, state["sent"], state["results"],
            state["skipped"] + state["dropped"], byte_list, state)


# ---------------------------------------------------------------------------
# THE INVARIANT, asserted rather than described.
#
# Everything above measures. This checks the one property the transport is not
# allowed to lose: a client that ignores the drop rule entirely and floods the
# socket must NOT be able to make frame age grow. The server's single slot is
# what makes that true -- a newer frame evicts the waiting one -- and a
# regression that turned that slot into a list would show up here as an age
# that climbs with every frame sent, and nowhere else until a drive.
# ---------------------------------------------------------------------------
async def burst_invariant(base, frames, n=90, fps=60.0):
    import websockets

    sid = start_session(base, "burst")
    url = (base.replace("http://", "ws://").replace("https://", "wss://")
           + f"/headway_ws?session_id={sid}")
    ages, order = [], []
    async with websockets.connect(url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        tuning = ready.get("tuning") or {}
        stop = {"v": False}

        async def reader():
            while not stop["v"]:
                try:
                    msg = json.loads(await ws.recv())
                except Exception:
                    return
                if msg.get("op"):
                    continue
                if msg.get("frame_age_ms") is not None:
                    ages.append(msg["frame_age_ms"])
                    order.append(msg.get("seq"))

        rt = asyncio.create_task(reader())
        for i in range(n):
            jpeg = encode(frames[i % len(frames)], int(tuning["max_side_px"]), 0.62)
            head = json.dumps({"seq": i, "cap_t": time.time(), "v": 18.0,
                               "va": 0.1, "src": "clip"}).encode()
            await ws.send(len(head).to_bytes(4, "big") + head + jpeg)
            await asyncio.sleep(1.0 / fps)
        await asyncio.sleep(1.0)
        stop["v"] = True
        rt.cancel()
    end_session(base, sid)

    fails = []
    if not ages:
        fails.append("no frame ages came back at all")
    else:
        first, last = ages[: max(1, len(ages) // 3)], ages[-max(1, len(ages) // 3):]
        f50, l50 = pct(first, 0.5), pct(last, 0.5)
        # Age must not TREND upward under a flood. A little jitter is fine; a
        # queue shows up as the last third being several times the first.
        if l50 > max(120.0, f50 * 2.0):
            fails.append(f"frame age grew under flood: {f50} -> {l50} ms "
                         f"(a queue formed somewhere)")
        if pct(ages, 1.0) > 1500:
            fails.append(f"worst frame age {pct(ages, 1.0)} ms under flood")
        if len(ages) >= n:
            fails.append(f"{len(ages)} results for {n} frames flooded in: "
                         f"nothing was dropped, so nothing was protecting the age")
        if order != sorted(order):
            fails.append("results came back out of order")
    return {"sent": n, "results": len(ages),
            "age_p50": pct(ages, 0.5), "age_max": pct(ages, 1.0),
            "failures": fails}


# ---------------------------------------------------------------------------
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--clip", default=str(DEFAULT_CLIP))
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--frames", type=int, default=120)
    # Derived from the first real drive's log -- see the module docstring.
    ap.add_argument("--uplink-mbit", type=float, default=2.0)
    ap.add_argument("--rtt-ms", type=float, default=60.0)
    ap.add_argument("--unshaped", action="store_true",
                    help="also run both transports with no link model at all")
    ap.add_argument("--burst", action="store_true",
                    help="only assert the drop invariant; exits non-zero on failure")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    frames = load_frames(Path(args.clip), args.frames)
    print(f"clip: {args.clip}  ({len(frames)} frames, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]})")
    print(f"link: {args.uplink_mbit} Mbit/s up, {args.rtt_ms} ms RTT "
          f"(derived from session 06af3214)")
    print()

    out = {"link": {"uplink_mbit": args.uplink_mbit, "rtt_ms": args.rtt_ms},
           "runs": []}

    if args.burst:
        r = await burst_invariant(args.base, frames)
        print(json.dumps(r, indent=2))
        if r["failures"]:
            for f in r["failures"]:
                print("  FAIL  " + f)
            sys.exit(1)
        print("  ok    a flooded socket drops frames instead of ageing them "
              f"({r['results']}/{r['sent']} processed, p50 {r['age_p50']} ms, "
              f"worst {r['age_max']} ms)")
        return

    async def one(label, shaped):
        link = Link(args.uplink_mbit, args.rtt_ms) if shaped else Unshaped()

        print(f"--- {label}: POST (before) ---", flush=True)
        t0 = time.perf_counter()
        ages, sent, res, skipped, blist = await run_post(
            args.base, start_session(args.base, f"{label}-post"),
            frames, link, args.seconds)
        before = summarise("post (shipped)", ages, sent, res, skipped, blist,
                           time.perf_counter() - t0)
        before["link"] = label
        print(json.dumps(before, indent=2), flush=True)

        link = Link(args.uplink_mbit, args.rtt_ms) if shaped else Unshaped()
        print(f"--- {label}: WebSocket (after) ---", flush=True)
        t0 = time.perf_counter()
        ages, sent, res, dropped, blist, st = await run_ws(
            args.base, start_session(args.base, f"{label}-ws"),
            frames, link, args.seconds)
        after = summarise("websocket", ages, sent, res, dropped, blist,
                          time.perf_counter() - t0)
        after["link"] = label
        after["final_fps"] = round(st["fps"], 1)
        after["final_quality"] = round(st["quality"], 2)
        after["final_inflight"] = st["max_inflight"]
        print(json.dumps(after, indent=2), flush=True)
        out["runs"].extend([before, after])
        return before, after

    before, after = await one("mobile", True)
    if args.unshaped:
        await one("unshaped", False)

    def imp(b, a):
        if not b or not a:
            return "--"
        return f"{b:.0f} -> {a:.0f} ms  ({(1 - a / b) * 100:.0f}% lower)"

    print("\n=== frame age at detection, mobile link ===")
    for k in ("p50", "p90", "p99"):
        print(f"  {k}: {imp(before['frame_age_ms'][k], after['frame_age_ms'][k])}")
    print(f"  fps: {before['fps']} -> {after['fps']}")
    print(f"  uplink: {before['uplink_mbit']} -> {after['uplink_mbit']} Mbit/s")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
