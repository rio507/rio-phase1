"""What the live pipeline needs from a GPU — measured under load, not added up.

    python -m tools.vram_budget
    python -m tools.vram_budget --seconds 30 --json /tmp/vram.json
    python -m tools.vram_budget --no-deep-dive     # skip the one paid call

WHY THIS EXISTS
---------------
"Qwen3-VL-8B is 16 GB of weights, so a 24 GB card is fine" is an addition, not
a measurement, and it is wrong in both directions. It ignores the CUDA context,
the allocator's reserve, RF-DETR and UFLDv2 sharing the card, and the
activations of a generate whose image tokens depend on how big the picture
is -- and it ignores that the allocator never gives memory back, so what a card
must hold is the HIGH-WATER MARK of a whole drive, not the average.

So this drives the real server through the real endpoints, at the real
cadences, and reads the card. Nothing here estimates anything.

WHAT IT DRIVES, one layer at a time, so a number can be attributed
-----------------------------------------------------------------
  resident         the server warm and idle: weights, context, nothing running
  detector         frames at 10 fps through /headway_ws -- RF-DETR and UFLDv2
                   on every frame, which is what a drive does continuously
  + observer       a realtime session minted, so the observer generates a
                   sentence about the road every second (app.py starts it on
                   the mint, never on /session/start -- see
                   tools/frame_contention_probe.py, which learned that the
                   hard way)
  + look           "what do you see?" and "what car is in front of us?"
                   through /realtime/tool: the observer fast path and the full
                   visual turn, which crops and runs a second Qwen pass
  + perceive       the page's /perceive every 15 s
  + deep_dive      a reasoning-model tool call in flight. It is REMOTE and
                   costs no VRAM by itself; it is here because it holds the
                   answer path open (_rio_has_the_gpu) while everything else
                   keeps running, which is the state a real question creates.
  everything       all of the above overlapping. This is the number that sizes
                   the card.

HOW IT MEASURES
---------------
nvidia-smi, every 200 ms, for the DEVICE -- not torch's allocator counters.
Two reasons. The device number includes the CUDA context and any fragmentation
the allocator is sitting on, which is what actually has to fit; and it can be
read from outside the server, so measuring costs the thing being measured
nothing. The run refuses to report if any process other than RIO's uvicorn has
memory on the card, because then the number is not ours.
"""
import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
REPO = Path(__file__).resolve().parent.parent
CLIP = REPO / "runs" / "road_clip.mp4"

import cv2      # noqa: E402
import httpx    # noqa: E402


# ---------------------------------------------------------------------------
# the card
# ---------------------------------------------------------------------------
def smi(query, extra=()):
    out = subprocess.run(
        ["nvidia-smi", f"--query-{query[0]}={query[1]}",
         "--format=csv,noheader,nounits", *extra],
        capture_output=True, text=True, timeout=10)
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def card():
    name, total = smi(("gpu", "name,memory.total"))[0].split(", ")
    return name, int(total)


def compute_apps():
    """[(pid, used_mib)] for every process holding memory on the card.

    The pids are the HOST's, not this container's, so they cannot be matched
    against our own -- which is why this is used to assert that there is
    exactly one, rather than to attribute memory to it.
    """
    rows = []
    for line in smi(("compute-apps", "pid,used_memory")):
        pid, used = line.split(", ")
        rows.append((int(pid), int(used)))
    return rows


class Sampler(threading.Thread):
    """memory.used every `period`, with a timestamp, until stopped."""

    def __init__(self, period=0.2, floor=0):
        super().__init__(daemon=True, name="vram-sampler")
        self.period = period
        # What is on the card that is NOT the server (a ballast, when one is
        # holding the card down to a smaller size). Subtracted at the source so
        # that every number downstream is the server's own.
        self.floor = floor
        self.rows = []          # (t, used_mib)
        # NOT `self._stop`: threading.Thread already has a private _stop()
        # method and shadowing it with an Event makes join() raise
        # "'Event' object is not callable" -- after the measurement, which is
        # the most annoying possible place for it.
        self._done = threading.Event()

    def run(self):
        while not self._done.is_set():
            try:
                used = int(smi(("gpu", "memory.used"))[0]) - self.floor
                self.rows.append((time.time(), used))
            except Exception:
                pass
            self._done.wait(self.period)

    def stop(self):
        self._done.set()

    def window(self, t0, t1):
        return [u for (t, u) in self.rows if t0 <= t <= t1]


def squeeze_to(free_gb):
    """Hold this card down to `free_gb` of free memory and keep it there.

    WHY A BALLAST AND NOT AN ESTIMATE. "The peak was 18.5 GB, so a 24 GB card
    fits" is arithmetic, and it assumes the thing most likely to be false: that
    a smaller card fails only by running out at the peak. It can also fail by
    fragmenting, by refusing a single large contiguous block while plenty is
    free, or by loading the model at all -- and none of those show up in a
    subtraction. So this takes the memory away and runs the drive in what is
    left.

    What it models: a card with this much usable memory.
    What it does not: a slower card, a narrower bus, or a different
    architecture's kernels. Bandwidth is a separate question and this tool does
    not pretend to answer it.
    """
    import torch
    target = int(free_gb * 1024)
    held = []
    print(f"\nsqueezing the card to {free_gb:g} GB free")
    while True:
        used = int(smi(("gpu", "memory.used"))[0])
        free = int(smi(("gpu", "memory.total"))[0]) - used
        if free <= target:
            break
        chunk = min(4096, max(64, free - target))
        try:
            held.append(torch.empty(chunk * 1024 * 1024, dtype=torch.uint8,
                                    device="cuda"))
        except RuntimeError as e:
            print(f"   ballast stopped early: {e}")
            break
    used = int(smi(("gpu", "memory.used"))[0])
    free = int(smi(("gpu", "memory.total"))[0]) - used
    print(f"   {len(held)} blocks held; {free} MiB free — that is the whole "
          f"card as far as\n   the server is concerned (this process's own "
          f"CUDA context is inside the\n   memory it is NOT being given)")
    return held


class Phases:
    def __init__(self, sampler):
        self.sampler = sampler
        self.marks = []     # (name, t0, t1)
        self.server_ms = []

    def mark(self, name, t0, t1):
        self.marks.append((name, t0, t1))

    def table(self):
        out = []
        for name, t0, t1 in self.marks:
            xs = self.sampler.window(t0, t1)
            if not xs:
                continue
            row = {"phase": name, "peak_mib": max(xs),
                   "mean_mib": int(statistics.mean(xs)),
                   "samples": len(xs), "seconds": round(t1 - t0, 1)}
            fr = [ms for (t, ms) in (self.server_ms or []) if t0 <= t <= t1]
            if fr:
                fr.sort()
                row["frame_ms_p50"] = round(fr[len(fr) // 2], 1)
                row["frame_ms_p95"] = round(fr[int(len(fr) * 0.95)], 1)
                row["frames"] = len(fr)
            out.append(row)
        return out


# ---------------------------------------------------------------------------
# the load
# ---------------------------------------------------------------------------
def load_frames(limit=400, max_side=640, quality=70):
    cap = cv2.VideoCapture(str(CLIP))
    out = []
    while len(out) < limit:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = max_side / max(h, w)
        f = cv2.resize(f, (int(w * s), int(h * s)))
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, quality])
        out.append(bytes(buf))
    if not out:
        raise SystemExit(f"no frames from {CLIP}")
    return out


async def wait_warm(base, timeout=300):
    """Warm, not merely answering: a cold server measures the load, not the model."""
    async with httpx.AsyncClient(timeout=20) as c:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                h = (await c.get(f"{base}/health")).json()
                warm = not any(d.get("component") == "warm"
                               for d in (h.get("degraded") or []))
                if warm:
                    return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="length of each driven phase")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-deep-dive", action="store_true",
                    help="skip the one tool call that costs money")
    ap.add_argument("--squeeze", type=float, default=None, metavar="GB",
                    help="hold this card down to GB of free memory, restart "
                         "the server into it, and run the same load. This is "
                         "the difference between 'the peak was 18.5 GB so 24 "
                         "would fit' and having run it in 24.")
    args = ap.parse_args()
    import websockets

    name, total = card()
    print(f"card: {name}, {total} MiB total")

    # Everything this tool reports is the SERVER's memory. Without a ballast
    # that is simply what the card says, because nothing else is on it; with
    # one, it is what the card says minus what the ballast holds -- which is
    # measured with the server stopped, so it is exact rather than assumed.
    ballast, floor = None, 0
    expect_procs = 1
    if args.squeeze:
        # STOP FIRST. The server is holding ~18 GB it will not give back while
        # it lives, so a ballast taken around it would be measuring a card that
        # is already half spent.
        subprocess.run(["bash", str(REPO / "boot.sh"), "stop"],
                       capture_output=True, text=True, timeout=300)
        await asyncio.sleep(3)
        ballast = squeeze_to(args.squeeze)
        floor = int(smi(("gpu", "memory.used"))[0])
        expect_procs = 2
        print(f"   ballast holds {floor} MiB; the server gets "
              f"{total - floor} MiB, and that is its whole world")
        print(f"   starting the server into it")
        r = subprocess.run(["bash", str(REPO / "boot.sh"), "start"],
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f"\n!! THE SERVER DID NOT COME UP IN "
                  f"{args.squeeze:g} GB — that is the answer for this size:")
            for line in (r.stdout or "").splitlines()[-25:]:
                print("     " + line)
            return 1

    apps = compute_apps()
    print(f"processes on the card: {len(apps)}  {apps}")
    if len(apps) > expect_procs:
        print("\n!! more processes have memory on this card than this run "
              "accounts for.\n!! Every number below would include them. If "
              "they are the teacher\n!! services: bash boot.sh teachers-stop")
        return 2

    if not await wait_warm(args.base):
        print("!! the server never reported warm"
              + (f" in {args.squeeze:g} GB — it came up and could not load "
                 f"the models" if args.squeeze else ""))
        return 1

    # WHAT THE SERVER SAID WHILE THIS RAN. Byte offset now, tail afterwards:
    # an OOM in a worker thread is caught, logged and swallowed by design (the
    # drive continues), so the only place the difference between "slower" and
    # "broken" is written down is this file.
    log_path = REPO / "uvicorn.log"
    log_at = log_path.stat().st_size if log_path.exists() else 0

    frames = load_frames()
    sampler = Sampler(floor=floor)
    sampler.start()
    ph = Phases(sampler)
    base = args.base

    # --- resident ----------------------------------------------------------
    print("\nphase: resident (warm, nothing driven)")
    t0 = time.time()
    await asyncio.sleep(10)
    ph.mark("resident", t0, time.time())
    resident = max(sampler.window(t0, time.time()))
    print(f"   {resident} MiB")

    # --- the session the page builds ---------------------------------------
    async with httpx.AsyncClient(timeout=60) as c:
        sid = (await c.post(f"{base}/session/start",
                            json={"metadata": {"source": "vram_budget"}})).json()["session_id"]
    print(f"session {sid}")

    url = base.replace("http://", "ws://") + f"/headway_ws?session_id={sid}&client_id=vram"
    stop_pump = asyncio.Event()
    sent = {"n": 0, "results": 0, "server_ms": []}

    async def pump(ws):
        """Frames at --fps, in the transport's own wire shape.

        uint32be header length, UTF-8 JSON header, then the JPEG -- see
        app._parse_ws_frame. A raw JPEG is not a malformed frame to this
        server, it is a DISCARDED one: the parser returns None and the receive
        loop continues, so a probe that sends bare bytes streams happily,
        reports every frame sent, and measures a pipeline that never ran. That
        is how the first run of this tool measured an idle card and called it a
        drive.
        """
        i = 0
        period = 1.0 / args.fps
        while not stop_pump.is_set():
            t = time.perf_counter()
            head = json.dumps({"seq": i + 1, "cap_t": time.time(),
                               "v": 12.0, "va": 0.1, "src": "vram_budget"}).encode()
            try:
                await ws.send(len(head).to_bytes(4, "big") + head
                              + frames[i % len(frames)])
                sent["n"] += 1
            except Exception:
                return
            i += 1
            await asyncio.sleep(max(0.0, period - (time.perf_counter() - t)))

    async def reader(ws):
        """Count only what the pipeline actually produced.

        `op` messages -- warming, skip, stale, error, pong -- are not results,
        and counting them would hide exactly the failure this loop exists to
        catch.
        """
        while not stop_pump.is_set():
            try:
                m = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                return
            try:
                j = json.loads(m)
            except Exception:
                continue
            if j.get("op"):
                sent.setdefault("ops", {})
                sent["ops"][j["op"]] = sent["ops"].get(j["op"], 0) + 1
                continue
            sent["results"] += 1
            # SERVER TIME PER FRAME, kept per phase. The question "does a
            # smaller card break loudly or quietly" cannot be answered by a
            # memory number alone: quietly means these get longer while
            # everything still reports success.
            ms = j.get("server_ms")
            if isinstance(ms, (int, float)):
                sent["server_ms"].append((time.time(), float(ms)))

    async def tool(cl, payload, label):
        t = time.time()
        try:
            r = await cl.post(f"{base}/realtime/tool?session_id={sid}", json=payload)
            j = r.json()
            print(f"   {label}: {time.time() - t:.1f} s  path={j.get('path')}")
            return j
        except Exception as e:
            print(f"   {label}: {type(e).__name__}: {e}")
            return {}

    async def perceive(cl, n):
        """/perceive, and SAY WHETHER IT DID ANYTHING.

        This endpoint answers instantly and successfully when it decides not to
        work -- `skipped: warming`, or a frame it judges stale -- so a probe
        that only timed it reported a fast /perceive and measured nothing. If
        it skipped, the phase it is in is not the phase it claims to be.
        """
        t = time.time()
        try:
            r = await cl.post(f"{base}/perceive?session_id={sid}",
                              files={"image": ("f.jpg",
                                               frames[(n * 37) % len(frames)],
                                               "image/jpeg")})
            j = r.json() if r.status_code == 200 else {}
            skipped = j.get("skipped")
            tm = (j.get("timing_ms") or {}).get("total")
            print(f"   perceive: {time.time() - t:.1f} s  "
                  + (f"SKIPPED ({skipped}) — no Qwen pass ran" if skipped
                     else f"qwen total={tm} ms"))
        except Exception as e:
            print(f"   perceive: {type(e).__name__}: {e}")

    async with websockets.connect(url, max_size=None) as ws:
        await ws.recv()                       # the ready frame
        pumper = asyncio.create_task(pump(ws))
        rdr = asyncio.create_task(reader(ws))

        # --- detector only -------------------------------------------------
        print(f"\nphase: detector — frames at {args.fps:g} fps "
              f"(RF-DETR + UFLDv2 per frame)")
        t0 = time.time()
        await asyncio.sleep(args.seconds)
        ph.mark("frames (detector + lanes)", t0, time.time())

        # --- the observer --------------------------------------------------
        print("\nphase: + observer at 1 Hz (the realtime mint starts it)")
        async with httpx.AsyncClient(timeout=90) as c:
            await c.post(f"{base}/realtime/session?session_id={sid}&client_id=vram")
        t0 = time.time()
        await asyncio.sleep(args.seconds)
        ph.mark("+ observer (Qwen generate, 1 Hz)", t0, time.time())

        async with httpx.AsyncClient(timeout=180) as cl:
            # --- look ------------------------------------------------------
            print("\nphase: + look() — the observer fast path, then a full "
                  "visual turn")
            t0 = time.time()
            await tool(cl, {"name": "look",
                            "arguments": {"question": "What do you see?"},
                            "spoken": "What do you see?"}, "look(scene)")
            await tool(cl, {"name": "look",
                            "arguments": {"question": "What car is in front of us?"},
                            "spoken": "What car is in front of us?"},
                       "look(object)")
            ph.mark("+ look (crop + second Qwen pass)", t0, time.time())

            # --- perceive --------------------------------------------------
            print("\nphase: + /perceive")
            t0 = time.time()
            await perceive(cl, 1)
            ph.mark("+ perceive", t0, time.time())

            # --- everything at once ---------------------------------------
            print("\nphase: EVERYTHING — frames, observer, look, perceive"
                  + ("" if args.no_deep_dive else ", deep_dive") + " overlapping")
            t0 = time.time()
            jobs = [
                tool(cl, {"name": "look",
                          "arguments": {"question": "What do you see?"},
                          "spoken": "What do you see?"}, "look(scene)"),
                tool(cl, {"name": "look",
                          "arguments": {"question": "What is the car ahead doing?"},
                          "spoken": "What is the car ahead doing?"}, "look(object)"),
                perceive(cl, 2),
            ]
            if not args.no_deep_dive:
                jobs.append(tool(cl, {"name": "deep_dive", "arguments": {
                    "question": "What is the speed limit on a UK dual carriageway?"},
                    "spoken": "What is the speed limit on a dual carriageway?"},
                    "deep_dive"))
            await asyncio.gather(*jobs)
            # Let the tail of the last generate land before closing the window.
            await asyncio.sleep(5)
            ph.mark("EVERYTHING at once", t0, time.time())

        stop_pump.set()
        pumper.cancel()
        rdr.cancel()

    sampler.stop()
    sampler.join(timeout=2)

    # --- the answer --------------------------------------------------------
    ph.server_ms = sent["server_ms"]
    rows = ph.table()
    peak = max(r["peak_mib"] for r in rows)
    print("\n" + "=" * 72)
    print(f"{'phase':<34} {'peak MiB':>9} {'mean':>7} {'+resident':>10} "
          f"{'frame p50/p95 ms':>18}")
    for r in rows:
        lat = (f"{r['frame_ms_p50']}/{r['frame_ms_p95']}"
               if r.get("frame_ms_p50") is not None else "-")
        print(f"{r['phase']:<34} {r['peak_mib']:>9} {r['mean_mib']:>7} "
              f"{r['peak_mib'] - resident:>+10} {lat:>18}")
    print("-" * 72)
    print(f"{'PEAK, whole run':<36} {peak:>9}")
    print(f"frames sent {sent['n']}, results {sent['results']}"
          + (f", ops {sent['ops']}" if sent.get("ops") else ""))
    if sent["results"] < 10:
        print("\n!! ONLY %d RESULTS CAME BACK. The detector and the lane model"
              "\n!! did not run, so every number above is an idle card wearing"
              "\n!! a drive's name. Not reporting a budget from it."
              % sent["results"])
        return 2

    if args.squeeze:
        budget = total - floor
        print(f"\nTHIS RAN IN {budget} MiB ({args.squeeze:g} GB), not next to "
              f"it: the card had\nnothing else to give. Peak {peak} MiB, "
              f"{budget - peak} MiB spare ({100.0 * peak / budget:.0f}% used).")

    print("\nwhat that means for a card, with the CUDA context and the "
          "allocator's\nreserve included, because they are in the number above:")
    for cap_gb in (24, 32, 40, 48, 80):
        cap = cap_gb * 1024
        head = cap - peak
        verdict = "FITS" if head > 0 else "DOES NOT FIT"
        pct = 100.0 * peak / cap
        print(f"   {cap_gb:>3} GB card: {verdict:<12} "
              f"{head:>+7} MiB headroom   ({pct:.0f}% used)")
    print("=" * 72)

    # --- did anything break, or did it just get slower? --------------------
    trouble = []
    try:
        with open(log_path, "rb") as f:
            f.seek(log_at)
            tail = f.read().decode("utf-8", "replace")
        for line in tail.splitlines():
            low = line.lower()
            if ("out of memory" in low or "outofmemory" in low
                    or "cuda error" in low or "cublas" in low
                    or "traceback" in low or "no kernel image" in low):
                trouble.append(line[:160])
    except Exception:
        pass
    print("\nwhat the server logged while this ran:")
    if trouble:
        print(f"   {len(trouble)} line(s) that matter:")
        for t in trouble[:12]:
            print(f"     {t}")
    else:
        print("   nothing about memory, kernels or exceptions")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"card": name, "total_mib": total, "resident_mib": resident,
             "peak_mib": peak, "phases": rows, "squeeze_gb": args.squeeze,
             "frames_sent": sent["n"], "results": sent["results"],
             "trouble": trouble}, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
