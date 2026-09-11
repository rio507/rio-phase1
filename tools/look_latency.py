"""How long "what do you see?" takes, per stage, and where it goes.

    python -m tools.look_latency
    python -m tools.look_latency --n 12
    python -m tools.look_latency --feed off     # no frames pushed first

TWO QUESTIONS, MEASURED SEPARATELY, because they are two different paths
through look() and averaging them hides both:

  scene    "What do you see?" -- the observer fast path. A sentence already
           written about a frame that arrived a moment ago. Target: under a
           second to first audio.
  object   "What car is in front of us?" -- the full visual turn: resolve the
           reference, crop, run Qwen for colour and body style, compose.
           Target: under three seconds.

WHAT IT DRIVES. The real /realtime/tool endpoint, which is the one a live
session calls, with frames pushed into the ring first so the observer has
something to describe -- a probe that measured an empty ring would measure the
refusal path and call it fast.

The per-stage numbers come back in the result: look() records them and the
endpoint adds the teacher-context fetch. Nothing here re-derives a timing.
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
CLIP = REPO / "runs" / "road_clip.mp4"

SCENE_Q = "What do you see?"
OBJECT_Q = "What car is in front of us?"

# The stages, in pipeline order.
ORDER = [
    ("teacher_ctx_ms", "teacher context fetch"),
    ("session_ms", "visual session"),
    ("observer_start_ms", "observer start/touch"),
    ("observer_cache_ms", "observer cache lookup"),
    ("observe_now_ms", "observe_now (cache MISS)"),
    ("route_ms", "router classify"),
    ("prepare_ms", "crop + Qwen (prepare)"),
    ("prep_select_frame", "  select frame"),
    ("prep_resolve", "  resolve referent"),
    ("prep_crop", "  crop"),
    ("prep_enrich", "  enrich (Qwen)"),
    ("prep_enrich_generate.lock_ms", "    ...lock wait"),
    ("prep_enrich_generate.prep_ms", "    ...input prep"),
    ("prep_enrich_generate.gen_ms", "    ...decode"),
    ("prep_enrich_generate.in_tokens", "    ...input tokens"),
    ("prep_enrich_generate.out_tokens", "    ...output tokens"),
    ("prep_enrich_generate.passes_in_call", "    ...Qwen passes"),
    ("compose_ms", "backend composition"),
    ("teacher_block_ms", "teacher block build"),
]


def post(url: str, payload: dict, timeout=90.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def clip_frames(n: int = 400) -> list:
    """The road, as JPEGs, decoded once."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(CLIP), "-vf",
         "scale=640:-2,fps=4", "-q:v", "6", "-f", "image2pipe",
         "-vcodec", "mjpeg", "-"], capture_output=True).stdout
    parts, i = [], 0
    while True:
        j = out.find(b"\xff\xd8", i + 2)
        if j < 0:
            parts.append(out[i:])
            break
        parts.append(out[i:j])
        i = j
    return [p for p in parts if len(p) > 2000][:n]


class Feed:
    """Frames flowing, for the whole measurement.

    A DRIVER ASKS WHILE DRIVING, which is the only state any of this is tuned
    for: the observer describes the newest frame about once a second and
    refuses to serve a description older than OBSERVER_FRESH_S. A probe that
    pushes a burst and then asks questions is measuring a stalled feed -- the
    ring goes stale within two seconds and every question takes the slow path,
    which is a real behaviour and not the one being measured here.
    """

    def __init__(self, base: str, session: str, fps: float = 4.0):
        self.base, self.session, self.dt = base, session, 1.0 / fps
        self.frames = clip_frames()
        self._stop = threading.Event()
        self.sent = 0
        self._t = None

    def start(self):
        import httpx

        # A REGISTERED SESSION FIRST. /headway_frame refuses a frame whose
        # session it has never heard of ({"ok": false, "reason":
        # "unknown_session"}), and a refused frame is never retained -- the
        # ring push is gated on the result being ok. A probe that skipped this
        # measured an empty ring and reported every question as taking the slow
        # path, which is true and is not the thing being measured.
        # /session/start MINTS the id -- it does not take one -- so the
        # session this probe drives is whatever it hands back, and everything
        # after this uses that.
        try:
            with httpx.Client(timeout=15.0) as cl:
                r = cl.post(f"{self.base}/session/start", json={})
                sid = (r.json() or {}).get("session_id")
                if sid:
                    self.session = sid
        except Exception as e:
            print(f"   session/start: {type(e).__name__}: {e}")
        self._t = threading.Thread(target=self._run, daemon=True,
                                   name="lookprobe-feed")
        self._t.start()
        return self

    def _run(self):
        import httpx
        i = 0
        last_beat = 0.0
        with httpx.Client(timeout=30.0) as cl:
            while not self._stop.is_set():
                # The drive says it is still driving, as the page does. Without
                # it the reaper ends the session mid-run and the frames start
                # being refused again.
                if time.time() - last_beat > 5.0:
                    last_beat = time.time()
                    try:
                        cl.post(f"{self.base}/session/heartbeat",
                                params={"session_id": self.session})
                    except Exception:
                        pass
                jpg = self.frames[i % len(self.frames)]
                i += 1
                try:
                    r = cl.post(f"{self.base}/headway_frame",
                                params={"session_id": self.session},
                                files={"image": ("f.jpg", jpg, "image/jpeg")},
                                data={"v_host": "13.0", "v_host_age_s": "0.1",
                                      "source": "clip"})
                    if r.status_code == 200:
                        self.sent += 1
                except Exception:
                    pass
                self._stop.wait(self.dt)

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)


def band(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    p95 = (statistics.quantiles(xs, n=20)[18] if len(xs) >= 20 else max(xs))
    return {"p50": round(statistics.median(xs), 1), "p95": round(p95, 1),
            "n": len(xs)}


def run(base: str, session: str, question: str, n: int) -> dict:
    totals, stages, paths = [], {}, []
    for _ in range(n):
        t0 = time.time()
        try:
            r = post(f"{base}/realtime/tool?session_id={session}",
                     {"name": "look", "arguments": {"question": question},
                      "spoken": question})
        except Exception as e:
            print(f"   {type(e).__name__}: {e}")
            continue
        totals.append((time.time() - t0) * 1000.0)
        paths.append(r.get("path") or ("REFUSED:" + str(r.get("note"))))
        for k, v in (r.get("stages") or {}).items():
            if isinstance(v, (int, float)):
                stages.setdefault(k, []).append(float(v))
            elif isinstance(v, dict):
                # e.g. prep_enrich_generate: the Qwen pass's own split
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)):
                        stages.setdefault(f"{k}.{k2}", []).append(float(v2))
        time.sleep(0.4)
    return {"total": band(totals), "stages": stages, "paths": paths}


def report(label: str, res: dict):
    t = res["total"]
    print(f"\n  {label}")
    print(f'    paths: { ", ".join(sorted(set(res["paths"]))) }')
    if not t:
        print("    no measurements")
        return
    print(f'    {"stage":<28}{"p50":>9}{"p95":>9}')
    named = set()
    for key, name in ORDER:
        named.add(key)
        b = band(res["stages"].get(key, []))
        if b:
            print(f'    {name:<28}{b["p50"]:>9.1f}{b["p95"]:>9.1f}')
    # ANYTHING THE LIST DOES NOT NAME. A stage that is not in ORDER is exactly
    # the one worth seeing: it is where the time went after the named ones
    # stopped accounting for it.
    for key in sorted(res["stages"]):
        if key in named:
            continue
        b = band(res["stages"][key])
        if b and (b["p50"] or 0) >= 1.0:
            print(f'    {("? " + key):<28}{b["p50"]:>9.1f}{b["p95"]:>9.1f}')
    print(f'    {"-" * 46}')
    print(f'    {"TOOL ROUND TRIP":<28}{t["p50"]:>9.1f}{t["p95"]:>9.1f}'
          f'   n={t["n"]}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--session", default="lookprobe")
    ap.add_argument("--feed", default="on")
    ap.add_argument("--warm", default="on",
                    help="off = skip the session mint, i.e. measure a cold "
                         "observer cache")
    ap.add_argument("--settle", type=float, default=6.0,
                    help="seconds between frames starting and the first "
                         "question")
    a = ap.parse_args()

    feed = None
    if a.feed != "off":
        print("frames flowing at 4 fps for the whole run ...", flush=True)
        feed = Feed(a.base, a.session).start()
        a.session = feed.session          # the id the server minted
        print(f"  session {a.session[:8]}", flush=True)
        # THE OBSERVER IS STARTED BY THE SESSION MINT, not by the first
        # question -- "start describing the road NOW, not when she is first
        # asked about it" (app.py). A probe that skips the mint measures the
        # cold-start path on every run and calls it the steady state.
        if a.warm != "off":
            try:
                import httpx
                with httpx.Client(timeout=30.0) as cl:
                    cl.post(f"{a.base}/realtime/session",
                            params={"session_id": a.session})
            except Exception as e:
                print(f"   realtime/session: {type(e).__name__}: {e}")
        time.sleep(a.settle)

    try:
        st = json.loads(urllib.request.urlopen(
            f"{a.base}/teachers/status", timeout=5).read())
        print(f'teachers: yield={st.get("yield")} '
              f'ctx_tally={st.get("context_tally")}')
    except Exception:
        pass

    print(f"\n== scene question: {SCENE_Q!r}", flush=True)
    report("scene", run(a.base, a.session, SCENE_Q, a.n))
    print(f"\n== object question: {OBJECT_Q!r}", flush=True)
    report("object", run(a.base, a.session, OBJECT_Q, a.n))
    if feed:
        feed.stop()
        print(f"\n  ({feed.sent} frames pushed during the run)")
    try:
        obs = json.loads(urllib.request.urlopen(
            f"{a.base}/realtime/status", timeout=5).read()).get("observer")
        print(f'  observer: {json.dumps((obs or {}).get("sessions", {}))[:200]}')
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
