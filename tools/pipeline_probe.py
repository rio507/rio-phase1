"""The whole frame pipeline, per stage, in a real browser, in both modes.

    python -m tools.pipeline_probe
    python -m tools.pipeline_probe --mode clip --seconds 40
    python -m tools.pipeline_probe --teachers off

WHY A BROWSER AND NOT A BENCH. tools/frame_transport_bench.py drives frames
from Python and measures the transport and the server beautifully. It cannot
see the half of the pipeline that lives on the page -- how often a picture is
actually taken, what the encode costs, what the overlay draw costs, and what
those do to each other when they share a thread. In clip mode that half is
where the work is: there is a second hidden video element running ahead of the
visible one, a keyframe pipeline, and an overlay holding a queue of results
waiting for the picture to reach them.

So this drives the REAL page, in the real browser, through the real socket, and
reads the page's OWN numbers back out (RIO.headway.streamStats()). Nothing here
re-derives a timing; a probe that computed its own would be measuring itself.

  clip     a file uploaded through the real file input, played, with
           auto-detect on -- which is the mode the complaint is about.
  camera   getUserMedia, fed by Chromium's fake device from the same footage,
           so the two modes differ in the mode and not in the road.

--teachers on|idle|off does not touch the teacher services; it only records
what they were doing, so a run can be labelled honestly. Use
tools/teacher_ctl.py to actually change them.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
CLIP = REPO / "runs" / "probe" / "road_40s.mp4"
CAM_Y4M = REPO / "runs" / "probe" / "cam.y4m"

# The stages, in the order a frame passes through them. Named here so the
# report reads as a pipeline rather than as an alphabetical dump.
ORDER = [
    ("capture_interval_ms", "capture interval"),
    ("encode_ms", "encode"),
    ("push_ms", "push (send->result)"),
    ("srv_transport_ms", "  transport"),
    ("srv_queue_ms", "  queue wait"),
    ("srv_decode", "  jpeg decode"),
    ("srv_lanes", "  lanes"),
    ("srv_depth", "  depth"),
    ("srv_detect", "  detector"),
    ("srv_membership", "  membership"),
    ("srv_track_filter", "  track/filter"),
    ("srv_total", "  server total"),
    ("overlay_draw_ms", "overlay draw"),
    ("frame_age_ms", "FRAME AGE at detection"),
]


def ensure_assets():
    """Build the probe's clip and fake-camera feed if they are not there.

    Not committed: the clip is 6 MB and the raw Y4M the fake camera wants is
    200. Both are derived from runs/road_clip.mp4, which is, so a fresh
    checkout can run this with no setup and every run uses the same road.
    """
    import subprocess
    src = REPO / "runs" / "road_clip.mp4"
    if not src.exists():
        raise SystemExit(f"no source clip at {src}")
    CLIP.parent.mkdir(parents=True, exist_ok=True)
    if not CLIP.exists():
        print(f"building {CLIP.name} ...", flush=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-stream_loop", "7",
                        "-i", str(src), "-c", "copy", str(CLIP)], check=True)
    if not CAM_Y4M.exists():
        print(f"building {CAM_Y4M.name} ...", flush=True)
        # 640x352 at 10 fps: the shape the socket path sends anyway, so the
        # fake camera is not quietly a higher-resolution source than a phone.
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(CLIP),
                        "-t", "20", "-vf", "scale=640:352,fps=10",
                        "-pix_fmt", "yuv420p", str(CAM_Y4M)], check=True)


def teacher_state() -> dict:
    """What the teachers were doing, recorded rather than assumed."""
    import urllib.request
    out = {}
    for name, port in (("alpamayo", 8801), ("cosmos", 8802)):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2) as r:
                d = json.loads(r.read())
            out[name] = {"loaded": d.get("loaded"),
                         "inflight": d.get("inflight"),
                         "queue": d.get("queue_depth")}
        except Exception as e:
            out[name] = {"loaded": False, "error": type(e).__name__}
    return out


_SETUP = """
(async (args) => {
  const [mode, clipName] = args;
  // The page decides what it looks at; this only asks. See rio_source.js.
  if (mode === 'camera') {
    if (window.RIO && RIO.source && RIO.source.useCamera) RIO.source.useCamera();
    // ACQUIRE THE PICTURE. rio_source owns this -- startFeed is the one
    // function on the page that calls getUserMedia, and nothing downstream
    // opens a camera for itself (that was the bug rio_source.js exists to
    // end). Without it the feed element has no stream and every tick skips.
    if (window.RIO && RIO.source && RIO.source.startFeed) {
      try { await RIO.source.startFeed(); } catch (e) { return 'feed: ' + e.message; }
    }
  }
  // Start the headway feed exactly as the drive does -- INCLUDING the element
  // getter, which startStream takes as an argument. Calling it bare leaves the
  // transport with no element to photograph and every tick skips silently,
  // which is a very convincing impression of a slow pipeline.
  if (window.RIO && RIO.headway && RIO.headway.unlock) { try { await RIO.headway.unlock(); } catch (e) {} }
  const elFn = () => (RIO.source && RIO.source.element && RIO.source.element())
                     || document.getElementById('video');
  if (window.RIO && RIO.headway && RIO.headway.startStream) {
    RIO.headway.startStream(elFn, () => {});
  }
  return true;
})
"""


def run_mode(url: str, mode: str, seconds: float, headed: bool) -> dict:
    from playwright.sync_api import sync_playwright

    args = ["--autoplay-policy=no-user-gesture-required",
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={CAM_Y4M}"]
    with sync_playwright() as p:
        b = p.chromium.launch(headless=not headed, args=args)
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)[:200]))
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(2500)

        if mode == "clip":
            # THE REAL UPLOAD PATH. Setting the input is what a driver does;
            # poking RIO.source directly would skip whatever the upload
            # handler sets up, which is part of what is being measured.
            page.set_input_files("input[type=file]", str(CLIP))
            page.wait_for_timeout(1500)
            # AUTO-DETECT ON PLAY is the path the complaint is about: the
            # driver presses play and the run starts itself. Driven exactly
            # that way rather than by calling runVideo(), so whatever the play
            # handler sets up is in the measurement.
            page.evaluate("""() => {
              if (RIO.headway && RIO.headway.resetReplayStats) {
                RIO.headway.resetReplayStats();
              }
              const b = document.getElementById('autodetect');
              if (b && b.getAttribute('aria-pressed') !== 'true') b.click();
              const v = document.getElementById('preview');
              if (v) { v.loop = true; v.muted = true; v.play(); }
            }""")
            page.wait_for_timeout(1200)

        if mode != "clip":
            page.evaluate(_SETUP, [mode, CLIP.name])
        page.wait_for_timeout(int(seconds * 1000))

        stats = page.evaluate(
            "() => (window.RIO && RIO.headway && RIO.headway.streamStats) "
            "? RIO.headway.streamStats() : null")
        # CLIP MODE IS A DIFFERENT LOOP and reports separately -- runVideo()
        # has its own capture, encode and POST, and never touches the socket.
        replay = page.evaluate(
            "() => (window.RIO && RIO.headway && RIO.headway.replayStats) "
            "? RIO.headway.replayStats() : null")
        lag = page.evaluate(
            "() => (window.RIO && RIO.overlay && RIO.overlay.lagStats) "
            "? RIO.overlay.lagStats() : null")
        b.close()
    return {"mode": mode, "stats": stats, "replay": replay, "lag": lag,
            "errors": errors[:5]}


def band_str(b) -> str:
    if not b or not b.get("n"):
        return "        --"
    def f(x):
        return "  --" if x is None else f"{x:6.1f}"
    return f'{f(b.get("p50"))} {f(b.get("p90"))} {f(b.get("p99"))}  n={b.get("n")}'


def report(res: dict):
    st = res.get("stats") or {}
    stages = st.get("stages") or {}
    rp = res.get("replay") or {}
    # The replay loop's numbers ARE the clip pipeline. When it ran, it is what
    # the report is about; the socket stats describe a feed that was idle.
    if rp and (rp.get("capture_interval_ms") or {}).get("n"):
        print(f'\n  mode={res["mode"]}  (replay loop — POST /headway_frame, '
              f'serial)')
        b = rp.get("bytes") or {}
        print(f'  frame bytes: p50 {b.get("p50")}  p90 {b.get("p90")}')
        ci = (rp.get("capture_interval_ms") or {}).get("p50")
        if ci:
            print(f'  effective fps (capture interval p50): {1000.0 / ci:.2f}')
        print(f'\n  {"stage":<26}{"p50":>7}{"p90":>7}{"p99":>7}')
        for key, label in ORDER:
            if key in rp:
                print(f"  {label:<26}{band_str(rp[key])}")
        if (stages.get("overlay_draw_ms") or {}).get("n"):
            print(f'  {"overlay draw":<26}{band_str(stages["overlay_draw_ms"])}')
        if res.get("errors"):
            print("  page errors:", res["errors"])
        return
    print(f'\n  mode={res["mode"]}  '
          f'sent={st.get("sent")} results={st.get("results")} '
          f'dropped_server={st.get("dropped_server")} '
          f'encoder={st.get("encoder")}')
    print(f'  skipped: inflight={st.get("skipped_inflight")} '
          f'buffer={st.get("skipped_buffer")} capture={st.get("skipped_capture")}'
          f'   depth={st.get("max_inflight")} quality={st.get("quality")} '
          f'bytes={st.get("mean_bytes")}')
    ci = (stages.get("capture_interval_ms") or {}).get("p50")
    print(f'  effective fps (from capture interval p50): '
          f'{(1000.0 / ci):.2f}' if ci else '  effective fps: --')
    print(f'\n  {"stage":<26}{"p50":>7}{"p90":>7}{"p99":>7}')
    for key, label in ORDER:
        if key in stages:
            print(f"  {label:<26}{band_str(stages[key])}")
    if res.get("errors"):
        print("  page errors:", res["errors"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888/")
    ap.add_argument("--mode", default="clip,camera")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--teachers", default="", help="label for this run")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    ensure_assets()
    before = teacher_state()
    print(f"teachers at start: {json.dumps(before)}")
    out = {"label": a.teachers, "teachers_before": before, "runs": []}
    for mode in [m.strip() for m in a.mode.split(",") if m.strip()]:
        print(f"\n== {mode} ({a.seconds:.0f}s) "
              f"{('[' + a.teachers + ']') if a.teachers else ''}", flush=True)
        try:
            r = run_mode(a.base, mode, a.seconds, a.headed)
        except Exception as e:
            print(f"   {type(e).__name__}: {e}")
            continue
        out["runs"].append(r)
        report(r)
    out["teachers_after"] = teacher_state()
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
