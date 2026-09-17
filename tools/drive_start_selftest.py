#!/usr/bin/env python3
"""drive_start_selftest.py — Start Drive on a phone must populate SESSION and push frames in 3 s.

    python3 tools/drive_start_selftest.py
    python3 tools/drive_start_selftest.py --base http://127.0.0.1:8888

THE DAY. 2026-09-17: four drive starts on an iPhone created their session
and never started the frame pipeline. SESSION stayed "—", FRAMES 0, the
status row said "Starting drive..." for as long as anyone looked. Headless
Chromium started 273 frames from the same page, because the step that killed
the phone is one Chromium never takes: on Safari `navigator.wakeLock.request`
rejects, or sits pending, and the start path reacted to both by dying
silently -- once because it awaited the pending promise, once because the
rejection's own diagnostic threw and Promise.race passed that on.

So this drives the REAL page in Chromium with an iPhone user agent and a fake
camera, and makes Chromium behave like the phone: `navigator.wakeLock.request`
is stubbed to REJECT in one run and to NEVER SETTLE in another. In every run
the SESSION cell must fill and FRAMES must climb within 3 s of the tap, and
the page must not throw. On the page as it was at 21:25 today, both runs fail.
"""
import argparse, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
CAM_Y4M = REPO / "runs" / "probe" / "cam.y4m"

IPHONE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 "
             "(KHTML, like Gecko) Version/26.5.2 Mobile/15E148 Safari/604.1")

STUBS = {
    "reject": "navigator.wakeLock = { request: () => Promise.reject(new DOMException('denied', 'NotAllowedError')) };",
    "pending": "navigator.wakeLock = { request: () => new Promise(() => {}) };",
    "native": "",
}

checks = 0
failures = 0
def ok(cond, what):
    global checks, failures
    checks += 1
    if not cond:
        failures += 1
    print(("  ok    " if cond else "  FAIL  ") + what, flush=True)


def run(base, mode, headed=False, within_s=3.0, html=None):
    from playwright.sync_api import sync_playwright
    errors, console = [], []
    old_html = Path(html).read_text() if html else None
    args = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={CAM_Y4M}", "--autoplay-policy=no-user-gesture-required"]
    with sync_playwright() as p:
        b = p.chromium.launch(headless=not headed, args=args)
        # A POSITION, or the permissions module waits for one: the start path
        # awaits RIO.permissions.request(), which on a granted-but-silent
        # geolocation never resolves. The phone had a fix; so does this.
        ctx = b.new_context(user_agent=IPHONE_UA, viewport={"width": 390, "height": 844},
                            is_mobile=True, has_touch=True,
                            permissions=["camera", "microphone", "geolocation"],
                            geolocation={"latitude": 34.0407, "longitude": -118.5263, "accuracy": 7})
        ctx.add_init_script(STUBS[mode])
        page = ctx.new_page()
        if old_html is not None:
            # The page as it was at some revision, with every other request --
            # the scripts, the API -- still going to the running server.
            page.route("**/?token=*", lambda route: route.fulfill(
                status=200, content_type="text/html", body=old_html))
        page.on("pageerror", lambda e: errors.append(str(e)[:200]))
        page.on("console", lambda m: console.append(m.text[:160]) if m.type == "error" else None)
        page.goto(base + "/?token=6yy5p039c732f6bwusl0", wait_until="domcontentloaded")
        page.wait_for_timeout(1500)
        page.click("#drive")
        t0 = time.time()
        session_at = frames_at = None
        status = ""
        while time.time() - t0 < within_s + 2.0:
            st = page.evaluate("""() => ({
                session: (window.RIO && RIO.sessionId) || null,
                driving: !!(window.RIO && RIO.driving),
                status: (document.getElementById('drivestatus') || {}).textContent || '',
                frames: (window.RIO && RIO.headway && RIO.headway.streamStats && RIO.headway.streamStats()) ? RIO.headway.streamStats().sent : 0,
            })""")
            status = st["status"]
            if st["session"] and session_at is None:
                session_at = time.time() - t0
            if st["frames"] and st["frames"] > 0 and frames_at is None:
                frames_at = time.time() - t0
                break
            page.wait_for_timeout(100)
        b.close()
    return {"session_at": session_at, "frames_at": frames_at, "status": status,
            "errors": errors, "console": console}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--html", default=None, help="serve this index.html for the root request (a past revision)")
    ap.add_argument("--modes", default="native,reject,pending")
    args = ap.parse_args()
    for mode in args.modes.split(","):
        print(f"\n=== wakeLock.request: {mode}{'  [' + args.html + ']' if args.html else ''} ===")
        r = run(args.base, mode, headed=args.headed, html=args.html)
        ok(r["session_at"] is not None and r["session_at"] <= 3.0,
           f"SESSION populates within 3 s of the tap ({r['session_at'] and round(r['session_at'], 2)} s)")
        ok(r["frames_at"] is not None and r["frames_at"] <= 3.0,
           f"FRAMES climbs within 3 s ({r['frames_at'] and round(r['frames_at'], 2)} s)")
        ok(not r["errors"], f"no page error ({r['errors'][:1]})")
        ok("Starting drive" not in r["status"],
           f"the status row moved on: {r['status'][:60]!r}")
    print(f"\n{'FAILED ' + str(failures) + '/' if failures else 'PASSED '}{checks} checks")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
