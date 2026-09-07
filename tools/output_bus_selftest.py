"""The shared output, in a real browser.

    python -m tools.output_bus_selftest
    python -m tools.output_bus_selftest --url http://127.0.0.1:8888/

static/rio_output.js routes every deterministic line RIO speaks through a Web
Audio bus and back out through a peer connection to itself, because that is the
one render path iOS's echo canceller has a reference signal for. The node
selftest (tools/echo_barge_selftest.js) proves the FALLBACKS: that a line still
has a mouth when the bus is missing, not ready, or fails mid-play.

This proves the other half, and it cannot be proved anywhere but in a browser:
that the bus comes up at all. The graph, the loopback offer/answer, the sink
element actually playing, a real MP3 decoding, and the meter reading a number
that moves when audio is playing. If any of that is wrong, every warning in the
car goes into a suspended audio graph and is never heard — which is a far worse
failure than the echo it was built to fix, and it is invisible from node.

Headless Chromium is not an iPhone and cannot demonstrate iOS echo
cancellation. What it demonstrates is that the mechanism runs, which is the
part that can silently break.

Requires playwright (`pip install playwright && playwright install chromium`).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(("  ok    " if cond else "  FAIL  ") + what)


def section(name):
    print(f"\n=== {name} ===")


# The page loads the bus but only unlocks it inside a user gesture. Headless
# has no gesture to offer, so the call is made directly -- which is exactly
# what the tap handler does.
UNLOCK = """
async () => {
  const o = window.RIO && window.RIO.output;
  if (!o) return { error: 'RIO.output is not loaded' };
  const covered = await o.unlock();
  return { covered: !!covered, state: o.state() };
}
"""

PLAY_CLIP = """
async (url) => {
  const o = window.RIO.output;
  const p = o.playUrl(url);
  const t0 = performance.now();
  // Read the meter while it plays: a bus that renders silence is a bus that
  // is not carrying the line.
  let peak = -100;
  const tick = setInterval(() => { peak = Math.max(peak, o.level()); }, 10);
  let err = null;
  try { await p.play(); } catch (e) { err = String(e && e.message || e); }
  clearInterval(tick);
  return { err: err, peak: peak, ms: Math.round(performance.now() - t0),
           state: o.state() };
}
"""

# The second play of the same file must not fetch or decode it again.
REPLAY = """
async (url) => {
  const o = window.RIO.output;
  const before = o.stats().decoded;
  await o.playUrl(url).play();
  return { before: before, after: o.stats().decoded, plays: o.stats().plays };
}
"""

# Abort must actually stop the source, not merely resolve the promise.
ABORT = """
async (url) => {
  const o = window.RIO.output;
  const p = o.playUrl(url);
  const started = p.play();
  await new Promise(r => setTimeout(r, 60));
  const during = o.level();
  p.abort();
  await new Promise(r => setTimeout(r, 120));
  const after = o.level();
  await started;                       // abort settles it rather than hanging
  return { during: during, after: after };
}
"""

# What rio_speak.js does with a real bus underneath it.
VIA_SPEAK = """
async (url) => {
  const el = document.createElement('audio');
  document.body.appendChild(el);
  const before = Object.assign({}, RIO.speak.stats());
  const p = RIO.speak.provider({ text: 'left here', channel: 'nav',
                                 clipUrl: url, clipFirst: true,
                                 clipElement: el, element: el });
  let err = null;
  try { await p.play(); } catch (e) { err = String(e && e.message || e); }
  const after = RIO.speak.stats();
  return { err: err, bus_before: before.bus, bus_after: after.bus,
           missed: after.bus_missed - before.bus_missed,
           element_src: el.src || '' };
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888/")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed:\n"
              "  pip install playwright && python -m playwright install chromium")
        return 2

    clip = "/static/audio/too_close.mp3"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=[
            "--no-sandbox",
            # No speaker in a container, and no gesture either. Neither is what
            # is under test: the graph, the loopback and the decode are.
            "--autoplay-policy=no-user-gesture-required",
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
        ])
        page = browser.new_page(viewport={"width": 390, "height": 844},
                                is_mobile=True, has_touch=True)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=20000)
        except Exception as e:
            print(f"could not load {args.url}: {e}")
            browser.close()
            return 2
        page.wait_for_timeout(700)

        section("the bus comes up")
        loaded = page.evaluate("() => !!(window.RIO && window.RIO.output)")
        ok(loaded, "rio_output.js is loaded on the page")
        if not loaded:
            browser.close()
            return 1

        ok(page.evaluate("() => RIO.output.ready() === false"),
           "and is not ready before the gesture — nothing may play into a "
           "suspended context")

        r = page.evaluate(UNLOCK)
        ok(not r.get("error"), "unlock() runs without throwing")
        ok(r.get("covered") is True,
           "the loopback connected and the sink element is playing — audio is "
           "going out through the path iOS cancels")
        st = r.get("state") or {}
        ok(st.get("context") == "running", "the shared AudioContext is running")
        ok(st.get("to_destination") is False,
           "and the bus has moved off ctx.destination, so nothing plays twice")
        ok(page.evaluate("() => RIO.output.ready()"),
           "ready() now says a line can be played")

        section("a real clip, through the bus")
        r = page.evaluate(PLAY_CLIP, clip)
        ok(r.get("err") is None, f"a real MP3 decodes and plays ({r.get('err')})")
        ok(r.get("ms", 0) > 100,
           f"and plays for its actual length ({r.get('ms')} ms), rather than "
           "resolving instantly on a source that never started")
        ok(r.get("peak", -100) > -60,
           f"the meter reads the audio while it plays (peak {r.get('peak')} dBFS)")
        ok((r.get("state") or {}).get("decoded", 0) >= 1, "the buffer is cached")

        r = page.evaluate(REPLAY, clip)
        ok(r["before"] == r["after"],
           "playing it again decodes nothing — the fast path stays a decoded "
           "buffer and a start()")

        r = page.evaluate(ABORT, clip)
        ok(r["during"] > -60 and r["after"] <= -60,
           f"abort() actually stops the source ({r['during']} → {r['after']} dBFS)")

        section("silence reads as silence")
        page.wait_for_timeout(200)
        ok(page.evaluate("() => RIO.output.level()") <= -60,
           "with nothing playing the meter is at the floor, so the echo gate "
           "cannot mistake an idle page for RIO talking")

        section("rio_speak.js takes the bus when it is there")
        r = page.evaluate(VIA_SPEAK, clip)
        ok(r.get("err") is None, f"the nav clip plays ({r.get('err')})")
        ok(r["bus_after"] == r["bus_before"] + 1,
           "through the shared output")
        ok(r["missed"] == 0, "with no fallback to the element")
        ok(r["element_src"] == "",
           "and the element was never given a src at all")

        ok(not errors, "no uncaught page errors" +
           ("" if not errors else ": " + "; ".join(errors[:3])))
        browser.close()

    print("\n" + "=" * 72)
    total = len(PASS) + len(FAIL)
    print(f"{len(PASS)}/{total} checks passed")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
