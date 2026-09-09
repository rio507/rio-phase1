"""The page's audio, in a real browser: the shared output, and the unlock.

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

The second half is the Start Drive unlock. iOS will not play an element from a
timer until that element has been played once inside a tap, so every warning
element is started during the Start Drive tap -- and it used to be started on
its OWN CONTENTS, muted. On a phone that is audible: the audio session is
switching route at that instant and the front of the file gets out, so pressing
Start Drive played a prerecorded safety clip on a drive where the server had
asked for nothing (250 frames, every band UNKNOWN, not one speak, in the
session that reported it). Priming plays silence now, and this asserts it by
watching every play() the unlock makes.

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


# Every play() the page makes, with what was loaded at the moment it was made.
# Installed before the unlock so the unlock's own calls are the ones recorded.
SPY = """
() => {
  window.__plays = [];
  window.__primed = [];          // the elements themselves, to inspect after
  const proto = HTMLMediaElement.prototype;
  const real = proto.play;
  proto.play = function () {
    window.__plays.push({
      id: this.id || '',
      src: String(this.src || ''),
      // What the element is ACTUALLY about to render. Assigning .src does not
      // unload the previous resource -- this is the field that catches a
      // priming call that is still holding the warning it was meant to skip.
      current: String(this.currentSrc || ''),
      // 0 = HAVE_NOTHING: load() has reset the element and dropped whatever it
      // was holding, so nothing already decoded can start. This is the field
      // that actually settles it -- `currentSrc` lags behind load() by a task
      // in Chromium and reports the old file for an instant after it has
      // already been abandoned.
      ready: this.readyState,
      muted: !!this.muted,
      volume: this.volume,
    });
    window.__primed.push(this);
    return real.apply(this, arguments);
  };
  return true;
}
"""

# The Start Drive tap, minus the camera and the session: the unlock is the part
# under test and it is the first thing that tap does.
UNLOCK_WARNINGS = """
async () => {
  if (!(window.RIO && RIO.headway && RIO.headway.unlock)) {
    return { error: 'RIO.headway.unlock is not exposed' };
  }
  RIO.headway.unlock();
  if (window.RIO.nav && RIO.nav.unlock) { try { RIO.nav.unlock(); } catch (e) {} }
  await new Promise(r => setTimeout(r, 400));
  return { plays: window.__plays };
}
"""

# ...and afterwards the elements must be holding their real files again, ready
# to fire with no network in the path.
# The warning clips are `new Audio(...)` and are never in the document, so the
# only handle on them is the one the spy kept.
CLIP_STATE = """
() => (window.__primed || []).map(a => ({
  id: a.id || '',
  src: String(a.src || ''),
  current: String(a.currentSrc || ''),
  ready: a.readyState,
}))
"""


# ---------------------------------------------------------------------------
# The bus watch, over a session's lifetime
# ---------------------------------------------------------------------------
# The watch that reads RIO.output.state() once a second is started when a live
# conversation opens and stopped when it closes -- and for a long time it was
# neither, because the whole block that declares it sat INSIDE toggleLive(),
# below that function's early return for ending a conversation. The second tap
# on the talk control ran stopBusWatch() against a `let busWatch` that had not
# been evaluated yet in that call and threw a TDZ ReferenceError, one line
# before setMicState('idle').
#
# Three things followed, and all three are asserted below:
#
#   1. the control never went back to idle -- it still said Listening, with a
#      green ring, on a conversation that was over;
#   2. `live` had already been nulled two lines above the throw, so the NEXT
#      tap fell through to the connect path and opened a SECOND concurrent
#      session, each with a bus watch of its own;
#   3. no watch was ever cleared, so every conversation leaked a 1s interval
#      that went on posting LIVE_BUS_HEALTH for a session that had ended.
#
# And one more that survived the hoist: a stopped session's tail events kept
# driving the control. LIVE_RESPONSE_END comes back out of stop() through the
# arbiter, sometimes synchronously and sometimes a tick later, and the late
# arm put the button back to Listening after the driver had ended it.
#
# THE SESSION IS A STUB. RIO.realtime.connect is replaced before the first tap,
# so this runs with no API key, no socket, no model and no audio -- and, more
# importantly, no race: the "late tail event" is delivered by hand at the exact
# moment that used to be a coin toss.
SESSION_STUB = """
() => {
  window.__iv = [];
  const si = window.setInterval, ci = window.clearInterval;
  window.setInterval = function (fn, ms) {
    const id = si.apply(this, arguments);
    if (ms === 1000) window.__iv.push({ id: id, live: true });   // the bus watch
    return id;
  };
  window.clearInterval = function (id) {
    const r = window.__iv.find(x => x.id === id);
    if (r) r.live = false;
    return ci.apply(this, arguments);
  };
  window.__sessions = 0;
  if (!(window.RIO && RIO.realtime && RIO.realtime.connect)) return false;
  RIO.realtime.connect = async (opts) => {
    window.__sessions++;
    const s = { onEvent: opts.onEvent, stopped: false,
                stop: function () { this.stopped = true; } };
    window.__last = s;
    return s;
  };
  return true;
}
"""

TALK_STATE = """
() => ({
  mic: document.getElementById('mic').dataset.state,
  ring: document.getElementById('voicering').className,
  sessions: window.__sessions,
  watches: window.__iv.length,
  running: window.__iv.filter(r => r.live).length,
})
"""

# The tail of a session that has already been ended, arriving late.
LATE_TAIL = """
() => {
  if (!window.__last) return false;
  window.__last.onEvent({ type: 'LIVE_RESPONSE_END', response_id: 'tail' });
  window.__last.onEvent({ type: 'LIVE_RESPONSE_START', response_id: 'tail' });
  return true;
}
"""


def run_bus_watch(page, errors):
    section("the bus watch starts and stops with the conversation")

    if not page.evaluate(SESSION_STUB):
        ok(False, "RIO.realtime.connect exists to be stubbed")
        return

    before = len(errors)

    page.click("#mic")
    page.wait_for_timeout(600)
    r = page.evaluate(TALK_STATE)
    ok(r["sessions"] == 1, f"one tap opens one session ({r['sessions']})")
    ok(r["mic"] == "listening",
       f"the control says the conversation is open (is {r['mic']!r})")
    ok(r["running"] == 1,
       f"and exactly one bus watch is running ({r['running']})")

    page.click("#mic")
    page.wait_for_timeout(400)
    r = page.evaluate(TALK_STATE)
    # The regression, asserted: this said 'listening' for as long as the
    # declarations lived inside toggleLive().
    ok(r["mic"] == "idle",
       f"a second tap returns the control to idle (is {r['mic']!r})")
    ok("listening" not in r["ring"] and "speaking" not in r["ring"],
       f"and the ring goes with it (is {r['ring']!r})")
    ok(r["running"] == 0,
       f"the bus watch was cleared rather than left running ({r['running']})")
    ok(len(errors) == before,
       "ending a conversation throws nothing"
       + ("" if len(errors) == before else ": " + "; ".join(errors[before:])))

    ok(page.evaluate(LATE_TAIL), "the stopped session can still be poked")
    page.wait_for_timeout(200)
    r = page.evaluate(TALK_STATE)
    ok(r["mic"] == "idle",
       f"a stopped session's late RESPONSE_END does not reopen the control "
       f"(is {r['mic']!r})")
    ok("speaking" not in r["ring"],
       f"nor put the ring back to speaking (is {r['ring']!r})")

    page.click("#mic")
    page.wait_for_timeout(600)
    r = page.evaluate(TALK_STATE)
    ok(r["sessions"] == 2,
       f"tapping again opens the second session, not a third ({r['sessions']})")
    ok(r["running"] == 1,
       f"with one watch running, not one per conversation ever held "
       f"({r['running']} of {r['watches']} started)")

    # Leave the page as it was found.
    page.click("#mic")
    page.wait_for_timeout(300)


# ---------------------------------------------------------------------------
# Cancelling a connect
# ---------------------------------------------------------------------------
# toggleLive() used to open with `if (liveBusy) return;`, which made the talk
# control dead for the whole of a connect -- an ephemeral-session POST,
# getUserMedia and a WebRTC offer/answer, about 1.3s against a warm server on
# a desk and longer on a phone over cellular. A driver who taps Talk to RIO,
# changes their mind and taps again got nothing, and then got a conversation
# they had already cancelled.
#
# connect() has no abort, so the cancel is expressed the way ending a session
# is -- the generation is spent -- and the session that eventually resolves is
# stopped where it lands. This asserts the three things that has to mean: the
# control answers the cancelling tap immediately, the arriving session is torn
# down rather than adopted, and no bus watch is started for it.
#
# The stub here is a connect held open on purpose, released by hand, so the
# window that used to be a race is a place the test can stand still inside.
HELD_CONNECT_STUB = """
() => {
  window.__iv = [];
  const si = window.setInterval, ci = window.clearInterval;
  window.setInterval = function (fn, ms) {
    const id = si.apply(this, arguments);
    if (ms === 1000) window.__iv.push({ id: id, live: true });
    return id;
  };
  window.clearInterval = function (id) {
    const r = window.__iv.find(x => x.id === id);
    if (r) r.live = false;
    return ci.apply(this, arguments);
  };
  window.__attempts = 0; window.__opened = 0; window.__stopped = 0;
  window.__resolve = null; window.__reject = null;
  if (!(window.RIO && RIO.realtime)) return false;
  RIO.realtime.connect = (opts) => new Promise((res, rej) => {
    window.__attempts++;
    window.__resolve = () => {
      window.__opened++;
      res({ onEvent: opts.onEvent, stop: function () { window.__stopped++; } });
    };
    window.__reject = () => rej(new Error('socket refused'));
  });
  return true;
}
"""

CONNECT_STATE = """
() => ({
  mic: document.getElementById('mic').dataset.state,
  ring: document.getElementById('voicering').className,
  attempts: window.__attempts,
  opened: window.__opened,
  stopped: window.__stopped,
  running: window.__iv.filter(r => r.live).length,
})
"""


def run_connect_cancel(page, errors):
    section("a tap during the connect cancels it")

    if not page.evaluate(HELD_CONNECT_STUB):
        ok(False, "RIO.realtime exists to be stubbed")
        return
    before = len(errors)

    page.click("#mic")
    page.wait_for_timeout(400)
    r = page.evaluate(CONNECT_STATE)
    ok(r["attempts"] == 1 and r["mic"] == "connecting",
       f"a tap starts a connect and the control says so (is {r['mic']!r})")

    # The regression: this used to stay 'connecting' for the whole connect.
    page.click("#mic")
    page.wait_for_timeout(300)
    r = page.evaluate(CONNECT_STATE)
    ok(r["mic"] == "idle",
       f"a second tap answers immediately, without waiting for the connect "
       f"(is {r['mic']!r})")
    ok(r["attempts"] == 1,
       f"and does not start a second connect on top of the first "
       f"({r['attempts']} attempts)")

    # Now let the cancelled connect land.
    page.evaluate("() => window.__resolve()")
    page.wait_for_timeout(500)
    r = page.evaluate(CONNECT_STATE)
    ok(r["opened"] == 1 and r["stopped"] == 1,
       f"the session that arrives is torn down, not adopted "
       f"(opened {r['opened']}, stopped {r['stopped']})")
    ok(r["mic"] == "idle",
       f"the control stays idle — the driver cancelled (is {r['mic']!r})")
    ok("listening" not in r["ring"] and "speaking" not in r["ring"],
       f"and the ring never lights for it (is {r['ring']!r})")
    ok(r["running"] == 0,
       f"no bus watch was started for a session nobody wanted ({r['running']})")
    ok(len(errors) == before,
       "cancelling throws nothing"
       + ("" if len(errors) == before else ": " + "; ".join(errors[before:])))

    # A cancelled connect that then FAILS must not cost the drive its live
    # mode: liveMode = false in the catch would drop every later tap to
    # hold-to-talk, on the strength of a session the driver had called off.
    page.click("#mic")
    page.wait_for_timeout(300)
    page.click("#mic")
    page.wait_for_timeout(300)
    page.evaluate("() => window.__reject()")
    page.wait_for_timeout(500)
    r = page.evaluate(CONNECT_STATE)
    ok(r["mic"] == "idle",
       f"a cancelled connect that then fails leaves the control idle "
       f"(is {r['mic']!r})")

    page.click("#mic")
    page.wait_for_timeout(400)
    r = page.evaluate(CONNECT_STATE)
    ok(r["mic"] == "connecting",
       f"and live conversation still works afterwards — the cancelled failure "
       f"did not fall the drive back to hold-to-talk (is {r['mic']!r})")

    # Leave the page idle: cancel the connect this check just started.
    page.click("#mic")
    page.wait_for_timeout(200)


def run_unlock(page):
    section("Start Drive primes on silence, never on a warning")
    page.evaluate(SPY)
    r = page.evaluate(UNLOCK_WARNINGS)
    ok(not r.get("error"), f"the warning unlock runs ({r.get('error')})")
    if r.get("error"):
        return
    plays = r.get("plays") or []
    ok(len(plays) > 0,
       f"the unlock does play elements — that is what unlocks them ({len(plays)})")

    named = [p for p in plays
             if "/static/audio/" in (p.get("src") or "") or
             (p.get("src") or "").endswith(".mp3")]
    ok(not named,
       "not one primed element was asked to play a warning file"
       + ("" if not named else
          " — asked for: " + ", ".join(sorted({p["src"].split("/")[-1]
                                               for p in named}))))

    silent = [p for p in plays if p.get("src", "").startswith("blob:")
              or "data:audio" in p.get("src", "")]
    ok(len(silent) == len(plays),
       f"every primed element was given the silent buffer ({len(silent)} of "
       f"{len(plays)})")

    # The one that decides whether a warning can actually be heard: with the
    # element reset to HAVE_NOTHING there is no decoded clip left in it to
    # start, whatever currentSrc still says for the next task.
    holding = [p for p in plays if (p.get("ready") or 0) != 0]
    ok(not holding,
       "and every one was reset to HAVE_NOTHING first, so there was no "
       "decoded warning left in it to escape"
       + ("" if not holding else
          " — readyState " + ", ".join(str(p.get("ready")) for p in holding)
          + " on " + ", ".join(sorted({(p.get("current") or "?").split("/")[-1]
                                       for p in holding}))))

    # And the files are back on the elements afterwards: a clip that has to be
    # re-fetched at the junction is what preloading exists to prevent.
    after = page.evaluate(CLIP_STATE)
    restored = [a for a in after if "/static/audio/" in (a.get("src") or "")]
    ok(len(restored) >= 3,
       f"the warning files are back on their elements afterwards "
       f"({len(restored)} holding a clip)")
    ok(all(a.get("ready", 0) >= 1 for a in restored),
       "and have loaded again, so firing one is still a play() with no "
       "network in its path")
    ok(all(not (a.get("src") or "").startswith("blob:") for a in after),
       "and no element was left holding the silent buffer")


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

        run_unlock(page)
        run_bus_watch(page, errors)
        run_connect_cancel(page, errors)

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
