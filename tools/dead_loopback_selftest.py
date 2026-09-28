"""A dead loopback, made on purpose: the bus must never be left talking into it.

    python -m tools.dead_loopback_selftest
    python -m tools.dead_loopback_selftest --url http://127.0.0.1:8888/

THE FAULT, from the drive of 2026-09-28 (4ae33786). The driver said "hello"
four times and was answered four times, and heard none of it. Every row the
log had said she spoke -- heard_frac 1, out_peak_db -11, playout `drained` --
because every one of them is measured on the bus, UPSTREAM of the loopback.
The bus_health rows said the rest: covered true, to_destination false, and
total_samples 0 for 227 seconds. The loopback had negotiated, the sink element
was "playing", and the receiver never took in a single sample. f7d708af and
3f219de7 show the same signature.

This makes that link on purpose, two ways, without any hook in the code under
test -- so the same file can be run against the old rio_output.js and seen to
fail:

  born dead     the loopback's sender has its track replaced with null before
                negotiation. SDP, ontrack, srcObject and play() all succeed;
                no audio is ever sent. This reproduces the drive exactly:
                covered, total_samples 0, clock_delta_ppm ~ -1,000,000.

  dies later    a healthy loopback carries audio, and then its sender's track
                is replaced with null. Samples keep being counted (the receiver
                fills with silence), so no counter sees it: only the sink meter
                -- sound going in, none coming out -- can.

For each: the bus must end up on ctx.destination, a `fallback` event must say
why, a clip played afterwards must be audible on the path it is actually on,
and (dies-later) the loopback must be rebuilt and proved before the bus moves
back onto it.

Requires playwright (`pip install playwright && playwright install chromium`).
"""
import argparse
import os
import sys

PASS, FAIL = [], []


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(("  ok    " if cond else "  FAIL  ") + what)


def section(name):
    print(f"\n=== {name} ===")


# Installed before the unlock. Records every sender a peer connection makes and,
# in 'born_dead' mode, empties it on the spot. Only the loopback adds tracks on
# this page before a live session exists, so every sender here is the loopback's.
SABOTAGE = """
(mode) => {
  window.__senders = [];
  window.__busEvents = [];
  const real = RTCPeerConnection.prototype.addTrack;
  RTCPeerConnection.prototype.addTrack = function (t, s) {
    const snd = real.apply(this, arguments);
    window.__senders.push(snd);
    if (window.__sabotage === 'born_dead') snd.replaceTrack(null);
    return snd;
  };
  window.__sabotage = mode;
  if (RIO.output.onEvent) {
    RIO.output.onEvent((kind, d) => window.__busEvents.push({ kind, d }));
  }
  if (RIO.output._limits) RIO.output._limits({ rebuild_ms: 6000 });
  return true;
}
"""

UNLOCK = """
async () => {
  const covered = await RIO.output.unlock();
  return { covered: !!covered, state: RIO.output.state() };
}
"""

STATE = """
async () => {
  await RIO.output.sample();
  const s = RIO.output.state();
  return { covered: s.covered, to_destination: s.to_destination,
           path: s.path === undefined ? null : s.path,
           fallback: s.fallback === undefined ? null : s.fallback,
           total_samples: s.health.total_samples,
           stats: s.stats, events: window.__busEvents || [] };
}
"""

# Play a clip and read both meters while it plays. `sink` is null where the
# module has no sink meter at all -- which is the old code, and a failure.
PLAY = """
async (url) => {
  const o = RIO.output;
  const p = o.playUrl(url);
  let bus = -100, sink = null, paths = {};
  const tick = setInterval(() => {
    bus = Math.max(bus, o.level());
    if (o.sinkLevel) {
      const s = o.sinkLevel();
      paths[s.path] = (paths[s.path] || 0) + 1;
      if (s.db !== null) sink = Math.max(sink === null ? -100 : sink, s.db);
    }
  }, 20);
  let err = null;
  try { await p.play(); } catch (e) { err = String(e && e.message || e); }
  clearInterval(tick);
  return { err, bus, sink, paths };
}
"""

# Keep her talking until the watchdog acts, or give up. The watchdog samples
# once a second and wants several loud ticks in a row.
TALK_UNTIL_FALLBACK = """
async ([url, budgetMs]) => {
  const o = RIO.output;
  const t0 = performance.now();
  while (performance.now() - t0 < budgetMs) {
    if (!o.state().covered) break;
    try { await o.playUrl(url).play(); } catch (e) {}
  }
  return { ms: Math.round(performance.now() - t0), covered: o.state().covered };
}
"""


# Every report the page makes is captured here and NEVER sent: a selftest must
# not write rows into whatever drive the server has open.
CAPTURE_REPORTS = """
(() => {
  window.__reports = [];
  const real = window.fetch;
  window.fetch = function (u, opts) {
    if (String(u).indexOf('/realtime/cutoff') !== -1) {
      try { window.__reports.push(JSON.parse(opts.body)); } catch (e) {}
      return Promise.resolve(new Response('{}', { status: 200 }));
    }
    return real.apply(this, arguments);
  };
})();
"""

REPORTS = "() => window.__reports || []"


# A conversation with the network stubbed out, so events can go through the
# page's real onEvent -> reportLive path. Same stub as output_bus_selftest.
OPEN_SESSION = """
async () => {
  if (!(window.RIO && RIO.realtime && RIO.realtime.connect)) return false;
  RIO.realtime.connect = async (opts) => {
    const s = { onEvent: opts.onEvent, stopped: false,
                stop: function () { this.stopped = true; } };
    window.__last = s;
    return s;
  };
  return true;
}
"""

# What the barge gate emits, through the session's own event hook.
GATE_EVENTS = """
() => {
  const on = window.__last && window.__last.onEvent;
  if (!on) return false;
  on({ type: 'LIVE_BARGE_IN' });
  on({ type: 'LIVE_ECHO_SUPPRESSED', mic_db: -30, out_db: -20 });
  on({ type: 'LIVE_CUTOFF', cause: 'false_barge_in' });
  return true;
}
"""


def reports(page, kind):
    return [r for r in page.evaluate(REPORTS) if r.get("kind") == kind]


def fresh(browser, url, mode):
    page = browser.new_page(viewport={"width": 390, "height": 844},
                            is_mobile=True, has_touch=True)
    page.add_init_script(CAPTURE_REPORTS)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url, wait_until="domcontentloaded", timeout=20000)
    page.wait_for_timeout(700)
    page.evaluate(SABOTAGE, mode)
    return page, errors


def run_healthy(browser, url, clip):
    section("control: a healthy loopback is kept, and the sink meter hears it")
    page, errors = fresh(browser, url, "none")
    r = page.evaluate(UNLOCK)
    ok(r["covered"] is True, "a healthy loopback is still taken")
    page.wait_for_timeout(3500)
    st = page.evaluate(STATE)
    ok(st["covered"] and not st["to_destination"],
       "and kept: three seconds of watchdog do not fall back from a live link "
       f"(covered {st['covered']}, to_destination {st['to_destination']})")
    ok(st["total_samples"] > 0,
       f"the receiver counts samples ({st['total_samples']})")
    r = page.evaluate(PLAY, clip)
    ok(r["err"] is None, f"a clip plays through it ({r['err']})")
    ok(r["sink"] is not None and r["sink"] > -60,
       f"the sink meter hears it after the loopback (sink {r['sink']} dBFS, "
       f"bus {round(r['bus'], 1)})")
    ok(r["paths"].get("loopback", 0) > 0 and "direct" not in r["paths"],
       f"on the loopback path ({r['paths']})")
    st = page.evaluate(STATE)
    ok((st["stats"] or {}).get("bus_fallbacks", 0) == 0
       and not any(e["kind"] == "fallback" for e in st["events"]),
       "and no fallback was recorded for a link that never failed")
    ok(not errors, "no page errors" + ("" if not errors else ": " + errors[0]))
    page.close()


def run_born_dead(browser, url, clip):
    section("born dead: negotiated, playing, and carrying nothing")
    page, errors = fresh(browser, url, "born_dead")
    page.evaluate(UNLOCK)
    # The proof budget is 3 s; give it room.
    page.wait_for_timeout(4500)
    st = page.evaluate(STATE)
    ok(st["total_samples"] == 0,
       "the sabotage reproduces the drive: the receiver counts zero samples "
       f"({st['total_samples']})")
    ok(st["covered"] is False,
       "the bus was NOT moved onto a link that carries nothing "
       f"(covered {st['covered']})")
    ok(st["to_destination"] is True,
       "it stayed on ctx.destination, where she can be heard "
       f"(to_destination {st['to_destination']})")
    fails = [e for e in st["events"] if e["kind"] == "loopback_failed"]
    ok(any((e["d"] or {}).get("at") == "no_samples" for e in fails),
       f"and the refusal names why: loopback_failed at=no_samples ({fails[:1]})")
    r = page.evaluate(PLAY, clip)
    ok(r["err"] is None and r["bus"] > -60, "a clip still plays")
    ok(r["paths"].get("direct", 0) > 0 and "loopback" not in r["paths"],
       f"and the sink reading says which path it is on: direct ({r['paths']})")
    rows = reports(page, "bus_loopback_failed")
    ok(any((x.get("detail") or {}).get("step") == "no_samples" for x in rows),
       f"the refusal reaches the drive log: bus_loopback_failed step=no_samples "
       f"({len(rows)} row(s)) -- wired at load, before any conversation")
    ok(not errors, "no page errors" + ("" if not errors else ": " + errors[0]))
    page.close()


def run_dies_later(browser, url, clip):
    section("dies later: a working loopback goes dead mid-drive")
    page, errors = fresh(browser, url, "none")
    r = page.evaluate(UNLOCK)
    ok(r["covered"] is True, "the loopback comes up healthy first")
    ok(page.evaluate(OPEN_SESSION), "RIO.realtime.connect exists to be stubbed")
    page.click("#mic")
    page.wait_for_timeout(1500)
    page.evaluate("() => window.__senders.forEach(s => s.replaceTrack(null))")
    before = page.evaluate(STATE)
    r = page.evaluate(TALK_UNTIL_FALLBACK, [clip, 12000])
    st = page.evaluate(STATE)
    ok(st["total_samples"] > before["total_samples"] or st["covered"] is False,
       "(the dead link was still counting samples -- no counter can see this)")
    ok(st["to_destination"] is True and st["covered"] is False,
       f"the bus fell back to ctx.destination while she was talking "
       f"(after {r['ms']} ms; covered {st['covered']}, "
       f"to_destination {st['to_destination']})")
    fb = [e for e in st["events"] if e["kind"] == "fallback"]
    ok(len(fb) >= 1 and (fb[0]["d"] or {}).get("why") == "sink_silent",
       f"a fallback event says why: sink_silent ({fb[:1]})")
    ok(bool(fb) and (fb[0]["d"] or {}).get("bus_db") is not None
       and (fb[0]["d"] or {}).get("sink_db") is not None,
       "and carries both meters at the moment it fired")
    ok((st["stats"] or {}).get("bus_fallbacks", 0) >= 1,
       f"stats count it (bus_fallbacks {(st['stats'] or {}).get('bus_fallbacks')})")
    ok((st["fallback"] or {}).get("why") == "sink_silent",
       f"state() says the bus is in fallback, and why ({st['fallback']})")
    rows = reports(page, "bus_fallback")
    d = (rows[0].get("detail") or {}) if rows else {}
    ok(len(rows) == 1 and d.get("why") == "sink_silent"
       and d.get("bus_db") is not None and d.get("sink_db") is not None,
       f"the fallback reaches the drive log with why and both meters ({d.get('why')}, "
       f"bus {d.get('bus_db')}, sink {d.get('sink_db')})")

    # What the barge gate does while the canceller is bypassed: the rows it
    # emits carry the path, and the window counts them.
    ok(page.evaluate(GATE_EVENTS), "the gate's events go through the live session's hook")
    b = reports(page, "barge_detected")
    e = reports(page, "echo_suppressed")
    ok(bool(b) and (b[-1].get("detail") or {}).get("bus_path") == "direct"
       and bool(e) and (e[-1].get("detail") or {}).get("bus_path") == "direct",
       "a barge_detected / echo_suppressed during the fallback says bus_path=direct")

    r = page.evaluate(PLAY, clip)
    ok(r["err"] is None and r["bus"] > -60 and r["paths"].get("direct", 0) > 0
       and "loopback" not in r["paths"],
       f"she is audible, on the direct path only, during the fallback ({r['paths']})")

    # The dead sender belongs to the dropped link; a rebuild makes new ones.
    # rebuild_ms (6 s, from the fallback) + the 3 s proof budget + slack.
    page.wait_for_timeout(6000 + 1500)
    st = page.evaluate(STATE)
    ok(st["covered"] is True and st["to_destination"] is False,
       f"the loopback was rebuilt, proved, and the bus moved back onto it "
       f"(covered {st['covered']})")
    rs = [e for e in st["events"] if e["kind"] == "restored"]
    ok(len(rs) == 1 and (rs[0]["d"] or {}).get("why") == "sink_silent"
       and (rs[0]["d"] or {}).get("down_ms", 0) > 0,
       f"a restored event closes the fallback, with how long it lasted ({rs[:1]})")
    ok(st["fallback"] is None, "and state() no longer reports a fallback")
    rows = reports(page, "bus_restored")
    w = ((rows[0].get("detail") or {}).get("fallback_window") or {}) if rows else {}
    ok(len(rows) == 1 and w.get("barge_detected") == 1
       and w.get("echo_suppressed") == 1
       and (w.get("cutoffs") or {}).get("false_barge_in") == 1,
       f"bus_restored carries what the barge gate did while it lasted ({w})")
    page.evaluate("() => window.__last.onEvent({ type: 'LIVE_BARGE_IN' })")
    b = reports(page, "barge_detected")
    ok((b[-1].get("detail") or {}).get("bus_path") == "loopback",
       "and a barge after the restore says bus_path=loopback")
    r = page.evaluate(PLAY, clip)
    ok(r["sink"] is not None and r["sink"] > -60
       and r["paths"].get("loopback", 0) > 0,
       f"and the sink meter hears her through the new link (sink {r['sink']})")
    ok(not errors, "no page errors" + ("" if not errors else ": " + errors[0]))
    page.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888/")
    args = ap.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed")
        return 2
    clip = "/static/audio/too_close.mp3"
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=[
            "--no-sandbox",
            "--autoplay-policy=no-user-gesture-required",
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
        ])
        run_healthy(browser, args.url, clip)
        run_born_dead(browser, args.url, clip)
        run_dies_later(browser, args.url, clip)
        browser.close()

    print("\n" + "=" * 72)
    print(f"{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.exit(main())
