"""Phone-layout acceptance for the driving surface.

    python -m tools.mobile_layout_selftest
    python -m tools.mobile_layout_selftest --url http://127.0.0.1:8888/
    python -m tools.mobile_layout_selftest --shot /tmp/rio_390x844.png

Headless Chromium at 390x844 -- an iPhone in portrait, which is where the bug
this test exists for was found: the talk control was 42dvh of full-width fixed
viewport, so on a phone it covered the camera box entirely. The picture is the
one thing on this page a driver is looking at, and nothing may sit on top of
it.

WHAT IS ACTUALLY ASSERTED. Geometry, read out of a real browser, in the real
stylesheet, in the state that produced the bug (body.driving, a live session
open). Two rules:

  1. No control intersects the camera box. Not the talk button, not the bar it
     sits in, not anything else fixed to the viewport. Checked in every state
     the control has, because a caption change resizes it.

  2. Nothing in the feed's own top-right corner overlaps anything else there.
     The fullscreen button and the source badge were positioned independently
     with hand-measured offsets; iOS renders the button taller than the desk
     browser they were measured in, and the two collided.

Plus the two numbers a phone makes non-negotiable: a 56px minimum target, and
a bottom inset the home indicator does not cut into.

Requires playwright (`pip install playwright && playwright install chromium`).
The page is served, not opened from disk, so the stylesheet and scripts are
exactly what a phone would get.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []

PORTRAIT = (390, 844)          # iPhone 14/15/16, portrait, CSS pixels
LANDSCAPE = (844, 390)
MIN_TARGET_PX = 56             # the floor this test was asked for
MIN_INSET_PX = 14              # gutter when env(safe-area-inset-*) is 0

# The states the talk control can be in. Each one is a different caption and so
# a different box; the one that used to be a full-screen panel is 'listening'.
MIC_STATES = ["idle", "connecting", "listening", "speaking", "recording"]


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(("  ok    " if cond else "  FAIL  ") + what)


def section(name):
    print(f"\n=== {name} ===")


def intersects(a, b):
    """Two DOMRects, as dicts. Touching edges is not intersecting."""
    if not a or not b:
        return False
    if a["width"] <= 0 or a["height"] <= 0 or b["width"] <= 0 or b["height"] <= 0:
        return False
    return not (a["x"] + a["width"] <= b["x"] or b["x"] + b["width"] <= a["x"] or
                a["y"] + a["height"] <= b["y"] or b["y"] + b["height"] <= a["y"])


def overlap(a, b):
    """How much they overlap, in px, for the failure message."""
    w = min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])
    h = min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"])
    return (round(max(0.0, w), 1), round(max(0.0, h), 1))


# ---------------------------------------------------------------------------
# Page setup. A drive is simulated by the class the stylesheet keys off, which
# is the same switch startDrive() throws -- no camera, no session, no server
# state, so this runs anywhere and tests exactly the layout in question.
# ---------------------------------------------------------------------------
SET_STATE = """
(state) => {
  document.body.classList.add('driving');
  // The page's own state machine, not a copy of it: the captions, the aria
  // label and the class are whatever the real control would show, so the box
  // measured below is the size the real text makes it.
  if (!window.RIO || !RIO.ui || typeof RIO.ui.talkState !== 'function') {
    throw new Error('RIO.ui.talkState is not exposed — the control cannot be driven');
  }
  RIO.ui.talkState(state);
  return document.getElementById('mic').dataset.state;
}
"""

RECTS = """
() => {
  const box = (sel) => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    return { x: r.x, y: r.y, width: r.width, height: r.height,
             position: cs.position, display: cs.display,
             pointerEvents: cs.pointerEvents,
             paddingBottom: parseFloat(cs.paddingBottom) || 0,
             paddingRight: parseFloat(cs.paddingRight) || 0 };
  };
  // Everything pinned to the viewport, whatever it is: a fixed element that
  // lands on the feed is the bug regardless of which control it belongs to.
  const fixed = [];
  document.querySelectorAll('body *').forEach(el => {
    const cs = getComputedStyle(el);
    if (cs.position !== 'fixed') return;
    if (cs.display === 'none' || cs.visibility === 'hidden') return;
    if (parseFloat(cs.opacity || '1') < 0.05) return;
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return;
    // A positioner that takes no taps and paints nothing is not a control;
    // what matters is whether anything visible inside it lands on the feed,
    // and its children are in this same sweep.
    if (cs.pointerEvents === 'none' && cs.backgroundImage === 'none' &&
        (cs.backgroundColor === 'rgba(0, 0, 0, 0)' || cs.backgroundColor === 'transparent')) return;
    fixed.push({ what: el.id ? '#' + el.id : (el.className || el.tagName).toString().split(' ')[0],
                 x: r.x, y: r.y, width: r.width, height: r.height });
  });
  return {
    feed: box('.cam-wrap'),
    mic: box('#mic'),
    micbar: box('#micbar'),
    camfull: box('#camfull'),
    camsource: box('#camsource'),
    camtr: box('.cam-tr'),
    body: box('body'),
    scrollHeight: document.documentElement.scrollHeight,
    fixed: fixed,
  };
}
"""


SCROLL_SWEEP = """
(step) => {
  const feed = document.querySelector('.cam-wrap');
  const mic = document.getElementById('mic');
  const hit = (a, b) => !(a.x + a.width <= b.x || b.x + b.width <= a.x ||
                          a.y + a.height <= b.y || b.y + b.height <= a.y);
  const max = document.documentElement.scrollHeight - window.innerHeight;
  let found = null;
  for (let y = 0; y <= Math.max(0, max) + step; y += step) {
    window.scrollTo(0, y);
    const f = feed.getBoundingClientRect(), m = mic.getBoundingClientRect();
    if (hit(f, m)) { found = Math.round(window.scrollY); break; }
  }
  window.scrollTo(0, 0);
  return { found: found, max: Math.round(max) };
}
"""


TRANSITION_MS = 200     # #mic transitions background/border over 120ms


def measure(page, state, settle=False):
    got = page.evaluate(SET_STATE, state)
    if got != state:
        ok(False, f"[{state}] the control accepted the state (got {got!r})")
    # Geometry is immediate; colour is not. Only the colour checks pay for the
    # wait.
    if settle:
        page.wait_for_timeout(TRANSITION_MS)
    return page.evaluate(RECTS)


# ---------------------------------------------------------------------------
# A. The control never covers the picture
# ---------------------------------------------------------------------------
def run_control_vs_feed(page, size, label):
    section(f"A. talk control vs the camera box — {label} {size[0]}x{size[1]}")
    page.set_viewport_size({"width": size[0], "height": size[1]})

    for state in MIC_STATES:
        r = measure(page, state)
        feed, mic, bar = r["feed"], r["mic"], r["micbar"]
        if not feed or not mic:
            ok(False, f"[{state}] the feed and the control both exist")
            continue
        ok(not intersects(mic, feed),
           f"[{state}] the button does not intersect the feed"
           + ("" if not intersects(mic, feed) else f" (overlap {overlap(mic, feed)}px)"))
        ok(not intersects(bar, feed),
           f"[{state}] the bar it sits in does not intersect the feed either")
        ok(mic["height"] >= MIN_TARGET_PX,
           f"[{state}] the button is at least {MIN_TARGET_PX}px tall "
           f"(is {round(mic['height'], 1)}px)")
        # Compact: a button, not a panel. Anything over about a third of the
        # viewport is the shape of the bug this test exists for.
        ok(mic["height"] <= size[1] * 0.34,
           f"[{state}] the button is compact — {round(mic['height'], 1)}px "
           f"of a {size[1]}px viewport")
        ok(mic["width"] < size[0],
           f"[{state}] it does not span the full width of the viewport")

    # Every other fixed thing on the page, in the state that opens the most of
    # them (a live session, driving).
    r = measure(page, "listening")
    feed = r["feed"]
    hits = [f["what"] for f in r["fixed"] if intersects(f, feed)]
    ok(not hits, "no fixed-position element lands on the feed"
       + (f" (found: {', '.join(sorted(set(hits)))})" if hits else ""))

    # The corner it was asked for.
    mic = r["mic"]
    ok(mic["x"] + mic["width"] <= size[0] + 1
       and size[0] - (mic["x"] + mic["width"]) <= 24
       and size[1] - (mic["y"] + mic["height"]) <= 24,
       "the control is anchored in the bottom-right corner of the viewport")

    # Not overlapping at scroll 0 is not the same as not overlapping. What
    # makes it hold at EVERY scroll offset is one of two separations, and which
    # one it is depends on the shape of the viewport:
    #
    #   vertical   — the feed's bottom is above the control at scroll 0, and
    #                scrolling can only carry the feed further up. Portrait.
    #   horizontal — the control has a column of its own beside the feed, so
    #                the scroll offset is irrelevant. Landscape, where a 16:9
    #                box is taller than the screen and vertical separation is
    #                not available at any offset.
    vertical = feed["y"] + feed["height"] <= mic["y"]
    horizontal = (feed["x"] + feed["width"] <= mic["x"] or
                  mic["x"] + mic["width"] <= feed["x"])
    ok(vertical or horizontal,
       "feed and control are separated along a whole axis — "
       + ("the feed ends above the control and can only scroll upward"
          if vertical else
          "the control has its own column beside the feed")
       + " — so no scroll offset can bring them together")

    # And then walk the scroll and check it, because an argument about geometry
    # is worth exactly what the sweep that agrees with it is worth.
    swept = page.evaluate(SCROLL_SWEEP, 40)
    ok(swept["found"] is None,
       f"swept the whole scroll range (0..{swept['max']}px in 40px steps): the "
       "control never meets the feed"
       + ("" if swept["found"] is None else
          f" — collides at scrollY {swept['found']}"))


# ---------------------------------------------------------------------------
# B. The feed's own top-right corner
# ---------------------------------------------------------------------------
def run_badges(page, size, label):
    section(f"B. badge collision in the feed's top-right — {label}")
    page.set_viewport_size({"width": size[0], "height": size[1]})
    r = measure(page, "listening")
    full, src, feed = r["camfull"], r["camsource"], r["feed"]

    ok(full and src, "the fullscreen button and the source badge both render")
    if not (full and src):
        return
    ok(not intersects(full, src),
       "FULL and the source badge do not overlap"
       + ("" if not intersects(full, src) else f" (overlap {overlap(full, src)}px)"))
    ok(src["y"] >= full["y"] + full["height"] - 0.5,
       "the source badge sits below the fullscreen button, not across it")
    for name, b in (("FULL", full), ("the source badge", src)):
        inside = (b["x"] >= feed["x"] - 0.5 and b["y"] >= feed["y"] - 0.5 and
                  b["x"] + b["width"] <= feed["x"] + feed["width"] + 0.5 and
                  b["y"] + b["height"] <= feed["y"] + feed["height"] + 0.5)
        ok(inside, f"{name} stays inside the camera box")

    # The badge is written from the source label, and a clip's filename is not
    # short. It must still not push into the button above it.
    page.evaluate("""() => {
      const el = document.getElementById('camsource');
      if (el) { el.textContent = 'CLIP · DASHCAM_2026_09_07_LONG_NAME.MP4'; }
    }""")
    r2 = page.evaluate(RECTS)
    ok(not intersects(r2["camfull"], r2["camsource"]),
       "a long source label still does not reach the fullscreen button")
    ok(r2["camsource"]["x"] >= r2["feed"]["x"] - 0.5,
       "a long source label is clipped inside the box rather than escaping it")
    page.evaluate("""() => {
      const el = document.getElementById('camsource');
      if (el) el.textContent = 'REAR CAM';
    }""")

    # The other corner is the LIVE badge and the headway HUD; they were already
    # arranged around each other, and the stack must not have reached them.
    hud = page.evaluate("""() => {
      const hud = document.getElementById('headwayhud');
      const tr = document.querySelector('.cam-tr');
      if (!hud || !tr) return null;
      hud.style.display = 'flex';           // as body.headway-on would
      const a = hud.getBoundingClientRect(), b = tr.getBoundingClientRect();
      hud.style.display = '';
      return { a: {x: a.x, y: a.y, width: a.width, height: a.height},
               b: {x: b.x, y: b.y, width: b.width, height: b.height} };
    }""")
    ok(hud and not intersects(hud["a"], hud["b"]),
       "the headway HUD on the left still clears the right-hand stack")


# ---------------------------------------------------------------------------
# C. Safe areas
# ---------------------------------------------------------------------------
def run_safe_area(page, size, html):
    section("C. safe-area insets")
    page.set_viewport_size({"width": size[0], "height": size[1]})
    r = measure(page, "listening")

    # Headless Chromium reports every inset as 0, so the computed value proves
    # the fallback arm of max() and the source proves the env() arm. Both have
    # to be right: the fallback is what a desk browser gets, the env() is what
    # clears the home indicator.
    ok(r["micbar"]["paddingBottom"] >= MIN_INSET_PX,
       f"the control keeps a {MIN_INSET_PX}px bottom gutter with no inset "
       f"reported (is {round(r['micbar']['paddingBottom'], 1)}px)")
    ok(r["micbar"]["paddingRight"] >= MIN_INSET_PX,
       "and the same gutter on the right")

    block = re.search(r"body\.driving #micbar \{(.*?)\}", html, re.S)
    body = block.group(1) if block else ""
    ok("env(safe-area-inset-bottom" in body,
       "#micbar's bottom padding is max(gutter, env(safe-area-inset-bottom))")
    ok("env(safe-area-inset-right" in body,
       "#micbar's right padding honours the right inset too")
    ok(re.search(r"body\.driving \{[^}]*env\(safe-area-inset-bottom", html, re.S) is not None,
       "the page reserves the inset at the foot of the document as well, so "
       "the last column is not left under the control")
    ok("viewport-fit=cover" in html,
       "the viewport meta opts into the inset area in the first place")

    # The button's own bottom edge, with the reported (zero) inset, still clears
    # the viewport floor by the gutter.
    mic = r["mic"]
    ok(size[1] - (mic["y"] + mic["height"]) >= MIN_INSET_PX - 0.5,
       "the button's bottom edge clears the viewport floor by the gutter")


# ---------------------------------------------------------------------------
# D. The state the control is in is visible
# ---------------------------------------------------------------------------
def run_states(page, size):
    section("D. the control shows which state it is in")
    page.set_viewport_size({"width": size[0], "height": size[1]})
    seen = {}
    for state in MIC_STATES:
        page.evaluate(SET_STATE, state)
        page.wait_for_timeout(TRANSITION_MS)
        seen[state] = page.evaluate("""() => {
          const mic = document.getElementById('mic');
          const dot = mic.querySelector('.mic-dot');
          const hint = document.getElementById('michint');
          const cs = getComputedStyle(mic);
          return {
            state: mic.dataset.state,
            label: (document.getElementById('miclabel') || {}).textContent || '',
            hint: getComputedStyle(hint).display === 'none' ? '' : hint.textContent,
            dot: getComputedStyle(dot).backgroundColor,
            border: cs.borderTopColor,
            aria: mic.getAttribute('aria-label') || '',
          };
        }""")
    for state in MIC_STATES:
        ok(seen[state]["state"] == state and seen[state]["label"].strip() != "",
           f"[{state}] the button carries the state and says something")
    colours = {s: seen[s]["dot"] for s in MIC_STATES}
    ok(len(set(colours.values())) == len(MIC_STATES),
       "each state has its own dot colour — they are told apart at a glance")
    ok(seen["listening"]["border"] != seen["speaking"]["border"],
       "listening and speaking are not the same colour")
    for s in ("listening", "speaking"):
        ok("end" in seen[s]["hint"].lower(),
           f"[{s}] the control says a tap ends the conversation")
        ok("end the conversation" in seen[s]["aria"].lower(),
           f"[{s}] and says so to a screen reader")
    ok(seen["idle"]["hint"] == "",
       "idle carries no end hint — there is nothing to end")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888/",
                    help="served page to test (default: the local uvicorn)")
    ap.add_argument("--shot", default=None, help="write a PNG of the portrait layout")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed:\n"
              "  pip install playwright && python -m playwright install chromium")
        return 2

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": PORTRAIT[0], "height": PORTRAIT[1]},
                                device_scale_factor=3, is_mobile=True,
                                has_touch=True)
        page.on("console", lambda m: None)
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=20000)
        except Exception as e:
            print(f"could not load {args.url}: {e}")
            browser.close()
            return 2
        page.wait_for_timeout(600)          # fonts, and the DOMContentLoaded wiring
        html = page.content()

        run_control_vs_feed(page, PORTRAIT, "portrait")
        run_badges(page, PORTRAIT, "portrait")
        run_safe_area(page, PORTRAIT, html)
        run_states(page, PORTRAIT)
        run_control_vs_feed(page, LANDSCAPE, "landscape")

        if args.shot:
            page.set_viewport_size({"width": PORTRAIT[0], "height": PORTRAIT[1]})
            measure(page, "listening")
            page.screenshot(path=args.shot)
            print(f"\nwrote {args.shot}")
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
