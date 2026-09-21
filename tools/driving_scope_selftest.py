"""driving_scope_selftest.py — RIO.driving means two things and governs nothing else.

    python tools/driving_scope_selftest.py

WHAT IT MEANS NOW, and it is a narrower claim than it has ever been:

    1. THIS SESSION IS BEING LOGGED. There is a session id, so a mark or a
       heartbeat has something to be written against.
    2. THE LIVE CAMERA IS FEEDING, and the drive therefore OWNS it -- which is
       why a second loop must stand down rather than upload the same road
       twice.

It does NOT mean "vision is on", "headway is on" or "the voice works". Every
one of those ran off `if (!RIO.driving) return` and every one of them was
therefore off on 2026-09-21, while a clip played, boxes were drawn, the
detector tracked three cars and the observer read the road once a second. The
card said FRAMES 0 / SESSION — / Standby. Nothing was broken. Nobody had
pressed a button that half the page had been wired to.

This file is the audit, kept. Every remaining read of RIO.driving in
static/index.html must be one of the two meanings above, and the list below
names which -- so the next one added has to be classified by somebody rather
than appearing quietly.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = (REPO / "static" / "index.html").read_text()

checks = 0
failures = 0


def ok(name, cond, extra=""):
    global checks, failures
    checks += 1
    if not cond:
        failures += 1
    print(("  ok   " if cond else "  FAIL ") + name + (f" — {extra}" if extra else ""))


# EVERY READ OF RIO.driving IN THE PAGE, CLASSIFIED. This is the audit, kept.
#
# A read that is not in this table fails the suite, which is the point: the
# next one somebody adds has to be justified as one of the two meanings rather
# than appearing quietly, because appearing quietly is exactly how a logging
# flag became the switch for vision, headway and half the status line.
#
# Keyed by the line's own text (whitespace-normalised). Comments and CSS
# selectors are not reads and are skipped.
LOGGED = "logged"          # there is a session id to write against
OWNS = "owns"              # the drive's camera, and who stands down for it
UI = "ui"                  # saying which of the two is true, on the glass

AUDIT = {
    # --- the drive OWNS the camera it opened; others stand down --------------
    "if (RIO.driving || liveFrames.timer) return false;": OWNS,
    "if (!RIO.driving) {": OWNS,
    "if (!RIO.driving) { try { RIO.ui.feedLabel(null); } catch (e) {} }": OWNS,
    "if (RIO.source.isClip() && !RIO.driving) return null;": OWNS,
    "if (!RIO.driving && !liveFrames.timer) return null;": OWNS,
    "if (RIO.driving) return;": OWNS,
    "if (RIO.driving) return;              // live stream owns the box mid-drive": OWNS,
    "if (blob && !RIO.driving) {": OWNS,
    "if (RIO.driving) return;           // a live drive owns the capture loop": OWNS,
    # --- this session is being LOGGED ---------------------------------------
    "if (!RIO.driving || !RIO.speak) return;": LOGGED,
    "if (!RIO.driving) return;": LOGGED,
    "if (!RIO.driving) return true;": LOGGED,
    "if (RIO.driving) scheduleHeartbeat();": LOGGED,
    "if (j && j.stale && RIO.driving) { await staleStop(); return; }": LOGGED,
    "if (document.visibilityState !== 'visible' || !RIO.driving) return;": LOGGED,
    "if (RIO.driving && navigator.sendBeacon) {": LOGGED,
    "mark('DRIVE_START_FAILED', { error: why.slice(0, 200), driving: !!RIO.driving });": LOGGED,
    "driving: !!RIO.driving });": LOGGED,
    # --- and saying which of the two is true --------------------------------
    "+ (eyes || RIO.driving ? '' : ' (no camera — she cannot see)'));": UI,
    "return RIO.driving ? 'Drive active' : 'Live feed';": UI,
    "return RIO.driving ? ' - session ' + RIO.sessionId.slice(0, 8)": UI,
    "RIO.ui.feedLabel((RIO.driving ? 'drive · ' : 'live · ')": UI,
    "if (RIO.driving) await endDrive(); else await startDrive();": UI,
    # The clip replay writes the status line only when no drive is writing it.
    "if (st && !RIO.driving) {": UI,
}

reads = []
for i, ln in enumerate(PAGE.splitlines(), 1):
    if "RIO.driving" not in ln:
        continue
    t = ln.strip()
    # Prose, not code. A comment that quotes the flag is documentation of this
    # very change and must not be audited as a use of it.
    if t.startswith(("*", "//", "/*")) or "`" in t:
        continue
    reads.append((i, " ".join(t.split()) if False else t))

print("\n== 1. every read of RIO.driving is one of the two meanings ==\n")
unclassified = [(n, t) for n, t in reads if t not in AUDIT]
ok(f"all {len(reads)} live reads are classified", not unclassified,
   "; ".join(f"line {n}: {t[:60]}" for n, t in unclassified[:4]))
by_kind = {}
for n, t in reads:
    by_kind.setdefault(AUDIT.get(t, "?"), []).append(n)
for kind, label in ((LOGGED, "this session is being logged"),
                    (OWNS, "the drive owns the camera it opened"),
                    (UI, "saying which of the two is true")):
    print(f"       {len(by_kind.get(kind, [])):>2}  {label}")
ok("...and none of them governs a capability",
   not any(k not in (LOGGED, OWNS, UI) for k in by_kind))
# Stale entries are as bad as missing ones: a table nobody prunes stops being
# an audit and becomes a wish.
stale = [t for t in AUDIT if t not in {t2 for _, t2 in reads}]
ok("the table has no entries for lines that are gone", not stale,
   "; ".join(t[:50] for t in stale[:3]))

print("\n== 2. the loops that were gated on it no longer are ==\n")
for fn, what in [
    ("async function captureOnce", "the sensor reading (/perceive)"),
    ("function schedule()", "...and its rescheduling"),
    ("function scheduleHeadway", "the headway fallback loop"),
]:
    i = PAGE.index(fn)
    body = PAGE[i:i + 1400]
    ok(f"{what} asks feedingFrom(), not RIO.driving",
       "feedingFrom()" in body and "!RIO.driving" not in body.split("\n\n")[0])

ok("...and there is one predicate they all share",
   PAGE.count("function feedingFrom()") == 1)
ok("...which refuses a clip only while no drive owns it",
   "RIO.source.isClip() && !RIO.driving" in PAGE)
ok("...and refuses a camera nothing has opened",
   "!RIO.driving && !liveFrames.timer" in PAGE)

print("\n== 3. capture follows the picture, not the button ==\n")
ok("a live conversation opening a feed starts capture",
   "RIO.capture.sync('live_frames_start')" in PAGE)
ok("...and closing it re-asks rather than assuming",
   "RIO.capture.sync('live_frames_stop')" in PAGE)
ok("the headway toggle acts wherever there is a picture",
   "RIO.capture.sync('headway_toggle')" in PAGE)
ok("startDrive re-asks rather than starting loops itself",
   "syncCapture('drive_start')" in PAGE
   and "schedule();\n    startHeadway();" not in PAGE)
ok("endDrive stops what capture owns, and keeps the transport stats",
   "const tstats = stopCapture('drive_end');" in PAGE)
ok("the loops are idempotent, so every caller may simply re-ask",
   "if (timer) return;                 // one loop" in PAGE)

print("\n== 4. the status line stops claiming a drive that is not running ==\n")
ok("the feed word is derived, not a literal",
   "function feedWord()" in PAGE
   and PAGE.count("'Drive active - ' + frames") == 0)
ok("...and an unlogged feed says so",
   "not logged (no drive)" in PAGE)

print("\n== 5. and she is told which picture she is looking at ==\n")
sys.path.insert(0, str(REPO))
import rio_prompts as rp                                  # noqa: E402
clip = rp.reading_caveats({"source": "clip", "truncated": False,
                           "stripped": [], "fields": []})
cam = rp.reading_caveats({"source": "camera", "truncated": False,
                          "stripped": [], "fields": []})
ok("a clip is named as a clip", "UPLOADED CLIP" in clip)
ok("...and she is forbidden from calling it the road ahead",
   "never speak about it as what is ahead" in clip)
ok("a camera is named as the road ahead right now", "road ahead right now" in cam)
ok("...and the two are different sentences", clip != cam)
obs = (REPO / "observer.py").read_text()
ok("the source comes off the frame's own origin, not from the drive",
   'origin.rsplit(":", 1)[-1]' in obs)

print(f"\n{checks - failures}/{checks} checks passed")
sys.exit(1 if failures else 0)
