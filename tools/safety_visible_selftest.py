"""safety_visible_selftest.py — a safety layer that cannot conclude says so.

    python tools/safety_visible_selftest.py

TWO FAULTS FROM ONE DRIVE, and they share a shape: something was wrong for the
whole run, the page had the information, and nothing about the screen said the
consequence.

THE BAND NEVER LEFT UNKNOWN. 2174 frames of 2174 carried `band: UNKNOWN` and
`voice_reason: suppressed_by_unknown_speed`. The detector tracked 19,433
objects, the depth model ran, a lead gap was computed on 1034 frames -- and not
one word could ever have been spoken, because time-to-contact is a gap over a
speed and there was no speed. The HUD said "no speed" in mono on the fourth
line of a five-line block. That is a fact about a FIELD. What a driver needs is
the consequence: the warnings are off.

THE GPS WATCHDOG CHURNED THE RADIO 33 TIMES IN 167 SECONDS. Every one of those
marks read `attempt: 1`, which is what a counter that never gets anywhere looks
like: a desktop wifi fix arrives about every five seconds, the watchdog's
four-second threshold fired between every pair, the watch was torn down and
rebuilt, the rebuild returned a cached fix, and the counter reset. The radio
was working. The threshold was wrong for it.
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


print("\n== 1. the banner exists and says the consequence ==\n")
ok("there is a node for it, empty in the markup",
   'id="hwinert" class="hw-inert" hidden' in PAGE)
ok("it is painted from the headway result, not from a timer of its own",
   "paintInert(j);" in PAGE and "function paintInert(j)" in PAGE)
ok("it reads the SERVER's reason, not the page's guess about speed",
   "voice_reason" in PAGE.split("function paintInert")[1][:1200]
   and "suppressed_by_unknown_speed" in PAGE)
body = PAGE.split("function paintInert")[1][:3000]
ok("...it names the consequence, not the field",
   "warnings are OFF" in body, "'no speed' alone is what was already there")
ok("...it gives the one action that turns them on",
   "Set a speed override" in body and "allow Location" in body.replace("\\u2014", ""))
ok("...and says which of the two situations this is",
   "isClip" in body)
ok("it waits before appearing, so the first frames of a drive are quiet",
   "INERT_AFTER_S" in PAGE)
ok("...and it clears with the HUD when the feed stops",
   "inertSince = null;" in PAGE.split("elBand.textContent = 'Standby'")[1][:400])

print("\n== 2. the watchdog learns what this radio calls 'delivering' ==\n")
ok("there is a measured effective threshold",
   "function effectiveWatchdogS()" in PAGE)
ok("...built from the observed gaps between fixes", "fixGaps" in PAGE)
ok("...floored at the configured value, so a quiet radio still trips it",
   "Math.max(base, slowest * 2)" in PAGE)
ok("...and capped, so a pathological gap cannot switch it off",
   "Math.min(60," in PAGE)
ok("the watchdog compares against it rather than the constant",
   "if (quiet < effectiveWatchdogS()) return;" in PAGE
   and "if (quiet < watchCfg.watchdog_s) return;" not in PAGE)
ok("the rearm counter only clears on a fix that beat the threshold",
   "< effectiveWatchdogS()) {" in PAGE,
   "clearing on every fix is what made 33 rearms read 'attempt: 1'")
ok("...and a rearm mark records what it was measured against",
   "watchdog_s: Math.round(effective * 10) / 10" in PAGE
   and "fix_gaps_s" in PAGE)
ok("a fresh watch forgets the old radio's rhythm", "fixGaps = [];" in PAGE)

print("\n== 3. the arithmetic, on the drive's own numbers ==\n")


def effective(base, gaps, cap=60):
    if not gaps:
        return base
    return min(cap, max(base, max(gaps) * 2))


# The drive: fixes about every 4.7-5.1 s against a 4.0 s watchdog.
DRIVE_GAPS = [4.7, 4.2, 5.1, 4.8, 4.7]
eff = effective(4.0, DRIVE_GAPS)
ok(f"a radio delivering every ~5 s gets a {eff:.1f} s watchdog, not 4.0",
   eff > max(DRIVE_GAPS), f"{eff:.1f} s")
ok("...so none of the drive's 33 rearms would have fired",
   all(g < eff for g in DRIVE_GAPS))
ok("a radio that genuinely stops still trips it",
   30.0 >= effective(4.0, DRIVE_GAPS), "30 s of silence is past 10.2 s")
ok("with no measurements yet it is exactly what it always was",
   effective(4.0, []) == 4.0)
ok("one pathological gap cannot switch the watchdog off",
   effective(4.0, [600.0]) == 60.0)

print(f"\n{checks - failures}/{checks} checks passed")
sys.exit(1 if failures else 0)
