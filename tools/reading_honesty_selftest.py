"""reading_honesty_selftest.py — a reading may not invent, and may not pretend
it finished.

    python tools/reading_honesty_selftest.py

TWO FAULTS FROM ONE PAYLOAD, both of which reached RIO on 2026-09-21 as what
the camera saw:

    "ROAD: four, two, asphalt, moderate |
     TRAFFIC: car 20 km/h, car 40 km/h, car 50 km/h |
     RISK: collision between car 20 km/h and car"

A single frame cannot show a speed -- there is no second frame to difference
against -- and the model produced three of them. And the reading stops dead
after "and car", because it hit the 48-token cap; there is no ellipsis, no
dangling bracket, nothing in the text that says so. A cut landing a word later
would have read as a completed "RISK: none".

Neither is fixed by re-wording the prompt. SENSOR_PROMPT_TERSE already forbids
both ("no speed or distance in numbers unless you can read them in the frame",
"the three fields only") and both happened anyway. A prompt is a request; these
are properties.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import rio_prompts as rp                                  # noqa: E402

checks = 0
failures = 0


def ok(name, cond, extra=""):
    global checks, failures
    checks += 1
    if not cond:
        failures += 1
    print(("  ok   " if cond else "  FAIL ") + name + (f" — {extra}" if extra else ""))


DRIVE = ("ROAD: four, two, asphalt, moderate | "
         "TRAFFIC: car 20 km/h, car 40 km/h, car 50 km/h | "
         "RISK: collision between car 20 km/h and car")

print("\n== 1. the reading that shipped ==\n")
clean, removed = rp.strip_measurements(DRIVE)
ok("all four invented speeds are removed", len(removed) == 4, str(removed))
ok("...and no unit survives anywhere in the reading",
   not any(u in clean for u in ("km/h", "mph", "m/s", " m,", " m ")), clean)
ok("...while what the model actually SAW is kept",
   "car, car, car" in clean, clean)
ok("...and the other fields are untouched",
   clean.startswith("ROAD: four, two, asphalt, moderate"), clean)
ok("...with the field structure intact", clean.count("|") == 2, clean)

print("\n== 2. a number is only a measurement when it carries a unit ==\n")
for t, why in [
    ("ROAD: four, two, asphalt", "lane counts"),
    ("TRAFFIC: two cars ahead", "a count of cars"),
    ("ROAD: motorway, 4 lanes", "a count with a noun"),
    ("TRAFFIC: car in lane 2", "a lane number"),
]:
    ok(f"left alone — {why}", rp.strip_measurements(t)[0] == t, t)
for t, unit in [
    ("TRAFFIC: sedan 20 m ahead", "metres"),
    ("TRAFFIC: lorry about 30 metres ahead", "spelled metres"),
    ("TRAFFIC: queue at 5-10 mph", "a range in mph"),
    ("TRAFFIC: van 100 ft back", "feet"),
    ("TRAFFIC: car doing 13 m/s", "m/s"),
    ("TRAFFIC: bus 0.5 km ahead", "km"),
]:
    clean_t, rm = rp.strip_measurements(t)
    ok(f"removed — {unit}", bool(rm) and unit.split()[0] not in clean_t,
       f"{t!r} -> {clean_t!r}")

print("\n== 3. the edit leaves readable English ==\n")
for before, after in [
    ("TRAFFIC: queue ahead at 5-10 mph", "TRAFFIC: queue ahead"),
    ("TRAFFIC: car 20 km/h, car 40 km/h", "TRAFFIC: car, car"),
    ("TRAFFIC: sedan 20 m ahead, van behind",
     "TRAFFIC: sedan ahead, van behind"),
]:
    got = rp.strip_measurements(before)[0]
    ok(f"no dangling punctuation or preposition: {before[:38]!r}",
       got == after, f"got {got!r}")

print("\n== 4. what RIO is told about a reading she cannot see the flaw in ==\n")
cut = {"truncated": True, "stripped": [],
       "fields": [{"name": "ROAD", "present": True, "text": "four, two"},
                  {"name": "TRAFFIC", "present": True, "text": "car, car"},
                  {"name": "RISK", "present": True, "text": "collision between car and",
                   "truncated": True}]}
rule = rp.reading_caveats(cut)
ok("a truncated reading is declared to her", "CUT OFF" in rule)
ok("...naming the field that is a fragment", "Its RISK field" in rule, rule[:90])
ok("...and forbidding the all-clear reading of it",
   "all-clear" in rule and "do not finish it" in rule.lower())
strip = {"truncated": False, "stripped": ["20 km/h"], "fields": []}
rule2 = rp.reading_caveats(strip)
ok("a stripped reading is declared too", "NO SPEEDS OR DISTANCES" in rule2)
ok("...and she is told not to supply the numbers herself",
   "not state or estimate" in rule2.lower().replace("do not", "not"))
ok("...and where a real distance comes from", "tracker" in rule2)
# ...AND EVERY READING CARRIES THE UNVERIFIED LABEL, clean or not. This check
# used to assert the opposite -- that a clean reading got no caveat -- and that
# was right while the reading was three constrained fields. It is wrong now.
# Asked a plain question, Cosmos sees the gist and invents specifics with
# complete fluency: "the speedometer shows 60 miles per hour" on a frame with
# no speedometer, makes and models at distances where no badge resolves, lane
# counts of seven and thirteen on a four-lane road. Nothing can separate the
# true half from the invented half of one sentence, so the whole thing is
# labelled rather than any part of it trusted.
clean = rp.reading_caveats({"truncated": False, "stripped": [], "fields": []})
ok("every reading is labelled an unverified impression", "UNVERIFIED" in clean)
ok("...naming what it is reliably right about", "GIST" in clean.upper())
ok("...and what it invents", "makes and models" in clean or "models" in clean)
ok("...forbidding any specific being stated as fact",
   "as though it were established" in clean)
ok("...and naming the thing that actually knows", "tracker" in clean)

print("\n== 5. the two facts are counted, not just prevented ==\n")
import vision                                             # noqa: E402
fr = vision.flag_rate()
for k in ("invented_units", "truncated"):
    ok(f"{k} is a counted flag with a rate", k in fr and k in (fr["rate"] or {}))
vsrc = (REPO / "vision.py").read_text()
ok("the strip runs on the live path, after every publish/refuse guard",
   vsrc.index("reading_refused(text, repeats)")
   < vsrc.index("text, stripped = strip_measurements(text)"))
ok("...and truncation is asked of the TOKENS, not of the text",
   "_reading_truncated(out, inputs" in vsrc and "out.shape[1]" in vsrc)
ok("...requiring BOTH the cap and a non-EOS last token",
   "if generated < cap" in vsrc and "not in _eos_ids()" in vsrc)

print(f"\n{checks - failures}/{checks} checks passed")
sys.exit(1 if failures else 0)
