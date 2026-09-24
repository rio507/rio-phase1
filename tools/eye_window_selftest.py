"""eye_window_selftest.py — the grounded video eye, checked without a GPU.

    python -m tools.eye_window_selftest

WHAT THIS DEFENDS, AND WHY IT IS NOT tools/eye_video_gates.py's JOB.

The gates run the real models over real footage and ask whether the READINGS
are good. That is slow, it needs a card, and it cannot be run on every change.
This asks the structural questions that a refactor gets wrong silently and
that no amount of good readings would reveal:

  1  THE WINDOW IS A VIDEO. Metadata carries the real fps and duration, and
     without it the processor silently resamples twenty-four frames down to
     two temporal groups -- a still in disguise, with every other field on the
     card still saying twenty-four. Measured, not asserted: see the note in
     eyewindow.Window.metadata.

  2  THE MEASURED BLOCK SAYS WHOSE NUMBERS THEY ARE. Every line names a source
     and no number is mangled on the way out. `_fmt(90.0, 0)` returned "9"
     once, which rendered 25 m/s as "(9 km/h)" and produced a confident,
     coherent reading built on a false premise. A formatting bug in a grounding
     block is not a formatting bug.

  3  A RESTATEMENT IS NOT AN INVENTION, AND AN INVENTION IS NOT A
     RESTATEMENT. The numbers rule has to let through a value we supplied --
     in any unit it could have been converted into -- and cut one we did not.
     Both directions, because a guard that cut everything would look identical
     in the counters to a guard that cut nothing.

  4  A CITATION IS CHECKED AGAINST THE MEASUREMENT, NOT JUST THE TRACK LIST.
     A reading that cites track 18 and says it is slowing, when track 18 was
     measured opening at 9.8 m/s, is the model using its grounding as
     decoration. That reading happened; it scored as corroborated until this
     check existed.

  5  THE ECHO GUARD DOES NOT EAT THE GROUNDING. Half the prompt is now
     measurements the model was explicitly told to repeat, so an echo check
     against the whole prompt would refuse every correctly grounded reading --
     the guard firing on exactly the behaviour it exists to encourage.

  6  NOTHING NAMES A MODEL IN A LITERAL, in the new card or the new modules.
     /health, the page and the drive log ask the role. This project has
     shipped that bug twice.

No GPU and no network: every check here is a decision, a string or a source
file.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

_checks, _fails = [], []


def ok(name, cond, extra=""):
    _checks.append(name)
    if cond:
        print(f"  ok   {name}")
    else:
        _fails.append(name)
        print(f"  FAIL {name}" + (f" -- {extra}" if extra else ""))


def section(n, title):
    print(f"\n{n}. {title}")


import config          # noqa: E402
import eyeread         # noqa: E402
import eyewindow       # noqa: E402
import grounding       # noqa: E402


# --- 1. the window is a video ----------------------------------------------
section(1, "THE WINDOW IS A VIDEO")

import numpy as np                                                # noqa: E402


class _FakeRF:
    def __init__(self, i, t):
        self.frame_id = f"f{i:06d}"
        self.wall_t = t
        self.objects = []
        self.measured = {}


_n = 24
# A MOVING PICTURE WITH SOMETHING IN IT. A flat colour per frame has a
# luminance sd of zero and is indistinguishable from a covered lens, which is
# how the first version of this fixture "proved" that a moving window is
# blank. Structure AND motion both have to be real for the negative case to
# mean anything.
_grad = np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (64, 1))
_frames = [np.dstack([np.roll(_grad, i * 4, axis=1)] * 3) for i in range(_n)]
_w = eyewindow.Window(_frames, [i * 0.25 for i in range(_n)],
                      [_FakeRF(i, 1000.0 + i * 0.25) for i in range(_n)], 4.0)

md = _w.metadata()[0]
ok("metadata carries the real fps", md["fps"] == 4.0)
ok("metadata carries the real frame count", md["total_num_frames"] == _n)
ok("metadata duration matches the span, not a guess",
   abs(md["duration"] - _w.span_s) < 0.51, f"{md['duration']} vs {_w.span_s}")
ok("span is first-to-last, not count/fps", abs(_w.span_s - 5.75) < 0.01,
   str(_w.span_s))

meta = _w.to_meta()
ok("the card is given a span and an end, not an age",
   "span_s" in meta and "end_age_s" in meta and "age_s" not in meta)
ok("provenance drops are reported", "dropped_unverified" in meta)
ok("MIN_FRAMES is at least the processor's own floor (4)",
   eyewindow.MIN_FRAMES >= 4)
ok("the pixel budget is NVIDIA's", eyewindow.TOTAL_PIXELS == 3774873)
ok("the default rate is NVIDIA's", eyewindow.DEFAULT_FPS == 4.0)
ok("the window never outruns the ring",
   eyewindow.DEFAULT_SECONDS <= config.RING_SECONDS,
   f"{eyewindow.DEFAULT_SECONDS} > {config.RING_SECONDS}")
ok("do_sample_frames is off — we thin, the processor does not",
   eyewindow.processor_kwargs().get("do_sample_frames") is False)

# A frozen window and a black window are FACTS about the frames, not opinions.
_still = eyewindow.Window([np.full((64, 96, 3), 120, dtype=np.uint8)] * 8,
                          [i * 0.25 for i in range(8)],
                          [_FakeRF(i, 1000.0 + i * 0.25) for i in range(8)], 4.0)
ok("a frozen window is labelled static", _still.to_meta()["static"] is True)
_black = eyewindow.Window([np.zeros((64, 96, 3), dtype=np.uint8)] * 8,
                          [i * 0.25 for i in range(8)],
                          [_FakeRF(i, 1000.0 + i * 0.25) for i in range(8)], 4.0)
ok("a black window is labelled blank", _black.to_meta()["blank"] is True)
ok("a moving window is neither", not _w.to_meta()["static"]
   and not _w.to_meta()["blank"])


# --- 2. the measured block ---------------------------------------------------
section(2, "THE MEASURED BLOCK SAYS WHOSE NUMBERS THEY ARE")

ok("_fmt does not eat trailing zeros of an integer",
   grounding._fmt(90.0, 0) == "90", grounding._fmt(90.0, 0))
ok("_fmt still trims a decimal", grounding._fmt(71.250, 1) == "71.2",
   grounding._fmt(71.250, 1))
ok("_fmt says 'unknown' rather than guessing", grounding._fmt(None) == "unknown")

STATE = {
    "window": {"span_s": 5.8, "n_frames": 24, "blank": False, "static": False},
    "ego": {"speed_ms": 25.0, "speed_source": "obd", "speed_first_ms": 25.0},
    "headway": {"gap_m": 18.4, "gap_first_m": 27.9, "ttc_s": 4.6,
                "band": "GETTING_UNSAFE", "plausibility": "accepted",
                "closing_ms": -2.1, "series": []},
    "lane": {"n_detected": 3, "plausible": [True, True, True], "conf": 0.98,
             "source": "ufld"},
    "tracks": [
        {"id": 18, "label": "truck", "last_range_m": 54.4, "first_range_m": 44.6,
         "range_rate_ms": 9.8, "side": "left of the ego lane", "held_s": 1.0,
         "n_frames": 5, "first_t": 2.0, "last_t": 3.0, "confirmed": True,
         "in_lane": False, "is_lead": False, "vulnerable": False,
         "range_reject": None, "samples": []},
        {"id": 6, "label": "car", "last_range_m": 21.7, "first_range_m": 20.9,
         "range_rate_ms": 0.3, "side": "in the ego lane", "held_s": 3.5,
         "n_frames": 15, "first_t": 2.2, "last_t": 5.8, "confirmed": True,
         "in_lane": True, "is_lead": True, "vulnerable": False,
         "range_reject": None, "samples": []},
    ],
}
text = grounding.render(STATE)
ok("the ego speed converts correctly", "(90 km/h)" in text, text.split("\n")[1])
ok("the ego speed says it is not a limit",
   "not a limit" in text and "another vehicle" in text)
ok("every line names a source",
   all("[source:" in ln for ln in text.split("\n")
       if ln and not ln.startswith(" ") and not ln.startswith("MEASURED")
       and not ln.startswith("TRACKS")),
   text)
ok("the gap carries its plausibility verdict", "plausibility: accepted" in text)
ok("the band is present", "GETTING_UNSAFE" in text)
ok("the time to contact is present", "4.6 s" in text)
ok("a track carries its id, range, side and direction",
   "track 18" in text and "54.4 m" in text
   and "left of the ego lane" in text and "opening at 9.8 m/s" in text)
ok("a track carries how long it was held and over what seconds",
   "held 1 s" in text and "2–3 s" in text, text)
ok("the lead is named as the lead", "this is the lead" in text)

# The measured block must never carry a model's words.
ok("nothing in measured_from_result comes from a model",
   not any(k in grounding.measured_from_result({"voice_line": "slow down",
                                                "observation": "a road"})
           for k in ("voice_line", "observation", "caption", "reading")))

blank_text = grounding.render(dict(STATE, window=dict(STATE["window"], blank=True,
                                                      structure_std=0.4)))
ok("a blank window is stated as measured fact", "CAMERA" in blank_text
   and "pixel statistics" in blank_text)
static_text = grounding.render(dict(STATE, window=dict(STATE["window"], static=True,
                                                       motion=0.02)))
ok("a static window is stated as measured fact",
   "not changing" in static_text and "pixel statistics" in static_text)


# --- 3. the numbers rule, both directions -----------------------------------
section(3, "A RESTATEMENT IS NOT AN INVENTION")

supplied = grounding.supplied_numbers(STATE)
ok("the supplied set contains the gap",
   any(abs(s["value"] - 18.4) < 1e-6 and s["family"] == "distance"
       for s in supplied))
ok("the supplied set contains the ttc",
   any(abs(s["value"] - 4.6) < 1e-6 and s["family"] == "time" for s in supplied))

clean, src, inv = eyeread.check_numbers(
    "The lead is 18.4 m ahead with 4.6 s to contact.", supplied)
ok("a restated gap is sourced and kept",
   len(src) == 2 and not inv and "18.4 m" in clean, f"{src} {inv} {clean}")

clean2, src2, inv2 = eyeread.check_numbers(
    "The lead is 40 m ahead.", supplied)
ok("an invented gap is cut and counted",
   len(inv2) == 1 and "[unmeasured]" in clean2 and "40 m" not in clean2,
   f"{inv2} {clean2}")

clean3, src3, inv3 = eyeread.check_numbers("We are doing 90 km/h.", supplied)
ok("a converted speed is a restatement, not an invention",
   len(src3) == 1 and not inv3, f"{src3} {inv3}")

clean4, src4, inv4 = eyeread.check_numbers("Three lanes, track 18 matters.",
                                           supplied)
ok("a lane count and a track id are not measurements",
   not inv4 and clean4 == "Three lanes, track 18 matters.", f"{inv4} {clean4}")

clean5, src5, inv5 = eyeread.check_numbers("The gap is 60 m.", [])
ok("with nothing supplied, every measurement is an invention", len(inv5) == 1)


# --- 4. a citation is checked against the measurement ------------------------
section(4, "A CITATION IS CHECKED AGAINST THE MEASUREMENT")

c = eyeread.corroborate("Track 18: truck slowing down on the far left.", STATE)
ok("a real track is counted as cited", c["cited"] == [18], str(c["cited"]))
ok("saying 'slowing' about a track measured opening is a contradiction",
   len(c["contradicts"]) == 1 and c["contradicts"][0]["id"] == 18,
   str(c["contradicts"]))
ok("a contradiction makes the whole reading unverified",
   c["verdict"] == "unverified")

c2 = eyeread.corroborate("Track 18 is pulling away on the left.", STATE)
ok("agreeing with the measurement is not a contradiction",
   not c2["contradicts"] and c2["verdict"] == "corroborated", str(c2))

c3 = eyeread.corroborate("Track ID 99 is critical.", STATE)
ok("a track we do not hold is a bad cite", c3["bad_cites"] == [99])

c4 = eyeread.corroborate("A cyclist is in the ego lane.", STATE)
ok("naming a class the detector holds none of is a fabrication",
   "cyclist" in c4["fabricated"], str(c4["fabricated"]))

c5 = eyeread.corroborate("A truck on the left and a car ahead.", STATE)
ok("naming classes we DO hold is not a fabrication", not c5["fabricated"],
   str(c5["fabricated"]))

c6 = eyeread.corroborate("Nothing here is critical for safe navigation.", STATE)
ok("an in-lane lead the reading never mentions is a miss",
   any(m["id"] == 6 for m in c6["missed"]), str(c6["missed"]))

c7 = eyeread.corroborate("Tracks 6 and 18 matter.", STATE)
ok("a multi-id citation is fully parsed", c7["cited"] == [6, 18],
   str(c7["cited"]))
c8 = eyeread.corroborate("Track IDs 6, 18, and 99 are critical.", STATE)
ok("a comma-and list is fully parsed",
   c8["cited"] == [6, 18] and c8["bad_cites"] == [99], str(c8))


# --- 5. the echo guard does not eat the grounding ----------------------------
section(5, "THE ECHO GUARD DOES NOT EAT THE GROUNDING")

instr = eyeread.instructions_only(text)
ok("the echo baseline excludes the measured block",
   "MEASURED STATE" not in instr and "track 18" not in instr)
# THE FORMAT IS ASKED OF THE ROLE, so the guard's baseline has to be too.
# This named COSMOS_FORMAT flat, which was a statement about which model
# happened to be resident rather than about the echo guard: the baseline must
# contain the instructions THAT WERE SENT, and under a non-reasoner those end
# with PLAIN_FORMAT. A baseline that carried the other model's format would
# have the guard watching for words the prompt never used while the words it
# did use went unwatched -- the same fault d3ffa27 found in the worked-example
# detector.
ok("...and still includes every instruction",
   eyeread.AV_COT in instr and eyeread.ASKS in instr
   and eyeread.NUMBERS_RULE in instr
   and eyeread.answer_format() in instr)

prompt = eyeread.build_prompt(text)
ok("the prompt is NVIDIA's question first", prompt.startswith(eyeread.AV_COT))
ok("the prompt carries the answer format for THIS role, last",
   prompt.rstrip().endswith(eyeread.answer_format()))

# ...AND THE RULE ITSELF, both ways, which is what the two checks above used to
# imply and no longer can. A non-reasoner asked for NVIDIA's shape returns the
# shape's CONTENTS: "Your reasoning." was the first line of all three window
# answers on 2026-09-24 (config.local_vision_reasons).
import importlib
import config as _cfg
_was = _cfg.LOCAL_VISION_MODEL
try:
    for _role, _want, _placeholder in (("cosmos", eyeread.COSMOS_FORMAT, True),
                                       ("qwen", eyeread.PLAIN_FORMAT, False)):
        _cfg.LOCAL_VISION_MODEL = _role
        _fmt = eyeread.answer_format()
        ok(f"...{_role} is asked for its own answer format", _fmt == _want)
        ok(f"...and the <think> placeholder "
           f"{'reaches' if _placeholder else 'never reaches'} it",
           ("Your reasoning." in eyeread.build_prompt(text)) is _placeholder)
finally:
    _cfg.LOCAL_VISION_MODEL = _was
ok("the prompt lets the model say nothing, for both questions",
   "it is a complete answer" in prompt and "nothing the driver needs" in prompt)
ok("the prompt contains no field template",
   not re.search(r"\bROAD:|\bTRAFFIC:|\bRISK:", prompt))
ok("the prompt contains no worked example",
   "->" not in prompt and "frame:" not in prompt)
ok("an ungrounded prompt carries no numbers rule",
   "Measured state" not in eyeread.build_prompt(""))

r, a, unterm = eyeread.split_reasoning("<think>weighing it up</think>The answer.")
ok("the trace is kept, not stripped", r == "weighing it up" and a == "The answer.")
ok("...and is not counted as a fault", "think_trace" not in eyeread._flags)
r2, a2, unterm2 = eyeread.split_reasoning("<think>never closed and no answer")
ok("an unterminated trace yields no answer", unterm2 and not a2 and r2)


# --- 6. nothing names a model ------------------------------------------------
section(6, "NOTHING NAMES A MODEL IN A LITERAL")

_BAD = re.compile(r"(cosmos-reason|Cosmos-Reason2-2B|qwen3-vl|Qwen3-VL|"
                  r"alpamayo|nvidia/)", re.I)
def _code_only(path):
    """Source with comments and docstrings removed. -> [lines].

    COMMENTS AND DOCSTRINGS MAY NAME ANYTHING -- that is where the reasoning
    about a particular checkpoint belongs, and stripping it from the prose
    would make these files unreadable in order to defend a rule about code.
    For .py this uses the tokenizer rather than doing it by hand: a
    hand-rolled docstring tracker got its own toggle wrong and reported this
    very module's header as executable code.
    """
    body = path.read_text()
    if path.suffix == ".py":
        import io
        import tokenize
        drop = set()
        prev_type = None
        for tok in tokenize.generate_tokens(io.StringIO(body).readline):
            if tok.type == tokenize.COMMENT:
                drop.update(range(tok.start[0], tok.end[0] + 1))
            elif tok.type == tokenize.STRING and prev_type in (
                    None, tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                    tokenize.DEDENT):
                # A string in statement position is a docstring.
                drop.update(range(tok.start[0], tok.end[0] + 1))
            if tok.type not in (tokenize.NL, tokenize.COMMENT):
                prev_type = tok.type
        return [ln for i, ln in enumerate(body.split("\n"), 1) if i not in drop]
    # JS: block comments and line comments.
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    return [ln for ln in body.split("\n") if not ln.strip().startswith("//")]


for rel in ("static/rio_eye.js", "eyeread.py", "eyewindow.py", "grounding.py"):
    hits = [ln.strip() for ln in _code_only(REPO / rel) if _BAD.search(ln)]
    ok(f"{rel} names no checkpoint in code", not hits, str(hits[:2]))

ok("the card asks the record for the model's name",
   "rec.model" in (REPO / "static/rio_eye.js").read_text())


# --- 7. scope: this path cannot speak or warn --------------------------------
section(7, "THE READING CANNOT SPEAK OR WARN")

body = (REPO / "eyeread.py").read_text() + (REPO / "eyewindow.py").read_text()
for banned in ("safety_speech", "voice.", "arbiter", "speak(", "policy.band",
               "set_band"):
    ok(f"the video path never touches {banned}", banned not in body)

obs = (REPO / "observer.py").read_text()
ok("the window record has no route into RIO's evidence",
   "window_record" in obs and "serve_to(state[\"window_record\"]" not in obs)


# --- 8. the arms, the rubric and the borrowing check -------------------------
section(8, "THE ARMS ARE A COMPARISON OF QUESTIONS")

import rubric                                                     # noqa: E402

ok("all five arms exist", set(eyeread.ARMS) == {"A", "B", "C", "D", "E"},
   str(sorted(eyeread.ARMS)))
ok("arm B is NVIDIA's embodied prompt with nothing added",
   eyeread.ARM_B == "What can be the next immediate action?",
   repr(eyeread.ARM_B))
ok("arm D is the hazard engine and goes in the SYSTEM slot",
   eyeread.system_for("D").startswith("SYSTEM ROLE"))
ok("every other arm keeps NVIDIA's system prompt",
   all(eyeread.system_for(a) == eyeread.SYSTEM for a in "ABCE"))
ok("a system-slot arm still asks a question in the user turn",
   eyeread.AV_COT in eyeread.build_prompt("", arm="D"))
ok("arm E is much shorter than arm D",
   len(eyeread.ARM_E) * 10 < len(eyeread.ARM_D),
   f"{len(eyeread.ARM_E)} vs {len(eyeread.ARM_D)}")
ok("arm E keeps the boundary rule",
   "not the collision-warning system" in eyeread.ARM_E)
ok("arm E keeps the silence default", "silent by default" in eyeread.ARM_E.lower())
ok("arm E keeps the language rules",
   all(w in eyeread.ARM_E for w in ("appears", "possible", "likely",
                                    "developing", "partially obscured")))
ok("arm E carries no worked example", not eyeread.ARM_EXAMPLES["E"])
ok("arm D's worked examples are registered so they can be watched for",
   len(eyeread.ARM_EXAMPLES["D"]) == 2)

# THE RUBRIC IS READ OFF THE PROMPT, NOT TYPED OUT. A hand-copied rubric is
# correct the day it is written and silently wrong the first time the prompt
# changes -- and a silently wrong rubric scores every arm against rules
# nobody is applying.
ok("the false-positive list is parsed from the prompt file",
   len(rubric.FALSE_POSITIVE_RULES) == 10,
   str(len(rubric.FALSE_POSITIVE_RULES)))
ok("...and it is the engine's own wording",
   "cars safely traveling in adjacent lanes" in rubric.FALSE_POSITIVE_RULES)
ok("the priority levels are parsed from the prompt file",
   rubric.PRIORITY_LEVELS == ["NORMAL", "WATCH", "CAUTION", "HIGH",
                              "CRITICAL-CANDIDATE"],
   str(rubric.PRIORITY_LEVELS))
ok("WATCH is silent and CAUTION is not, per the prompt's own wording",
   not rubric.is_elevated("WATCH") and rubric.is_elevated("CAUTION"))
ok("the two scales map both ways in one place",
   rubric.to_engine("urgent") == "HIGH" and rubric.to_target("CAUTION") == "advise")

_ST = {"tracks": [
    {"id": 1, "label": "car", "side": "right of the ego lane",
     "range_rate_ms": 2.0, "last_range_m": 30, "in_lane": False, "is_lead": False},
    {"id": 2, "label": "car", "side": "in the ego lane", "range_rate_ms": -3.0,
     "last_range_m": 20, "in_lane": True, "is_lead": True}]}
ok("elevating a car measured beside us and not closing is a violation",
   rubric.grade("", {"priority": "CAUTION", "about": [1]}, _ST)["n_violations"] >= 1)
ok("elevating the closing in-lane lead is NOT a violation",
   rubric.grade("", {"priority": "CAUTION", "about": [2]}, _ST)["n_violations"] == 0)
ok("a WATCH elevates nothing, so it cannot violate",
   rubric.grade("", {"priority": "WATCH", "about": [1]}, _ST)["n_violations"] == 0)
ok("the four unmeasurable rules are named every time, not quietly dropped",
   len(rubric.grade("", {"priority": "HIGH", "about": [2]}, _ST)["unchecked"]) == 4)

# BORROWING IS DERIVED FROM THE PROMPT, for the same reason as the rubric.
_demo = "Watch for a deer, a stroller, standing water and steel plates."
ok("the hazard vocabulary is derived from the prompt text",
   {"deer", "stroller"} <= eyeread.prompt_nouns(_demo))
ok("a borrowed hazard with no track behind it is caught",
   {b["term"] for b in eyeread.borrowed_terms(
       "A deer may emerge.", _demo, _ST)} >= {"deer"})
ok("naming a class the detector DOES hold is not borrowing",
   not eyeread.borrowed_terms("A car is ahead.", "cars and deer", _ST))
ok("arm D's hazard vocabulary is large enough to be worth watching",
   len(eyeread.prompt_nouns(eyeread.ARM_D)) > 100,
   str(len(eyeread.prompt_nouns(eyeread.ARM_D))))

# THE SCHEMA IS A GRAMMAR, NOT A PARAGRAPH.
sch = rubric.JUDGEMENT_SCHEMA
ok("the schema pins priority to the engine's own enum",
   sch["properties"]["priority"]["enum"] == rubric.PRIORITY_LEVELS)
ok("the schema admits no extra fields", sch["additionalProperties"] is False)
ok("the schema's only free text is one short line",
   sch["properties"]["why"]["maxLength"] <= 200)
ok("track ids are integers, so a name cannot arrive as a citation",
   sch["properties"]["about"]["items"]["type"] == "integer")

# THE GROUND TRUTH MUST NOT REACH THE MODEL.
_gstate = dict(STATE, loop={"spoke": True, "reasons": ["closing"],
                            "bands": ["UNSAFE"]})
_rendered = grounding.render(_gstate)
ok("the loop's speak decision is NEVER rendered into the prompt",
   "spoke" not in _rendered and "closing\"" not in _rendered
   and "loop" not in _rendered.lower().split(),
   _rendered[:120])
ok("...though the band, which is measured state, still is",
   "GETTING_UNSAFE" in _rendered)
for _arm in ("A", "B", "C", "D", "E"):
    _p = eyeread.build_prompt(_rendered, arm=_arm) + eyeread.system_for(_arm)
    ok(f"arm {_arm}'s prompt never contains the loop's decision",
       "should_speak" not in _p.replace(eyeread.JUDGEMENT, "")
       and "loop fired" not in _p)


# --- verdict -----------------------------------------------------------------
print(f"\n{len(_checks) - len(_fails)}/{len(_checks)} checks passed")
if _fails:
    print("FAILED:")
    for f in _fails:
        print(f"  - {f}")
    sys.exit(1)
print("ok")
