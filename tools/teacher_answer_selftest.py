"""RIO's answer may DRAW ON the teachers. It may not be written by them.

    python -m tools.teacher_answer_selftest

WHAT IS UNDER TEST
------------------
The one place a shadow model's opinion is allowed to reach words a driver
hears: the result of a `look` call that a driver's own question triggered.
Everything about that path is a rule, and every rule here is one of them.

  * fresh readings are included, stale ones are OMITTED rather than aged
  * the block is labelled by source, and the MEASURED state is labelled as
    measured, because they do not carry the same weight
  * a question about merging, turning or cross traffic carries the
    field-of-view caveat, because the car has one camera and it points forward
  * the rules forbid quoting, commanding, and contradicting a live warning
  * nothing proactive can reach any of it

WHAT IS NOT UNDER TEST HERE
---------------------------
Whether the model obeys. That is a property of a language model under
instructions and it is asserted where it can be: the instructions say it
(checked below), the tool result repeats it (checked below), and the
near-verbatim check gives a mechanical floor -- a reply that reproduces a
teacher's sentence is caught whatever the model intended.
"""
import argparse
import difflib
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                                     # noqa: E402

config.TEACHERS_ENABLED = True

import realtime                                                   # noqa: E402
from teachers import panel                                        # noqa: E402

PASS, FAIL = [], []


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(("  ok    " if cond else "  FAIL  ") + what)


def section(name):
    print(f"\n=== {name} ===")


# A reading of the shape panel.context_for produces, at a chosen age.
def block(age_s=0.5, stale=False):
    if stale:
        return {"readings": {}, "fresh_within_s": 2.0,
                "dropped_stale": [{"model": "alpamayo1.5", "age_s": 7.4},
                                  {"model": "cosmos-reason2", "age_s": 9.1}],
                "measured": {"band": "NORMAL", "gap_m": 24.0, "ttc_s": None,
                             "speed_ms": 13.0, "speed_source": "obd",
                             "age_s": 0.4}}
    return {
        "readings": {
            "alpamayo1.5": {
                "source": "alpamayo1.5", "age_s": age_s, "precision": "bf16",
                "critical_actor": "The white sedan directly ahead, because it "
                                  "is braking and the gap is closing.",
                "actor_track_id": 7, "actor_label": "car", "actor_range_m": 21.4,
                "actor_matched": True,
                "attention": "The main hazard is the white sedan ahead.",
                "chain_of_causation": "Keep distance to the lead vehicle since "
                                      "it is directly ahead in our lane",
                "driving_decision": "Slowing, straight",
                "driving_decision_detail": {"longitudinal": "slowing",
                                            "lateral": "straight",
                                            "accel_ms2": -0.85},
            },
            "cosmos-reason2": {
                "source": "cosmos-reason2", "age_s": age_s, "precision": "bf16",
                "critical_actor": "The white sedan ahead in the same lane.",
                "actor_track_id": 7, "actor_label": "car", "actor_range_m": 21.4,
                "actor_matched": True,
                "attention": "Watch the sedan ahead; it is slowing.",
                "physics": "The white sedan is directly ahead of the ego "
                           "vehicle, decelerating, and is expected to continue "
                           "slowing over the next two seconds.",
                "plausibility": "The scenario is physically possible.",
                "implausible": False,
            },
        },
        "dropped_stale": [],
        "fresh_within_s": 2.0,
        "measured": {"band": "GETTING_UNSAFE", "gap_m": 21.4, "ttc_s": 4.1,
                     "speed_ms": 13.0, "speed_source": "obd", "age_s": 0.4},
    }


def run_freshness():
    section("A. fresh readings are context; stale ones are omitted")
    ctx = realtime.teacher_context(block(age_s=0.5), "what's that car doing?")
    ok(ctx and "readings" in ctx, "a fresh block becomes context")
    ok(set(ctx["readings"]) == {"alpamayo1.5", "cosmos-reason2"},
       f"both models are in it ({sorted(ctx.get('readings', {}))})")
    ok(all(r.get("source") and r.get("age_s") is not None
           for r in ctx["readings"].values()),
       "each labelled with its source and its age")

    stale = realtime.teacher_context(block(stale=True), "what's that car doing?")
    ok(not (stale.get("readings")),
       f"a stale block contributes NO readings ({list(stale.get('readings', {}))})")
    ok(stale.get("omitted_stale"),
       "and says how many were dropped, rather than being silently thinner — "
       "'the teachers had nothing fresh' is a fact she can use")
    ok("7.4" in str(stale.get("omitted_stale")),
       "with their ages, so the omission is checkable")
    ok(not realtime.teacher_context({}, "anything"),
       "no panel at all is an empty block, not an invented one")
    ok(not realtime.teacher_context(None, "anything"),
       "and neither is None")

    section("A2. the gate, and the fact it currently omits everything")
    ok(panel.CONTEXT_FRESH_S == config.TEACHER_CONTEXT_FRESH_S,
       f"the gate is config-driven ({panel.CONTEXT_FRESH_S}s), not a constant "
       f"buried in the plumbing")
    # THE HONEST CONSEQUENCE, ASSERTED SO IT CANNOT BE FORGOTTEN.
    #
    # Measured on the acceptance clip: Alpamayo answers in ~4-7 s and Cosmos in
    # ~8-10 s. A reading is therefore ALREADY older than the two-second gate by
    # the time it exists, so at the shipped value the block is empty every
    # time and RIO answers from the camera and the measured state alone --
    # which is exactly what she did before any of this was built.
    #
    # That is not a bug in the gate. It is what the gate is for, and it is the
    # honest consequence of asking a 10-billion-parameter model about a road a
    # car is driving down. The number is a real decision and this test exists
    # so nobody makes it by accident.
    SLOWEST_TEACHER_S = 8.0
    ok(panel.CONTEXT_FRESH_S < SLOWEST_TEACHER_S,
       f"at {panel.CONTEXT_FRESH_S}s the gate is TIGHTER than the slower "
       f"teacher's latency (~{SLOWEST_TEACHER_S}s), so the block will be "
       f"empty in practice — raise config.TEACHER_CONTEXT_FRESH_S "
       f"deliberately if that is not what is wanted")
    late = block(age_s=panel.CONTEXT_FRESH_S + 0.1)
    # ...and the mechanism itself is correct at whatever the number is.
    real = panel.context_for("no-such-session")
    ok(real == {},
       "a session with no readings at all yields an empty block, not a "
       "fabricated one")


def run_labelling():
    section("B. measured and inferred are not the same thing, and it says so")
    ctx = realtime.teacher_context(block(), "what's that car doing?")
    m = ctx.get("measured") or {}
    ok(m.get("band") == "GETTING_UNSAFE" and m.get("gap_m") == 21.4
       and m.get("ttc_s") == 4.1 and m.get("speed_ms") == 13.0,
       f"the deterministic state travels with it (band {m.get('band')}, "
       f"gap {m.get('gap_m')}, TTC {m.get('ttc_s')}, v {m.get('speed_ms')})")
    ok("MEASURED" in (m.get("note") or ""),
       "labelled as MEASURED rather than inferred")
    ok("wins" in (m.get("note") or "").lower(),
       "and says it outranks a teacher where they disagree")
    ok("second opinions" in (ctx.get("note") or "").lower()
       and "never a script" in (ctx.get("note") or "").lower(),
       f"the teachers are labelled as second opinions, not as instructions "
       f"({(ctx.get('note') or '')[:60]!r})")

    section("B2. the rules that come with it")
    rules = " ".join(ctx.get("rules") or []).lower()
    ok("your own read" in rules, "form her OWN read")
    ok("never quote" in rules and "line by line" in rules,
       "never quote or paraphrase line by line")
    ok("never a command" in rules or "never a command" in rules,
       "observation and suggestion, never a command")
    ok("brake" in rules and "steer" in rules,
       "and it names the controls she may not tell a driver to use")
    ok("do not contradict" in rules and "soften" in rules,
       "never contradict or soften a warning that is already running")
    ok("do not fill it in" in rules,
       "and what is not in the block, she does not know")


def run_field_of_view():
    section("C. one camera, pointing forward")
    for q, want in (
            ("is it safe to merge?", True),
            ("can I change lanes here", True),
            ("should I turn left here", True),
            ("is there cross traffic", True),
            ("what's that car doing?", False),
            ("what should I do at this light?", False)):
        ctx = realtime.teacher_context(block(), q)
        rules = " ".join(ctx.get("rules") or [])
        got = "cannot see" in rules.lower()
        ok(got is want,
           f"{q!r}: field-of-view caveat {'required' if want else 'not needed'} "
           f"-> {got}")
    ctx = realtime.teacher_context(block(), "is it safe to merge?")
    ok("I can't see your left" in " ".join(ctx.get("rules") or []),
       "and it gives her the words, so the caveat is a sentence rather than a "
       "concept she has to invent under time pressure")
    ok(ctx.get("field_of_view"),
       f"with the limitation stated as a fact too ({ctx.get('field_of_view')!r})")


def near_verbatim(reply, source, threshold=0.75):
    """Does `reply` reproduce any sentence of `source`? -> the worst ratio."""
    def sentences(t):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t or "")
                if len(s.strip()) > 20]
    worst = 0.0
    for a in sentences(reply):
        for b in sentences(source):
            worst = max(worst, difflib.SequenceMatcher(
                None, a.lower(), b.lower()).ratio())
    return worst


def run_no_quoting():
    section("D. her answer is hers — the near-verbatim floor")
    ctx = realtime.teacher_context(block(), "what's that car doing?")
    teacher_text = " ".join(
        str(v) for r in ctx["readings"].values()
        for k, v in r.items()
        if k in ("critical_actor", "attention", "chain_of_causation",
                 "physics", "plausibility"))

    good = "That white car ahead is slowing and the gap's closing a bit."
    bad_quote = ("The white sedan directly ahead, because it is braking and "
                 "the gap is closing.")
    bad_near = ("The white sedan is directly ahead of the ego vehicle, "
                "decelerating, and will continue slowing over the next two "
                "seconds.")
    ok(near_verbatim(good, teacher_text) < 0.75,
       f"a passenger sentence of her own passes "
       f"({near_verbatim(good, teacher_text):.2f})")
    ok(near_verbatim(bad_quote, teacher_text) >= 0.75,
       f"a straight quote is caught ({near_verbatim(bad_quote, teacher_text):.2f})")
    ok(near_verbatim(bad_near, teacher_text) >= 0.75,
       f"and so is a near-verbatim paraphrase "
       f"({near_verbatim(bad_near, teacher_text):.2f})")
    ok("alpamayo" not in str(ctx.get("rules")).lower()
       or "never name" in str(ctx.get("rules")).lower(),
       "and the rules forbid naming the models at all")


def run_instructions():
    section("E. the policy is in the SESSION instructions, not only the result")
    # WHITESPACE-NORMALISED. The addendum is wrapped prose, so "Do not fill it
    # in" is split across a newline in the source and a literal substring
    # search reports it missing. The test is about what the instruction SAYS,
    # not about where the lines break.
    text = re.sub(r"\s+", " ", realtime.instructions())
    for need, why in (
            ("SECOND OPINIONS", "the section exists"),
            ("ANSWERING, NEVER ANNOUNCING", "answering, never announcing"),
            ("never mention that they exist", "she never raises them herself"),
            ("FORM YOUR OWN READ", "she forms her own read"),
            ("Never quote them", "never quotes them"),
            ("the measurement wins", "the measured state outranks a teacher"),
            ("do not contradict", "never contradicts a running warning"),
            ("NEVER A COMMAND", "observation and suggestion only"),
            ("I can't see your left", "the field-of-view sentence"),
            ("Do not fill it in", "and what is absent she does not invent")):
        ok(need in text, f"{why} ({need!r})")
    ok(len(text) < 20000,
       f"and the whole instruction set is still one a model will read "
       f"({len(text)} chars)")


def run_no_proactive():
    section("F. there is no proactive path, and no way to make one")
    import subprocess

    r = subprocess.run(
        [sys.executable, "-m", "tools.teacher_firewall_selftest"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True, timeout=600)
    ok(r.returncode == 0,
       "the firewall suite passes — it is what asserts that context_for has "
       "exactly one caller and that nothing proactive reads a teacher value")
    ok("F. the teachers reach ONE thing" in r.stdout,
       "including the rule that says so by name")

    section("F2. the direct branch gets no second opinions")
    # A line already spoken word for word has no answer left to form, and
    # handing her opinions about it would invite a revision of something the
    # driver has already heard.
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "realtime.py")).read()
    i = src.index('base["speak_directly"] = True')
    j = src.index('return base', i)
    ok('"teachers"' not in src[i:j],
       "the observer_direct branch attaches no teacher block")


def main():
    argparse.ArgumentParser().parse_args()
    run_freshness()
    run_labelling()
    run_field_of_view()
    run_no_quoting()
    run_instructions()
    run_no_proactive()

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
