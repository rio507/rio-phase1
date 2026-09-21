"""local_vision_selftest.py — the resident eye is a sensor, and cannot speak.

    python -m tools.local_vision_selftest

WHAT THIS DEFENDS, AND WHY IT IS NOT tools/visual_selftest.py's JOB.

visual_selftest runs the real pipeline against real frames on a GPU and asks
whether the ANSWERS are good. This asks the structural questions that a swap
gets wrong silently and that no amount of good answers would reveal:

  1  THE ROLE RESOLVES BOTH WAYS. config.LOCAL_VISION_MODEL=cosmos|qwen selects
     the weights, the prompt AND whether the output may be spoken. A rollback
     that moved the weights and left the prompt behind would ask an instrument
     to have a voice, or waste the one thing the 8B is kept for.

  2  A SENSOR CANNOT SPEAK AS HER. Under cosmos, observer._record must return
     speakable=False for EVERY reading, whatever the words are -- including a
     reading that would sail through persona.lint(). The gate is the model's
     role, asked first, not the shape of the sentence. look()'s
     `observer_direct` branch is reached only on `speakable`, so this one
     assertion is what makes "grok says it, not the camera" true.

  3  THE PROMPTS ARE FOR THE MODELS THEY ARE GIVEN TO. The sensor prompt must
     not carry her register, her rhythm or the four paired examples; the
     observer prompt must keep all three. The examples are a Qwen-specific cure
     (rio_prompts.OBSERVER_EXAMPLES) and carrying them into a different model's
     prompt untested is how a fix becomes a superstition -- see
     tools/vision_ab.py --prompt-ab, which measures it.

  4  THE GUARDS ARE REAL. A reasoning trace is stripped and an UNTERMINATED one
     is refused rather than published as perception. Recited label text and
     decoding loops are refused (teachers.canned, written because Alpamayo 1.5
     answered eight of ten keyframes with a training label, and a 2B is more
     prone to that, not less). An advisory reading is refused, because a warning
     on this car comes from measured geometry and never from a caption.

  5  NOTHING NAMES A MODEL IN A LITERAL. /health, the page and the drive log ask
     the role. This project has already shipped a card that said "ElevenLabs"
     for every backend that was not OpenAI's, and a perception card that says
     "Seeing" was two model generations out of date.

No GPU and no network: every check here is a decision, a string or a source
file. The model's actual readings are tools/vision_ab.py's job.
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

_checks = []
_fails = []


def ok(name, cond, extra=""):
    _checks.append(name)
    if cond:
        print(f"  ok   {name}" + (f" — {extra}" if extra else ""))
    else:
        _fails.append(name)
        print(f"  FAIL {name}" + (f" — {extra}" if extra else ""))


def section(t):
    print(f"\n== {t}")


def fresh(role: str):
    """config + rio_prompts + vision as they are under this role."""
    os.environ["LOCAL_VISION_MODEL"] = role
    for m in ("config", "rio_prompts", "vision", "observer", "persona"):
        sys.modules.pop(m, None)
    import config
    import rio_prompts
    return config, rio_prompts


class FakeFrame:
    frame_id = "f000001"
    wall_t = 1789900000.0
    age_s = 0.2
    origin = "s1:camera"
    jpeg = b"\xff\xd8ffff"


def main() -> int:
    # -----------------------------------------------------------------------
    section("1. the role selects the weights, the prompt and the voice")
    cfg, rp = fresh("cosmos")
    ok("cosmos resolves to the 2B",
       cfg.local_vision_model_id() == "nvidia/Cosmos-Reason2-2B",
       cfg.local_vision_model_id())
    ok("...labelled without the org prefix, for the glass",
       cfg.local_vision_label() == "Cosmos-Reason2-2B", cfg.local_vision_label())
    ok("...and it may NOT speak as her",
       cfg.local_vision_speaks_directly() is False)

    cfg_q, rp_q = fresh("qwen")
    ok("qwen is one env var away, and is the 8B",
       cfg_q.local_vision_model_id() == "Qwen/Qwen3-VL-8B-Instruct",
       cfg_q.local_vision_model_id())
    ok("...and IS allowed to speak, which is what it is kept for",
       cfg_q.local_vision_speaks_directly() is True)

    # The rollback has to be one variable, not two. A deployment that moved the
    # weights and forgot the prompt is the failure this check exists for.
    import vision as v_q
    ok("the prompt follows the weights: qwen gets the observer prompt",
       v_q.TEACHER_PROMPT == rp_q.OBSERVER_PROMPT)
    fresh("cosmos")
    import vision as v_c
    _rp = __import__("rio_prompts")
    # EITHER sensor variant is correct here; which one is a latency decision
    # (config.LOCAL_VISION_SENSOR_PROMPT, asserted in its own section below).
    # What this check is about is that the ROLE picks a sensor prompt at all.
    ok("...and cosmos gets a sensor prompt, not hers",
       v_c.TEACHER_PROMPT in (_rp.SENSOR_PROMPT, _rp.SENSOR_PROMPT_TERSE),
       "terse" if v_c.TEACHER_PROMPT == _rp.SENSOR_PROMPT_TERSE else "full")
    ok("...so one env var moves both", v_q.TEACHER_PROMPT != v_c.TEACHER_PROMPT)

    # -----------------------------------------------------------------------
    section("2. a sensor cannot speak as her — whatever the words are")
    fresh("cosmos")
    import observer as obs_c
    # A line written to PASS the persona lint. Under a sensor model it must
    # still be refused, because the gate is the model's role and not the
    # sentence: this is the assertion that keeps grok in front of her mouth.
    flattering = "Open freeway, light traffic — dry hills both sides"
    import persona
    ok("the test line really would pass the persona lint",
       not persona.lint(flattering), repr(persona.lint(flattering)))
    rec = obs_c._record(flattering, FakeFrame(), "s1")
    ok("...and the sensor model's record is STILL not speakable",
       rec["speakable"] is False, repr(rec["faults"]))
    ok("...refused for being a sensor, not for its prose",
       rec["faults"] == ["sensor_model"], repr(rec["faults"]))
    ok("...and the reading carries the instrument's name",
       rec.get("model") == "Cosmos-Reason2-2B", repr(rec.get("model")))
    ok("...with the frame's own clock, so age can be judged",
       rec["frame_id"] == "f000001" and rec["frame_wall_t"] == FakeFrame.wall_t)

    fresh("qwen")
    import observer as obs_q
    rec_q = obs_q._record(flattering, FakeFrame(), "s1")
    ok("under qwen the same line IS speakable — the rollback is intact",
       rec_q["speakable"] is True, repr(rec_q["faults"]))

    # look()'s direct branch is gated on exactly this field.
    rt = (REPO / "realtime.py").read_text()
    ok("look() reaches the direct path only on `speakable`",
       'if hit.get("speakable"):' in rt
       and 'base["speak_directly"] = True' in rt)
    ok("...and hands the composed path the instrument's name as evidence",
       'base["reading_from"]' in rt and "an instrument's reading" in rt)

    # -----------------------------------------------------------------------
    section("3. the prompts are written for the models they are given to")
    fresh("cosmos")
    import rio_prompts as rp2
    sp = rp2.SENSOR_PROMPT
    ok("the sensor prompt asks for a reading, not a sentence she would say",
       "ROAD:" in sp and "TRAFFIC:" in sp and "RISK:" in sp)
    ok("...it does not put her in it",
       "RIO" not in sp and "her own words" not in sp)
    ok("...it allows 'unreadable' as a real answer",
       "unreadable" in sp.lower())
    ok("...it forbids inventing a speed or a distance",
       "Never estimate a speed" in sp)
    ok("...and it asks for no reasoning trace, which the code also enforces",
       "No reasoning trace" in sp)
    for ex in rp2.OBSERVER_EXAMPLES:
        ok(f"...and carries none of Qwen's paired examples ({ex[:24]}…)",
           ex not in sp)
    ok("the A/B variant exists so the question is measured, not assumed",
       rp2.OBSERVER_EXAMPLES[0] not in sp
       and len(rp2.SENSOR_PROMPT_WITH_EXAMPLES) > len(sp))
    ab = (REPO / "tools" / "vision_ab.py").read_text()
    ok("...by a harness that runs both over the same frames",
       "SENSOR_PROMPT_WITH_EXAMPLES" in ab and "--prompt-ab" in ab)

    ok("the observer prompt KEEPS its examples — they are Qwen's cure",
       all(e in rp2.OBSERVER_PROMPT for e in rp2.OBSERVER_EXAMPLES))

    # -----------------------------------------------------------------------
    section("4. an instrument reports, and does not advise")
    ok("a clean reading has no faults",
       rp2.sensor_faults("ROAD: two lanes, wet | TRAFFIC: none | RISK: none seen")
       == [])
    for bad in ("RISK: slow down, stopped traffic ahead",
                "you should brake now",
                "be careful of the cyclist"):
        ok(f"...and an advisory one does ({bad[:28]}…)",
           rp2.sensor_faults(bad) != [], repr(rp2.sensor_faults(bad)))

    # -----------------------------------------------------------------------
    section("5. the guards")
    import vision as vis
    vis.reset_flags()
    answer, had, unterm = vis._strip_think(
        "<think>the road looks wet, but that could be glare</think>"
        "ROAD: two lanes, wet | TRAFFIC: none | RISK: none seen")
    ok("a closed reasoning trace is stripped and the reading kept",
       had and not unterm and answer.startswith("ROAD:"), repr(answer[:40]))
    answer2, had2, unterm2 = vis._strip_think(
        "<think>let me consider the lane markings and the")
    ok("an UNTERMINATED trace leaves nothing to publish",
       had2 and unterm2 and answer2 == "", repr(answer2))
    ok("...which the caller turns into a refusal, not a caption",
       'return ""' in (REPO / "vision.py").read_text().split(
           "_flags[\"think_unterminated\"] += 1")[1][:400])

    # -- the frame gate, which is the one that had to be ADDED -------------
    # Cosmos-Reason2-2B answered a featureless grey frame AND a frame of pixel
    # noise with the same confident sentence about a single-lane asphalt road.
    # Measured, both fabrications, neither caught by any guard that looks at the
    # WORDS: they are not the prompt's examples, they carry no memorised
    # whitespace, and teachers.canned needs three identical readings in a row.
    # So the question is asked of the picture instead, before any model sees it.
    from PIL import Image as _Image
    blank = _Image.new("RGB", (640, 360), (128, 128, 128))
    ok("a featureless frame measures as having nothing in it",
       vis.frame_structure(blank) < 8.0, f"{vis.frame_structure(blank):.2f}")
    import random as _random
    _random.seed(7)
    noisy = _Image.new("RGB", (640, 360))
    noisy.putdata([(_random.randrange(256),) * 3 for _ in range(640 * 360)])
    # Pixel noise survives neither the JPEG nor the downscale: it averages to
    # near-uniform grey, which is why the same number catches it.
    ok("...and so does pixel noise once it is downscaled",
       vis.frame_structure(noisy) < 30.0, f"{vis.frame_structure(noisy):.2f}")
    grad = _Image.new("RGB", (640, 360))
    grad.putdata([(x % 256, (x // 3) % 256, 200) for x in range(640 * 360)])
    ok("...while a frame with structure in it does not",
       vis.frame_structure(grad) > 8.0, f"{vis.frame_structure(grad):.2f}")
    # -- the generation policy, which is what made the cadence -------------
    fresh("cosmos")
    import vision as vis2
    kw = vis2._generate_kwargs()
    ok("the reading is capped at the measured budget, not at a trace's",
       kw["max_new_tokens"] == 48, str(kw["max_new_tokens"]))
    ok("...with the repetition penalty that removed the decoding loop",
       kw.get("repetition_penalty") == 1.05, str(kw.get("repetition_penalty")))
    ok("...and NOT the compiled static cache, which broke the live observer",
       "cache_implementation" not in kw, str(kw))
    ok("warm() makes the same call the observer will, so no compile or "
       "allocation lands on a drive's first frame",
       "_generate_kwargs()" in (REPO / "vision.py").read_text()
       .split("def warm(")[1][:2000])
    import rio_prompts as rp3
    ok("the sensor prompt is the terse one by default",
       vis2.TEACHER_PROMPT == rp3.SENSOR_PROMPT_TERSE)
    ok("...which names all three fields and carries no example reading",
       all(f in rp3.SENSOR_PROMPT_TERSE for f in ("ROAD:", "TRAFFIC:", "RISK:")))
    ok("...and no angle-bracket placeholder, which the model copied verbatim",
       "<" not in rp3.SENSOR_PROMPT_TERSE.replace("<br>", ""))

    _vsrc = (REPO / "vision.py").read_text()
    ok("the gate runs BEFORE the forward pass, not after the answer",
       _vsrc.index("structure < config.LOCAL_VISION_BLANK_STD")
       < _vsrc.index("_model.generate"))
    ok("...and is counted as its own flag",
       '"blank_frame": 0' in _vsrc
       and "blank_frame" in vis.flag_rate())

    from teachers import canned
    recited = "A vehicle controls loss. The vehicle driver is in distracted driving."
    ok("the canned guard still catches memorised label text",
       bool(canned.describe(recited, 0).get("markers")),
       str(canned.describe(recited, 0).get("markers")))
    loop = " ".join(["The sedan is also further away from the ego vehicle."] * 5)
    ok("...and a decoding loop",
       canned.describe(loop, 0).get("loop", 0) >= canned.LOOP_FLOOR)
    # THE TWO LOOPS THAT GOT THROUGH, both measured on real readings.
    #
    # A repeated WORD: loop_run splits on sentence boundaries, so 203 consecutive
    # "interstate" with no full stop was one sentence and scored clean. It was
    # also the worst reading of the run at 5721 ms.
    interstate = "ROAD: three lanes, asphalt, " + "interstate " * 203
    ok("...a word repeated to the token ceiling (the 5721 ms reading)",
       canned.describe(interstate, 0).get("word_loop", 0)
       >= canned.WORD_LOOP_FLOOR,
       f"run={canned.word_loop_run(interstate)}")
    # ...and a repeated piece with NO WHITESPACE, which str.split() sees as one
    # word. The terse sensor format's own separator is a bar, so the bar became
    # the loop's unit.
    bars = "ROAD: two|three|curb|curb|curb|curb|curb|curb|curb|"
    ok("...and a loop on the format's own separator, with no spaces in it",
       canned.describe(bars, 0).get("word_loop", 0) >= canned.WORD_LOOP_FLOOR,
       f"run={canned.word_loop_run(bars)}")
    for good in ("ROAD: five, two-way, asphalt | TRAFFIC: none | RISK: none",
                 "the the road ahead", "very very very good"):
        ok(f"...while real text is left alone ({good[:34]}…)",
           not canned.describe(good, 0))
    ok("...and repetition is a HINT, reported apart from proof",
       canned.describe("ROAD: clear", 4).get("strength") == "weak")

    # -----------------------------------------------------------------------
    # AND NOW THE REFUSAL ITSELF, which is a different claim from every check
    # above and is the one that was not being made.
    #
    # Everything before this line asks whether canned.describe FLAGS a bad
    # reading. It does, and it did on 2026-09-21, and the reading went to the
    # Perception card anyway: the branch in vision.py that acts on a verdict
    # listed `markers or loop` and word_loop had been added to the guard
    # without it. The suite passed because it tested the detector and then
    # grepped vision.py for a substring -- neither of which can fail the way
    # the system failed.
    #
    # So these run vision.reading_refused: the function the live path calls.
    # The two strings are the ones this pod's own log printed under "(not
    # refused)" while a clip was playing.
    LIVE_LOOPS = [
        ("the drive of 2026-09-21, bar-separated with case mixing",
         "ROAD: single|single|curb|CURB|CURB|CURB|CURB|CURB|CURB|CURB|CURB|"
         "CURB|CURB|CURB|"),
        ("...and its mixed-separator variant from the same drive",
         "ROAD: two|single|curb|CURB: curb|CURB CURB CURB CURB CURB CURB "
         "CURB CURB CURB CU"),
        ("the 5721 ms word loop", "ROAD: three lanes, asphalt, "
         + "interstate " * 203),
        ("a memorised label row", recited),
        ("a repeated sentence", loop),
    ]
    for why, text in LIVE_LOOPS:
        v = vis.reading_refused(text, 0)
        ok(f"the LIVE PATH refuses {why}", bool(v),
           (v or {}).get("why") or "NOT REFUSED — this reaches the card")
    for good in ("ROAD: five, two-way, asphalt | TRAFFIC: none | RISK: none",
                 "A white sedan two cars ahead, and the lane is clear.",
                 "the the road ahead", "very very very good"):
        ok(f"...and publishes a real reading ({good[:34]}…)",
           vis.reading_refused(good, 0) is None)
    ok("...while repetition ALONE is still only a hint, never a refusal",
       vis.reading_refused("ROAD: clear | TRAFFIC: none | RISK: none", 9) is None)
    # The refusal set is asked OF the guard rather than kept as a second list
    # here, which is what stops the next kind of loop repeating this fault.
    ok("...and the refusal set is canned.py's own strength, not a copied list",
       'verdict.get("strength") == "strong"' in (REPO / "vision.py").read_text())

    # -----------------------------------------------------------------------
    # THE TWO MODES THE DRIVE OF 2026-09-21 FOUND, neither of which is a loop
    # and neither of which was visible to anything above this line. 150
    # readings, and a large share of them were one or the other.
    DRIVE_CATALOGUES = [
        ("a taxonomy of vehicle types",
         "ROAD: two|four|asphalt|single|straight|no|no|no|no\nTRAFFIC: sedan|"
         "truck|car|bus|van|minibus|taxi|ambulance|fire truck|pol"),
        ("...and one of marques",
         "ROAD: two|four|dry asphalt | TRAFFIC: cars|minivans|sedans|trucks|"
         "jeeps|mazdas|xpo|bmws|hyundais|audi|volvos|chevys"),
        ("...and one with the prompt's own phrase in it",
         "ROAD: lanes, two, asphalt | LIGHT: red | TRAFFIC: cars, minivans, "
         "sedans, hatchbacks, SUVs, trucks, vans, motorcycles, bicycles, "
         "bicycles that matter, vehicles that matter, cars"),
        ("a list item repeated three times, which both loop checks allow",
         "ROAD: two|two|none|TRAFFIC: cars|cars|cars|RISK: none"),
    ]
    for why, text in DRIVE_CATALOGUES:
        v = vis.reading_refused(text, 0)
        ok(f"the LIVE PATH refuses {why}", bool(v),
           (v or {}).get("why") or "NOT REFUSED — this reaches the card")
    # ...and the place-word test is on WORD boundaries. As a substring, "back"
    # is inside "hatchbacks", and the twelve-item catalogue above read as a
    # scene that said where things were.
    ok("...a catalogue is not excused by a place word inside another word",
       canned.list_run("cars, minivans, hatchbacks, SUVs, trucks, vans, "
                       "motorcycles, bicycles") >= canned.LIST_FLOOR)
    for good in ("ROAD: five, two, single lane, asphalt, signalized | "
                 "TRAFFIC: cars, truck | RISK: none detected",
                 "ROAD: five lanes, asphalt | TRAFFIC: sedan on right lane, "
                 "sedan on left lane, sedan ahead in center lane | RISK: none",
                 "ROAD: unreadable | TRAFFIC: unreadable | RISK: no view"):
        ok(f"...while a real reading is published ({good[:30]}…)",
           vis.reading_refused(good, 0) is None)

    # THE PROMPT, HANDED BACK AS AN ANSWER. Derived from whichever prompt is in
    # use, so it cannot fall behind a rewrite -- which is the whole reason the
    # last three versions of this guard went stale.
    ok("a phrase from the live INSTRUCTIONS, used as a value, is refused",
       bool(rp3.echoes_prompt("ROAD: how many lanes you can count | "
                              "TRAFFIC: none | RISK: none",
                              vis.TEACHER_PROMPT)))
    ok("...including the rule that forbids a vocabulary, quoted back",
       bool(rp3.echoes_prompt("ROAD: two | TRAFFIC: a list of vehicle types | "
                              "RISK: none", vis.TEACHER_PROMPT)))
    ok("...and the worked example, returned verbatim",
       rp3.is_prompt_example(rp3.SENSOR_EXAMPLE))
    ok("...and the worked example, half copied",
       bool(rp3.echoes_prompt("ROAD: two lanes, dry | TRAFFIC: van braking "
                              "ahead, cyclist on the left | RISK: none",
                              vis.TEACHER_PROMPT)))
    ok("...while a real reading echoes nothing",
       not rp3.echoes_prompt("ROAD: five lanes, dry, day | TRAFFIC: sedan "
                             "ahead, hatchback behind | RISK: none",
                             vis.TEACHER_PROMPT))
    # THE EXAMPLE IS A SENTENCE ABOUT A ROAD, so a real reading of a similar
    # road resembles it and has copied nothing. Three words is coincidence
    # against an example and proof against an instruction; hence two floors.
    ok("...and a genuinely wet road at dusk is not an echo of the example",
       not rp3.echoes_prompt("ROAD: six lanes, wet, dusk | TRAFFIC: lorry "
                             "ahead in the next lane | RISK: none seen",
                             vis.TEACHER_PROMPT),
       "shares 'lanes wet dusk' with the example and nothing more")
    ok("...nor is an n-gram of nothing but function words",
       not rp3.echoes_prompt("ROAD: two lanes | TRAFFIC: car in the next "
                             "lane | RISK: none seen", vis.TEACHER_PROMPT))
    ok("...and it is counted as its own flag", "prompt_echo" in vis.flag_rate())

    # AN INVENTED FIELD IS SURFACED, NOT SWALLOWED. "LIGHT: red" mid-reading
    # used to become part of whatever field came before it.
    _inv = rp3.split_sensor_reading(
        "ROAD: lanes, two, asphalt | LIGHT: red | TRAFFIC: cars | RISK: none")
    ok("a field the model invented is named rather than hidden",
       _inv["unknown_fields"] == ["LIGHT"], str(_inv["unknown_fields"]))
    ok("...and it is cut out of its neighbour's value",
       _inv["fields"][0]["text"] == "lanes, two, asphalt",
       repr(_inv["fields"][0]["text"]))
    ok("...while what it said is kept, not dropped",
       "LIGHT: red" in (_inv["extra"] or ""), repr(_inv["extra"]))
    ok("...and a clean reading invents nothing",
       rp3.split_sensor_reading("ROAD: four | TRAFFIC: none | RISK: none "
                                "seen")["unknown_fields"] == [])

    vsrc = (REPO / "vision.py").read_text()
    ok("the canned guard runs on EVERY reading, not just teacher rows",
       "reading_refused(text, repeats)" in vsrc)
    ok("...and its refusals are counted as a rate",
       "def flag_rate" in vsrc and '"canned": 0' in vsrc)
    fr = vis.flag_rate()
    for k in ("prompt_example", "canned", "repeated", "think_trace",
              "think_unterminated", "advisory"):
        ok(f"...{k} is reported", k in fr and k in (fr["rate"] or {}))

    # -----------------------------------------------------------------------
    section("6. nothing names a model in a literal")
    app = (REPO / "app.py").read_text()
    ok("/health asks the role for the eye",
       '"vision_role": config.LOCAL_VISION_MODEL' in app
       and '"vision": vision.MODEL_ID' in app)
    ok("...and reports whether it may speak, which a model id cannot say",
       '"vision_speaks_directly"' in app)
    page = (REPO / "static" / "index.html").read_text()
    ok("the perception card renders what /health says, not a constant",
       "m.vision_speaks_directly === false" in page
       and "short(m.vision)" in page)
    ok("...and says 'reading' for a sensor rather than 'seeing'",
       "· reading" in page)
    ok("vision.model_label() asks config rather than holding a name",
       "def model_label" in vsrc and "config.local_vision_label()" in vsrc)
    hard = re.findall(r'"(?:Qwen|nvidia)/[A-Za-z0-9.\-]+"', vsrc)
    ok("vision.py hard-codes no checkpoint path at all",
       not hard, f"found {hard}" if hard else "")

    print("\n" + "-" * 62)
    print(f"  {'PASS' if not _fails else 'FAIL'}: {len(_fails)} failure(s) "
          f"of {len(_checks)} checks")
    for f in _fails:
        print(f"    - {f}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
