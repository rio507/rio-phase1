"""reading_discriminates.py — does the reading change when the picture changes?

    python tools/reading_discriminates.py [--question "..."] [--cap 512]

THE GATE THAT COMES BEFORE EVERY OTHER GATE
-------------------------------------------
On 2026-09-21 a prompt change was measured for format compliance, truncation,
decoding loops, prompt echoes, token count and latency. It passed all six by
wide margins and it was a disaster: the model returned a near-copy of the
prompt's own worked example on every frame. Thirteen of seventeen readings on
the drive that followed asserted a hazard that was not there -- "stalled
minibus blocking lane", "bicyclist entering lane" -- and those went to the
Perception card and to RIO as camera evidence.

Not one of those six measurements could have caught it. Every one asks about
the SHAPE of the answer. The question none of them asked is the only one that
matters about a sensor:

    DOES THE READING CHANGE WHEN THE PICTURE CHANGES?

A reading that is byte-identical across a daylight frame and a night frame is
not a reading, whatever else it scores. This runs first, and a change that
fails it is refused regardless of its other numbers. Format compliance is NOT a
criterion here and must not become one.

THE CORPUS IS CLEAN, AND THAT TOOK FINDING OUT
-----------------------------------------------
Every clip under runs/ is a SCREEN RECORDING OF RIO'S OWN DASHBOARD: the
detector boxes and the "car 24.2m" range labels are burned into the pixels,
and runs/night_clip.webm is the same daylight footage darkened. Asked what it
saw, the model said so itself -- "The black sedan is labeled as 'car 24.2m'".
Every vision measurement taken on those frames was measuring the model reading
RIO's own output back to us, including the prompt A/B that produced the
disaster above.

So the corpus here is built from /workspace/ufldv2/example.mp4, which is real
dashcam footage with nothing drawn on it. If that file ever goes, this file
must fail rather than fall back to runs/ -- a contaminated corpus is worse than
no corpus, because it passes.

Night is the only class that has to be synthesised (there is no real night
footage on this pod) and it is labelled `dark` rather than `night` for that
reason: it is the daylight clip with the brightness pulled down, which is a
weaker test than real night and is the strongest available.
"""
import argparse
import collections
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CLEAN_SRC = Path("/workspace/ufldv2/example.mp4")

# Ground truth the answers are checked against. Not a vocabulary the model is
# given -- it is never told these words.
DAY_WORDS = {"day", "daytime", "daylight", "bright", "sunny", "sunlit", "sun",
             "clear", "blue", "overcast", "afternoon", "morning", "midday"}
NIGHT_WORDS = {"night", "nighttime", "dark", "darkness", "dusk", "evening",
               "twilight", "unlit", "dim", "dimly", "headlights", "moonlit"}
ROAD_WORDS = {"lane", "lanes", "highway", "asphalt", "sedan", "motorway",
              "guardrail", "roadway", "traffic"}


def build_corpus(out_dir, n_day=8, n_dark=3):
    if not CLEAN_SRC.exists():
        raise SystemExit(
            f"no clean footage at {CLEAN_SRC}.\n"
            "Every clip under runs/ has RIO's own overlay burned into it and\n"
            "must not be used here — see the note at the top of this file.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(CLEAN_SRC),
                    "-vf", f"fps={n_day}/50", "-frames:v", str(n_day),
                    str(out / "day%02d.jpg")], check=True)
    for i in range(1, n_dark + 1):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y",
                        "-ss", str(10 + i * 12), "-i", str(CLEAN_SRC),
                        "-vf", "eq=brightness=-0.55:saturation=0.4",
                        "-frames:v", "1", str(out / f"dark{i:02d}.jpg")],
                       check=True)
    from PIL import Image
    import numpy as np
    Image.new("RGB", (1280, 720), (6, 6, 8)).save(out / "blank01.jpg")
    Image.fromarray(np.random.RandomState(1).randint(
        0, 255, (720, 1280, 3), dtype="uint8")).save(out / "noise01.jpg")
    groups = collections.defaultdict(list)
    for p in sorted(out.glob("*.jpg")):
        groups["".join(c for c in p.stem if c.isalpha())].append(str(p))
    return dict(groups)


def read_all(groups, question, cap):
    """The real model, asked the way NVIDIA's card says to ask it."""
    from PIL import Image
    import torch
    import config
    import vision
    vision._ensure_loaded()
    proc, model = vision._processor, vision._model
    out = {}
    for cls, paths in groups.items():
        rows = []
        for p in paths:
            pil = vision._downscale(Image.open(p).convert("RGB"), None)
            msgs = [{"role": "system",
                     "content": [{"type": "text", "text": vision.COSMOS_SYSTEM}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": pil},
                        {"type": "text",
                         "text": question + "\n\n" + vision.COSMOS_FORMAT}]}]
            inp = proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.inference_mode():
                gen = model.generate(
                    **inp, max_new_tokens=cap, do_sample=True,
                    temperature=config.TEACHER_TEMPERATURE,
                    top_p=config.TEACHER_TOP_P)
            ms = (time.time() - t0) * 1000.0
            raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0]
            answer = raw.split("</think>", 1)[1].strip() if "</think>" in raw else ""
            # WHAT WOULD ACTUALLY SHIP, not what the model emitted. The guards
            # are part of the reading path, so a gate that judged the raw
            # output would be grading something no driver ever sees -- and
            # would count three different spellings of "yes" as three distinct
            # readings, which is how a boilerplate arm passes a distinctness
            # test.
            refused = ""
            if answer:
                if vision._says_nothing(answer):
                    refused = "says nothing"
                elif vision.reading_refused(answer, 0):
                    refused = (vision.reading_refused(answer, 0) or {}).get("why", "guard")
                else:
                    import rio_prompts as _rp
                    if _rp.echoes_prompt(answer, vision.prompt_in_use()):
                        refused = "echoes the prompt"
            rows.append({"file": os.path.basename(p),
                         "answer": "" if refused else answer,
                         "raw_answer": answer, "refused": refused,
                         "ms": ms, "closed": "</think>" in raw})
        out[cls] = rows
    return out


def light_of(text):
    w = set(text.lower().replace(",", " ").replace(".", " ").split())
    night, day = w & NIGHT_WORDS, w & DAY_WORDS
    return "night" if night and not day else "day" if day and not night else "?"


def judge(readings):
    fails = []
    day = [r["answer"] for r in readings.get("day", []) if r["answer"]]
    dark = [r["answer"] for r in readings.get("dark", []) if r["answer"]]

    # 1. MOVES. One string for several different frames of a moving road is
    #    boilerplate, however plausible the string reads.
    for cls, vals in (("day", day), ("dark", dark)):
        if len(vals) > 1 and len(set(vals)) < max(2, len(vals) - 1):
            fails.append(f"{cls}: {len(set(vals))} distinct answers for "
                         f"{len(vals)} different frames — boilerplate")

    # 2. SEPARATES. An answer that appears for a daylight frame and a dark one
    #    is text the picture did not produce.
    shared = (set(day) & set(dark)) - {""}
    if shared:
        fails.append(f"{len(shared)} answer(s) identical between a daylight "
                     f"frame and a dark one ({sorted(shared)[0][:50]!r})")

    # 3. KNOWS. Ground truth: midday and midnight are not a subtle distinction.
    wrong = [(c, a) for c, vals in (("day", day), ("dark", dark))
             for a in vals
             if light_of(a) != "?" and light_of(a) != ("night" if c == "dark" else "day")]
    if wrong:
        fails.append(f"the light is called wrong in {len(wrong)} answer(s) "
                     f"({wrong[0][1][:50]!r})")

    # 4. A frame with nothing in it is not a road.
    for cls in ("blank", "noise"):
        for r in readings.get(cls, []):
            w = set(r["answer"].lower().replace(",", " ").split())
            if w & ROAD_WORDS:
                fails.append(f"{cls}: nothing in the frame, described as a road "
                             f"({r['answer'][:50]!r})")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=None)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--frames", default=None)
    a = ap.parse_args()
    import config
    import vision
    question = a.question or vision.COSMOS_QUESTION
    cap = a.cap or int(config.LOCAL_VISION_MAX_TOKENS)

    groups = build_corpus(a.frames or (os.getenv("TMPDIR", "/tmp") + "/rio_clean"))
    print(f"\n=== does the reading change when the picture changes? ===")
    print(f"    question : {question!r}")
    print(f"    cap      : {cap} tokens")
    print(f"    corpus   : {CLEAN_SRC} (no overlay), "
          + ", ".join(f"{k} {len(v)}" for k, v in sorted(groups.items())) + "\n")

    readings = read_all(groups, question, cap)
    for cls in ("day", "dark", "blank", "noise"):
        for r in readings.get(cls, []):
            if r["answer"]:
                txt = r["answer"]
            elif r["refused"]:
                txt = f"(refused: {r['refused']}) {r['raw_answer'][:40]!r}"
            else:
                txt = "(no answer — the trace filled the budget)"
            print(f"  {cls:<6} {txt[:112]}")
    print()
    answered = sum(1 for rs in readings.values() for r in rs if r["answer"])
    refused = sum(1 for rs in readings.values() for r in rs if r["refused"])
    total = sum(len(rs) for rs in readings.values())
    print(f"  refused by the guards: {refused}/{total}")
    ms = [r["ms"] for rs in readings.values() for r in rs]
    print(f"  answered {answered}/{total}   latency p50 {st.median(ms):.0f} ms  "
          f"max {max(ms):.0f} ms")
    day = [r["answer"] for r in readings.get("day", []) if r["answer"]]
    dark = [r["answer"] for r in readings.get("dark", []) if r["answer"]]
    print(f"  distinct: day {len(set(day))}/{len(day)}  dark {len(set(dark))}/{len(dark)}")

    fails = judge(readings)
    print()
    if fails:
        print("  FAILED — this reading does not move with the frame:")
        for f in fails:
            print(f"    - {f}")
        return 1
    print("  PASSED — the reading moves with the frame.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
