"""sensor_prompt_ab.py — does the prompt stop the model reciting? Measured.

    python tools/sensor_prompt_ab.py [--frames DIR] [--n 30]

WHY THIS EXISTS RATHER THAN A REWRITTEN PROMPT AND A HOPE
---------------------------------------------------------
The drive of 2026-09-21 produced 150 readings, and a large share of them were
not readings. Two modes, neither of which any guard could see at the time:

  THE PROMPT, HANDED BACK.  Road: lanes 5, surface asphalt, light green
                            Traffic: vehicles that matter cars, trucks, buses
                            ...|homes|vehicles_that_matter_and_where

  A VOCABULARY, EMPTIED.    TRAFFIC: sedan|truck|car|bus|van|minibus|taxi|
                                     ambulance|fire truck|pol
                            TRAFFIC: cars|minivans|sedans|trucks|jeeps|mazdas|
                                     xpo|bmws|hyundais|audi|volvos|chevys

Both come from the same place: the old format line was a fill-in-the-blank
template, and a template put in front of a model is a thing to complete. The
new SENSOR_PROMPT_TERSE describes the shape and names the fields instead.

That is a claim about behaviour and "the suite passes" is not a measurement of
it, so this runs BOTH prompts over the SAME frames through the real
vision._model, and counts what came back:

    echo        rio_prompts.echoes_prompt -- the prompt's own words as a value
    catalogue   canned.list_run           -- bare categories in a row
    item loop   canned.item_loop_run      -- one list item repeated
    truncated   the generation hit the cap without stopping
    refused     what vision.reading_refused would throw away
    length      chars, and tokens against the cap

The frames are a real road clip, one per second, which is the cadence the
observer actually runs at.

WHAT IT MEASURED, 30 frames of runs/probe/road_40s.mp4, Cosmos-Reason2-2B,
48-token cap:

                       all 3 fields   truncated   item loop   refused   tokens   ms
    old (the drive's)      30/30         9/30          3         3        44    408
    new (this one)         30/30         0/30          0         0        28    275

Two intermediate shapes were measured and rejected on the way, and they are
worth knowing about because both looked obviously right:

    no exemplar at all      0/30 carried all three fields -- the model stopped
                            writing the names
    a bare skeleton         0/12, and the model returned "ROAD: | TRAFFIC: |
    "ROAD: | TRAFFIC: |"    RISK:" unchanged on every frame
"""
import argparse
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import config                                              # noqa: E402
import rio_prompts as rp                                   # noqa: E402
import vision                                              # noqa: E402
from teachers import canned                                # noqa: E402

OLD_PROMPT = """Report this frame as a sensor reading for another system. Not a sentence to a person.

Exactly three fields, one line, separated by a vertical bar, at most six words each, in this order and with these names:

ROAD: lanes, surface, light | TRAFFIC: vehicles that matter and where | RISK: what could bite in the next few seconds, or none seen

Rules:
- Replace each field's description with what you actually see. Keep the field names.
- Write no angle brackets, no square brackets, no quotes around a field.
- Only what is visible in this frame. No guessing.
- If the frame cannot be read, write unreadable in each field. That is a valid reading.
- If this is not a road, say what it actually is. Do not describe a road.
- No advice, no instruction to a driver, no speed or distance in numbers unless you can read them in the frame.
- No reasoning, no preamble. The three fields only."""


def read_one(pil, prompt):
    """One generate with an explicit prompt. -> (text, truncated, ms).

    Deliberately NOT vision.observe(): that applies the guards, and the guards
    are what is being measured. This is the raw model answer, and every count
    below is taken on it.
    """
    import torch
    proc, model = vision._processor, vision._model
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil}, {"type": "text", "text": prompt}]}]
    inputs = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    kw = vision._generate_kwargs()
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, **kw)
    ms = (time.time() - t0) * 1000.0
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)[0].strip()
    trunc = vision._reading_truncated(out, inputs, kw)
    ntok = int(out.shape[1]) - int(inputs["input_ids"].shape[1])
    return vision._strip_think(text)[0].strip(), trunc, ms, ntok


def score(rows, prompt):
    n = len(rows) or 1
    echo = sum(1 for r in rows if rp.echoes_prompt(r["text"], prompt))
    cat = sum(1 for r in rows if canned.list_run(r["text"]) >= canned.LIST_FLOOR)
    iloop = sum(1 for r in rows
                if canned.item_loop_run(r["text"]) >= canned.ITEM_LOOP_FLOOR)
    wloop = sum(1 for r in rows
                if canned.word_loop_run(r["text"]) >= canned.WORD_LOOP_FLOOR)
    trunc = sum(1 for r in rows if r["trunc"])
    ref = sum(1 for r in rows if vision.reading_refused(r["text"], 0))
    fields = [len([f for f in rp.split_sensor_reading(r["text"])["fields"]
                   if f["present"]]) for r in rows]
    unknown = sum(1 for r in rows
                  if rp.split_sensor_reading(r["text"])["unknown_fields"])
    return {
        "n": n,
        "echo": echo, "catalogue": cat, "item_loop": iloop, "word_loop": wloop,
        "truncated": trunc, "refused": ref, "unknown_field": unknown,
        "all3": sum(1 for f in fields if f == 3),
        "chars_p50": int(st.median([len(r["text"]) for r in rows])),
        "tokens_p50": int(st.median([r["ntok"] for r in rows])),
        "ms_p50": int(st.median([r["ms"] for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=None)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    frames_dir = Path(a.frames) if a.frames else None
    if not frames_dir or not frames_dir.is_dir():
        print("--frames DIR of jpgs is required", file=sys.stderr)
        return 2
    from PIL import Image
    paths = sorted(frames_dir.glob("*.jpg"))[:a.n]
    print(f"\n{len(paths)} frames from {frames_dir}, "
          f"cap {vision._generate_kwargs()['max_new_tokens']} tokens, "
          f"model {config.local_vision_label()}\n")
    vision._ensure_loaded()

    arms = {"OLD (the drive's prompt)": OLD_PROMPT,
            "NEW (no template)": rp.SENSOR_PROMPT_TERSE}
    results = {}
    for name, prompt in arms.items():
        rows = []
        for i, p in enumerate(paths):
            pil = vision._downscale(Image.open(p).convert("RGB"),
                                    getattr(config, "LOCAL_VISION_MAX_SIDE", None))
            text, trunc, ms, ntok = read_one(pil, prompt)
            rows.append({"text": text, "trunc": trunc, "ms": ms, "ntok": ntok})
        results[name] = (score(rows, prompt), rows)
        print(f"  {name}: {len(rows)} readings")

    hdr = ["echo", "catalogue", "item_loop", "word_loop", "truncated",
           "unknown_field", "refused", "all3", "chars_p50", "tokens_p50", "ms_p50"]
    print(f"\n{'':<26}" + "".join(f"{h:>13}" for h in hdr))
    for name, (sc, _) in results.items():
        print(f"{name:<26}" + "".join(f"{sc[h]:>13}" for h in hdr))
    print(f"\n{'':<26}" + "".join(f"{'':>13}" for h in hdr))
    print("  counts are frames out of n; all3 = readings with all three fields present\n")

    for name, (_, rows) in results.items():
        print(f"--- {name}: first 6 readings ---")
        for r in rows[:6]:
            print(f"    {r['text'][:118]!r}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
