"""observer_parrot_rate.py — does the observer prompt still hand back its own examples?

    python -m tools.observer_parrot_rate                  # 20 road frames, all arms
    python -m tools.observer_parrot_rate --frames 20 --json runs/parrot.json

WHY THIS EXISTS, AND WHAT THE EXISTING NUMBER ACTUALLY IS
---------------------------------------------------------
The parroting fault (108ad47, 2026-09-07) was this: OBSERVER_PROMPT ended with
four BARE example sentences, and Qwen3-VL returned the FIRST one whatever the
frame was. What was measured at the time was four ADVERSARIAL frames -- a hand
over the lens, a black frame, random noise, and one road -- and the result was
4/4 identical, "Open freeway light traffic dry hills both sides", which is
example one with its punctuation dropped.

Two things about that measurement have been quietly misremembered since, and
both of them change what a new number means:

  THE PAIRED EXAMPLES ARE THE FIX, NOT THE FAULT. The prompt that parroted was
  the bare list. 108ad47 replaced it with frame->sentence PAIRS plus an
  explicit "those are four other frames", and that is the prompt shipping
  today. Scoring "the current prompt" therefore scores the CURE.

  IT WAS NEVER TWENTY ROAD FRAMES. It was four frames, three of which were
  deliberately unreadable -- which is the condition that provokes the fault,
  because a model with nothing to describe is the one that reaches for an
  example. Real road frames are the EASY case and a low rate on them is not
  evidence the bare prompt was fine.

So this measures all three arms on the same twenty real road frames, and the
bare prompt is recovered from git rather than paraphrased, so the comparison is
against the thing that actually failed.

    bare      OBSERVER_PROMPT as of 108ad47^ -- the four sentences, unpaired.
    paired    OBSERVER_PROMPT as it ships now. THIS IS WHAT RUNS under
              LOCAL_VISION_MODEL=qwen (vision.TEACHER_PROMPT).
    sensor    SENSOR_PROMPT_TERSE, which runs under `cosmos` and not under
              `qwen`. Scored because it is the other prompt in the building and
              a rate for it costs one more pass over frames already decoded.

WHAT IS COUNTED
---------------
    verbatim    rio_prompts.is_prompt_example -- the whole example back, modulo
                punctuation. This is the guard that fires in vision.observe and
                the metric the original 4/4 was.
    example 1   how many of those were the FIRST example specifically, because
                "it completed the pattern" and "it described the road and
                happened to agree" look different in that column.
    echo        rio_prompts.echoes_prompt -- three-plus content words lifted
                off the instructions. Verbatim's quieter half: a reading can
                borrow the vocabulary without returning a whole sentence.
    distinct    unique readings across the twenty. A parrot scores 1. This is
                the number that made the original fault visible and it is the
                one that cannot be argued with.

Every reading is printed. A rate nobody can check is a rate.
"""
import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv                                    # noqa: E402

load_dotenv(str(REPO / ".env"))

import config                                                     # noqa: E402
import rio_prompts as rp                                          # noqa: E402
import vision                                                     # noqa: E402
from tools.vision_ab import clip_frames                           # noqa: E402

# THE BARE PROMPT, READ OUT OF GIT RATHER THAN RETYPED. A transcribed copy of
# the thing being compared against is a copy that is wrong the first time
# somebody fixes a typo in it, and this whole file is about a measurement that
# drifted from what it claimed to measure.
BARE_REV = "108ad47^"


def bare_prompt() -> str:
    src = subprocess.run(["git", "-C", str(REPO), "show",
                          f"{BARE_REV}:rio_prompts.py"],
                         capture_output=True, text=True, check=True).stdout
    marker = 'OBSERVER_PROMPT = """'
    i = src.index(marker) + len(marker)
    return src[i:src.index('"""', i)]


# EVERY ARM IS SCORED AGAINST ITS OWN EXAMPLES, and getting this wrong is the
# easiest way to publish a clean number that means nothing. rio_prompts.
# is_prompt_example compares against whatever OBSERVER_EXAMPLES holds TODAY,
# so the moment the shipped examples change -- which is exactly what a run of
# this tool tends to cause -- the historical arm is being asked whether it
# returned one of the NEW examples, which it has never seen. It scores 0 and
# the bare prompt looks fixed.
#
# So the examples are extracted from each arm's OWN prompt text, by the same
# normalisation the guard uses.
def example_keys(prompt: str) -> set:
    """The sentences this prompt offers as examples. -> {normalised}

    BY SHAPE, NOT BY A LIST OF FIRST WORDS. Both forms of the prompt indent
    their examples and nothing else, so an indented line IS an example: the
    paired form writes "frame: ... -> <sentence>" and the bare form writes the
    sentence alone. Keying off the wording instead would be one more
    transcribed copy of the examples, which is the fault this whole file is
    about -- it would go stale the next time somebody edits them, and go stale
    silently, reporting zero.
    """
    keys = set()
    for line in prompt.splitlines():
        if not line[:1].isspace() or not line.strip():
            continue
        body = line.split("->", 1)[1] if "->" in line else line
        keys.add(rp._normalise(body))
    # ...UNION WHATEVER THE MODULE ITSELF CALLS AN EXAMPLE, where the prompt
    # actually contains it. SENSOR_PROMPT_TERSE sets its worked example flush
    # left under "The shape, on a DIFFERENT road from this one:", so the
    # indentation rule alone would report that prompt as offering none and
    # score it as incapable of being copied. Asked of rio_prompts rather than
    # matched on its wording, so a fifth example added there is covered here
    # without anybody remembering to come back.
    for ex in list(rp.OBSERVER_EXAMPLES) + [rp.SENSOR_EXAMPLE]:
        if ex in prompt:
            keys.add(rp._normalise(ex))
    return keys


def is_example_of(text: str, keys: set) -> bool:
    return rp._normalise(text) in keys


def read_one(pil, prompt):
    """One generate with an explicit prompt. -> (text, ms).

    Deliberately NOT vision.observe(): that applies the guards, and the guards
    are what is being measured. A refused reading would come back as "" and an
    empty string parrots nothing, which would score the fault as absent.
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
    return vision._strip_think(text)[0].strip(), ms


def _first_example(prompt: str) -> str:
    """The FIRST example the prompt offers, in prompt order -- the one the
    2026-09-07 fault came back as on every frame."""
    for line in prompt.splitlines():
        if not line[:1].isspace() or not line.strip():
            continue
        body = line.split("->", 1)[1] if "->" in line else line
        return rp._normalise(body)
    return ""


def score(rows, prompt):
    n = len(rows) or 1
    keys = example_keys(prompt)
    first = _first_example(prompt)
    verbatim = [r for r in rows if is_example_of(r["text"], keys)]
    return {
        "n": len(rows),
        "verbatim": len(verbatim),
        "verbatim_rate": round(len(verbatim) / n, 3),
        "example_1": sum(1 for r in verbatim
                         if rp._normalise(r["text"]) == first),
        "echo": sum(1 for r in rows if rp.echoes_prompt(r["text"], prompt)),
        "examples_offered": len(keys),
        "distinct": len({rp._normalise(r["text"]) for r in rows}),
        "chars_p50": int(st.median([len(r["text"]) for r in rows])),
        "ms_p50": int(st.median([r["ms"] for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    arms = [
        ("bare   (108ad47^, unpaired)", bare_prompt()),
        ("paired (ships under qwen)", rp.OBSERVER_PROMPT),
        ("sensor (terse, cosmos arm)", rp.SENSOR_PROMPT_TERSE),
    ]

    from PIL import Image
    import io
    frames = clip_frames(a.frames)
    print(f"\n{len(frames)} distinct road frames, "
          f"model {config.local_vision_label()} "
          f"(role {config.LOCAL_VISION_MODEL}), "
          f"cap {vision._generate_kwargs()['max_new_tokens']} tokens\n")
    vision._ensure_loaded()

    pils = [vision._downscale(Image.open(io.BytesIO(j)).convert("RGB"),
                              getattr(config, "LOCAL_VISION_MAX_SIDE", None))
            for _, j in frames]

    results = {}
    for name, prompt in arms:
        rows = []
        for pil in pils:
            text, ms = read_one(pil, prompt)
            rows.append({"text": text, "ms": ms})
        results[name] = (score(rows, prompt), rows)
        print(f"  {name}: {len(rows)} readings")

    hdr = ["n", "examples_offered", "verbatim", "example_1", "echo",
           "distinct", "chars_p50", "ms_p50"]
    W = max(11, max(len(h) for h in hdr) + 2)
    print(f"\n{'':<30}" + "".join(f"{h:>{W}}" for h in hdr))
    for name, (sc, _) in results.items():
        print(f"{name:<30}" + "".join(f"{sc[h]:>{W}}" for h in hdr))
    print("\n  verbatim  = is_prompt_example: a whole example back, out of n")
    print("  distinct  = unique readings; a parrot scores 1\n")

    for name, (_, rows) in results.items():
        print(f"--- {name} ---")
        for i, r in enumerate(rows):
            mark = ("  <-- PROMPT EXAMPLE"
                    if is_example_of(r["text"], example_keys(
                        dict(arms)[name])) else "")
            print(f"  {i:2d} {r['text'][:104]!r}{mark}")
        print()

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"at": time.time(),
                   "model": config.local_vision_label(),
                   "role": config.LOCAL_VISION_MODEL,
                   "frames": [f for f, _ in frames],
                   "arms": {k: {"score": v[0], "rows": v[1]}
                            for k, v in results.items()}},
                  open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
