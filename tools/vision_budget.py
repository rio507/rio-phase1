"""vision_budget.py — where an observer reading's milliseconds actually go.

    python -m tools.vision_budget --model cosmos
    python -m tools.vision_budget --model cosmos --configs tight
    python -m tools.vision_budget --model both --json runs/budget.json

THE QUESTION. Cosmos-Reason2-2B reads a frame in 1483 ms at p50 and 5721 ms at
the tail, against an observer that wants to run at about 1 Hz. "It is a 2B, it
should be fast" and "it is slow" are both true and neither is actionable, so
this splits the number into the three things it is made of:

    prefill    the vision tower and the prompt: one forward pass over ~N image
               tokens. Fixed by the IMAGE, not by the answer. Measured as a
               generate with max_new_tokens=1.
    decode     per output token, times the number of tokens. Fixed by how much
               the model is asked to WRITE.
    the cap    what happens when it will not stop. A decoding loop fills the
               whole budget, so the budget IS the tail.

WHAT WAS FOUND BY ASKING. The 5721 ms reading was the word "interstate" repeated
203 times -- 216 words, 14 distinct -- filling a 220-token ceiling. Not slow
inference: a degenerate loop, and one that teachers.canned called CLEAN because
its loop test splits on sentence boundaries and this had no full stops in it.
The p50 is a different story and a simpler one: Cosmos writes 222 characters at
p50 where Qwen writes 57, because SENSOR_PROMPT asks for three fields and
OBSERVER_PROMPT asks for one short sentence. Per character Cosmos is FASTER
(6.5 ms/char against 9.3).

So the levers are: how many tokens it may write, how big the picture is, and
whether a loop can run to the ceiling. This measures all three against accuracy
on the same frames, because a budget that truncates the RISK field has not made
the observer faster, it has made it blind to the thing it is for.

HOW ACCURACY IS SCORED HERE, and its limits. Not by a judge model. Four things
that can be counted:

    fields      ROAD / TRAFFIC / RISK all present. A cap that cuts RISK off is
                the failure that matters most, since risk is the narrowed job.
    truncated   the generate stopped because it hit the cap rather than because
                the model finished. Truncation is not always loss -- a loop
                truncated is a loop stopped -- so it is reported beside `loop`
                rather than as a fault on its own.
    loop        longest run of one repeated word. The tail's whole story.
    probes      the adversarial frames from tools/vision_ab.py, so a config that
                buys latency by fabricating is visible as fabrication.

Whether the prose is good is not scored and the readings are printed instead.
"""
import argparse
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                                   # noqa: E402

load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))

from tools.vision_ab import (PROBES, clip_frames, synth_frame,   # noqa: E402
                             gpu_used_mib, grade)

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# The configurations, and why each one is here.
# ---------------------------------------------------------------------------
# `max_new`  the token ceiling. 220 is what shipped (LOCAL_VISION_MAX_TOKENS
#            raised to LOCAL_VISION_THINK_BUDGET for the sensor prompt, to leave
#            room for a reasoning trace that never came).
# `max_side` the long edge of the image. Prefill scales with vision tokens and
#            vision tokens scale with pixels; 512 is OBSERVER_MAX_SIDE_PX.
# `rep_pen`  repetition_penalty. NVIDIA ships 1.0, which teachers/canned.py
#            notes "invites" the loop. Raising it is a real change to the
#            model's output distribution and is measured, not assumed.
# `cache`    the KV cache implementation. "static" preallocates it, which takes
#            decode from 25.8 to 8.1 ms/token -- a 3.2x speedup and by far the
#            biggest lever here, since decode is ~97% of a reading.
#
#            IT IS NOT A FREE SPEEDUP. Measured on 15 frames, static and dynamic
#            agreed on only 5: a padded cache changes the bf16 reduction order in
#            attention and that is enough to land on different tokens. Each is
#            deterministic on its own (5 of 5 identical reads) so this is a
#            different computation rather than a noisy one -- which is why it is
#            a scored dimension here and not an optimisation applied quietly.
def _c(name, max_new=64, max_side=512, rep_pen=1.0, cache="dynamic",
       prompt=None):
    return dict(name=name, max_new=max_new, max_side=max_side,
                rep_pen=rep_pen, cache=cache, prompt=prompt)


CONFIGS = {
    "shipped": [
        _c("as shipped", max_new=220),
    ],
    "budget": [
        _c("as shipped", max_new=220),
        _c("cap 96", max_new=96),
        _c("cap 64", max_new=64),
        _c("cap 48", max_new=48),
    ],
    "tight": [
        _c("as shipped", max_new=220),
        _c("cap 64", max_new=64),
        _c("cap 64 + rep 1.05", max_new=64, rep_pen=1.05),
        _c("cap 64 + 384 px", max_new=64, max_side=384),
        _c("cap 48 + 384 px + rep 1.05", max_new=48, max_side=384, rep_pen=1.05),
    ],
    # THE DECODE LEVER, isolated. Same cap, same picture, same sampling: the
    # only thing that moves is the KV cache, so the latency difference and the
    # accuracy difference are both attributable.
    # THE OUTPUT-LENGTH LEVER, which is the one that survives this server. Decode
    # is ~24 ms/token and ~97% of a reading, so halving the tokens halves the
    # reading. Scored against field completeness, because a shorter reading that
    # has dropped RISK is not a faster observer.
    "prompt": [
        _c("full prompt, cap 96", max_new=96, prompt="sensor"),
        _c("TERSE prompt, cap 96", max_new=96, prompt="terse"),
        _c("TERSE + rep 1.05", max_new=96, rep_pen=1.05, prompt="terse"),
        _c("TERSE + rep 1.05 + cap 48", max_new=48, rep_pen=1.05,
           prompt="terse"),
    ],
    "cache": [
        _c("dynamic cap 64", max_new=64, cache="dynamic"),
        _c("STATIC cap 64", max_new=64, cache="static"),
        _c("STATIC cap 64 + rep 1.05", max_new=64, rep_pen=1.05, cache="static"),
        _c("STATIC cap 96 + rep 1.05", max_new=96, rep_pen=1.05, cache="static"),
    ],
}


def word_loop(text: str) -> int:
    """Longest run of one repeated word. -> count (1 = no repetition).

    The tail's whole story, and the thing teachers.canned.loop_run cannot see:
    it splits on sentence boundaries, and 203 consecutive "interstate" with no
    full stop in it is one sentence.
    """
    ws = (text or "").split()
    best = run = 1 if ws else 0
    for i in range(1, len(ws)):
        run = run + 1 if ws[i] == ws[i - 1] else 1
        best = max(best, run)
    return best


def fields_present(text: str) -> int:
    t = (text or "").upper()
    return sum(1 for f in ("ROAD:", "TRAFFIC:", "RISK:") if f in t)


def q(vals, f):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(f * len(s)))], 1)


def run_config(role: str, cfg: dict, frames: list, probes: list) -> dict:
    """One configuration, over the road frames and the probes.

    Generates DIRECTLY rather than through vision.observe(), because the split
    between prefill and decode needs two generates per frame and observe() is
    one. Everything else is the real path: the same processor, the same
    downscale, the same prompt. The guards are then run over the text so a
    config cannot look good by producing something a guard would refuse.
    """
    os.environ["LOCAL_VISION_MODEL"] = role
    for m in ("config", "rio_prompts", "vision"):
        sys.modules.pop(m, None)
    import torch
    import config
    import vision
    from PIL import Image

    vision.MODEL_ID = config.local_vision_model_id()
    processor, model, lock = vision.get_handles()
    prompt = vision.TEACHER_PROMPT
    if cfg.get("prompt") == "terse":
        import rio_prompts
        prompt = rio_prompts.SENSOR_PROMPT_TERSE
    elif cfg.get("prompt") == "sensor":
        import rio_prompts
        prompt = rio_prompts.SENSOR_PROMPT

    def once(jpeg, max_new, want_prefill=False):
        pil = vision._downscale(
            Image.open(io.BytesIO(jpeg)).convert("RGB"), cfg["max_side"])
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[1]
        kw = dict(max_new_tokens=max_new, do_sample=False)
        if cfg["rep_pen"] and cfg["rep_pen"] != 1.0:
            kw["repetition_penalty"] = float(cfg["rep_pen"])
        if cfg.get("cache") == "static":
            kw["cache_implementation"] = "static"
        pre_ms = None
        if want_prefill:
            torch.cuda.synchronize()
            t = time.time()
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize()
            pre_ms = (time.time() - t) * 1000.0
        torch.cuda.synchronize()
        t = time.time()
        out = model.generate(**inputs, **kw)
        torch.cuda.synchronize()
        total_ms = (time.time() - t) * 1000.0
        n_out = int(out.shape[1] - n_in)
        text = processor.batch_decode(
            out[:, n_in:], skip_special_tokens=True)[0].strip()
        text, _, unterm = vision._strip_think(text)
        return {"ms": total_ms, "prefill_ms": pre_ms, "in_tokens": n_in,
                "out_tokens": n_out, "text": "" if unterm else text,
                "truncated": n_out >= max_new}

    rows = []
    with lock:
        # One warm generate before timing anything: the first call after a load
        # pays kernel selection, and reporting that as p50 would be a lie in the
        # model's favour on a short run and against it on a long one.
        once(frames[0][1], 8)
        for i, (fid, jpeg) in enumerate(frames):
            r = once(jpeg, cfg["max_new"], want_prefill=(i < 5))
            r["frame"] = fid
            rows.append(r)
        pr = []
        for p in probes:
            r = once(synth_frame(p["synth"]), cfg["max_new"])
            g = grade(p, r["text"])
            pr.append({"id": p["id"], "text": r["text"], "ms": r["ms"], **g})

    ms = [r["ms"] for r in rows]
    outs = [r["out_tokens"] for r in rows]
    pref = [r["prefill_ms"] for r in rows if r["prefill_ms"] is not None]
    dec = [(r["ms"] - statistics.median(pref)) / max(1, r["out_tokens"] - 1)
           for r in rows] if pref else []
    return {
        "role": role, "config": cfg,
        "in_tokens": statistics.median([r["in_tokens"] for r in rows]),
        "prefill_ms_p50": q(pref, 0.5),
        "decode_ms_per_token_p50": q(dec, 0.5),
        "ms": {"p50": q(ms, 0.5), "p90": q(ms, 0.9), "max": round(max(ms), 1)},
        "out_tokens": {"p50": q(outs, 0.5), "max": max(outs)},
        "truncated": sum(1 for r in rows if r["truncated"]),
        "looped": sum(1 for r in rows if word_loop(r["text"]) >= 5),
        "max_word_run": max(word_loop(r["text"]) for r in rows),
        "fields_all3": sum(1 for r in rows if fields_present(r["text"]) == 3),
        "n": len(rows),
        "probes": pr,
        "probe_seen": sum(1 for p in pr if p["seen"]),
        "probe_fab": sum(1 for p in pr if p["fabricated"]),
        "rows": rows,
    }


def print_config(r):
    c = r["config"]
    print(f"\n--- {c['name']:28s} max_new={c['max_new']:3d} "
          f"{c['max_side']}px rep={c['rep_pen']} cache={c.get('cache','dynamic')}")
    print(f"    prefill p50        {r['prefill_ms_p50']} ms "
          f"({r['in_tokens']:.0f} input tokens)")
    print(f"    decode             {r['decode_ms_per_token_p50']} ms/token")
    print(f"    total              p50 {r['ms']['p50']} ms   "
          f"p90 {r['ms']['p90']} ms   MAX {r['ms']['max']} ms")
    print(f"    out tokens         p50 {r['out_tokens']['p50']}   "
          f"max {r['out_tokens']['max']}")
    print(f"    truncated          {r['truncated']}/{r['n']}"
          f"    looped {r['looped']}/{r['n']} (longest word run "
          f"{r['max_word_run']})")
    print(f"    all 3 fields       {r['fields_all3']}/{r['n']}")
    print(f"    probes             saw {r['probe_seen']}/5   "
          f"FABRICATED {r['probe_fab']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cosmos", "qwen", "both"],
                    default="cosmos")
    ap.add_argument("--configs", choices=sorted(CONFIGS), default="budget")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--json", default="")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    args = ap.parse_args()

    used = gpu_used_mib()
    if used and used > 2000 and not args.allow_busy_gpu:
        print(f"REFUSING: {used} MiB already in use on the card.")
        return 2

    frames = clip_frames(args.frames)
    out = {"at": time.time(), "runs": []}
    for role in (["cosmos", "qwen"] if args.model == "both" else [args.model]):
        print(f"\n{'=' * 66}\n{role}\n{'=' * 66}")
        for cfg in CONFIGS[args.configs]:
            r = run_config(role, cfg, frames, PROBES)
            print_config(r)
            out["runs"].append(r)
            # The model stays loaded across configs -- same weights, different
            # generate arguments -- so nothing is freed between them.
        for m in ("vision",):
            sys.modules.pop(m, None)
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
