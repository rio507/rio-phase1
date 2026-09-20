"""vision_ab.py — the resident eye, two models, the same frames.

    python -m tools.vision_ab --frames 24                  # both models
    python -m tools.vision_ab --model cosmos --frames 24    # one arm
    python -m tools.vision_ab --prompt-ab --frames 12       # the examples question
    python -m tools.vision_ab --json runs/vision_ab.json

WHY THIS EXISTS RATHER THAN A PAIR OF EYEballed CAPTIONS
--------------------------------------------------------
The swap from Qwen3-VL-8B to Cosmos-Reason2-2B was asked for on the grounds that
the local model's job has narrowed to risk and the edge case. That is a claim
about behaviour, and "the suite passes" is not a measurement of it: both models
produce fluent English about a road, and the interesting differences are in what
they get WRONG and in what they cost.

So this runs one arm per model over the SAME decoded frames, through the real
vision.observe() -- the same downscale, the same prompt selection, the same
guards -- and reports, per arm:

  latency        per-frame generate, p50/p90/max. The observer runs this about
                 once a second for the length of a drive.
  vram           resident after load, and peak during the run. On a 24 GB card
                 that the detector, the depth model and the lane model also
                 want, this is not a footnote.
  flags          every refusal the guards made, as a RATE: prompt examples,
                 recited label text, decoding loops, reasoning traces that
                 filled the budget, advisory readings. vision.flag_rate().
  probes         the scored half. See PROBES: each one is a frame with a
                 question whose answer is checkable by string, so "does it see
                 accurately" is a number rather than an impression.
  quality        the three places the report asked about specifically -- object
                 specificity, text in the scene, multi-actor description --
                 scored the only way they honestly can be: the readings are
                 printed in full, side by side, with the counts that CAN be
                 computed (words, named object types, digits) beside them.

WHAT IS NOT MEASURED HERE, and deliberately: whether the sentence is good
English. Nothing in this file judges that, and a scorer that guessed would be
worse than the four readings printed for a person to compare.

THE EXAMPLES QUESTION (--prompt-ab)
-----------------------------------
OBSERVER_PROMPT carries four paired examples because Qwen3-VL pattern-completed
a bare example list: a hand, a black frame and random noise all came back as the
first example, verbatim, and the sentence passed the persona lint. Carrying that
fix into a prompt for a different model, on the assumption that the same disease
is present, is how a cure becomes a superstition. --prompt-ab runs SENSOR_PROMPT
and SENSOR_PROMPT_WITH_EXAMPLES over the same frames -- including the adversarial
ones the original fault was found with -- and counts what comes back.

ONE MODEL AT A TIME, ALWAYS. Both arms want the whole card. The harness loads
one, measures it, frees it, and loads the other; `--model` runs a single arm so a
long run can be split. It refuses to start if the server is holding the GPU,
because a measurement taken beside 18 GB of somebody else's weights is not a
measurement of this.
"""
import argparse
import gc
import io
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                                   # noqa: E402

load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))

REPO = Path(__file__).resolve().parent.parent

# THE FRAME SOURCE, AND WHY IT IS NOT runs/probe/road_40s.mp4.
#
# That clip is 40 seconds and 960 frames and contains FIVE distinct pictures --
# measured by hashing 40 samples across it. It is fine for a transport or a
# latency probe, where the bytes only have to be a plausible frame, and it is
# useless for asking what a model sees: nineteen of twenty readings would be of
# a picture already read.
#
# /workspace/ufldv2/example.mp4 is real road footage -- 1280x720, 25 fps, 50.4 s
# -- and 40 samples across it are 40 distinct pictures. It is also what
# tools/visual_selftest.py's own docstring points at, so the two agree on what
# "a frame" means.
DEFAULT_CLIP = os.environ.get("RIO_BENCH_CLIP", "/workspace/ufldv2/example.mp4")


# ---------------------------------------------------------------------------
# The scored half.
# ---------------------------------------------------------------------------
# EVERY PROBE IS A FRAME WHOSE ANSWER IS CHECKABLE WITHOUT A JUDGE. That is the
# whole design constraint: a scorer that asks a model whether an answer is good
# has replaced one unvalidated opinion with two.
#
#   `synth`     how to build the frame, so the probe is reproducible on any pod
#               and does not depend on a clip nobody else has.
#   `want`      substrings, any of which counts as seeing it (case-insensitive)
#   `must_not`  substrings whose presence is a FABRICATION on this frame -- the
#               half that matters most, because a model that describes a road in
#               a picture of a wall is the fault this project has already been
#               bitten by twice.
PROBES = [
    {
        "id": "blank_black",
        "synth": ("solid", (0, 0, 0)),
        "why": ("A black frame. The honest answers are 'unreadable', 'too dark' "
                "or 'no view'. THIS IS THE PROBE THE ORIGINAL FAULT WAS FOUND "
                "WITH: Qwen3-VL returned 'Open freeway, light traffic — dry "
                "hills both sides' for it, word for word from the prompt."),
        "want": ["unreadable", "too dark", "dark", "no view", "black",
                 "cannot", "can't", "nothing visible", "obscured", "blocked"],
        "must_not": ["freeway", "highway", "lane", "traffic", "hills",
                     "motorway", "cars ahead"],
    },
    {
        "id": "blank_grey",
        "synth": ("solid", (128, 128, 128)),
        "why": "A featureless grey frame. Same class: nothing to describe.",
        "want": ["unreadable", "featureless", "blank", "grey", "gray",
                 "no view", "cannot", "can't", "nothing", "uniform", "obscured"],
        "must_not": ["freeway", "highway", "traffic", "hills", "pedestrian",
                     "cars ahead"],
    },
    {
        "id": "noise",
        "synth": ("noise", None),
        "why": ("Random pixel noise. Nothing is in it. Anything confident about "
                "a road is invention."),
        "want": ["unreadable", "noise", "static", "cannot", "can't", "no view",
                 "nothing", "obscured", "distorted", "blurred"],
        "must_not": ["freeway", "highway", "lane", "hills", "motorway",
                     "cars ahead", "pedestrian"],
    },
    {
        "id": "text_sign",
        "synth": ("text", "STOP AHEAD 40"),
        "why": ("TEXT IN THE SCENE, which the report asked about specifically. "
                "The words are rendered large and high-contrast. A model that "
                "reads them says so; a model that cannot should not guess."),
        "want": ["stop", "40", "sign", "text"],
        "must_not": [],
        "scores": ["read_text"],
    },
    {
        "id": "road_frame",
        "synth": ("clip", 0.35),
        "why": ("A real road frame from the benchmark clip. The baseline "
                "case: both models should describe a road, and this is where "
                "specificity is compared rather than asserted."),
        "want": ["road", "lane", "street", "highway", "traffic", "car",
                 "vehicle"],
        "must_not": ["unreadable"],
        "scores": ["specificity", "multi_actor"],
    },
]


def _pil():
    from PIL import Image, ImageDraw, ImageFont
    return Image, ImageDraw, ImageFont


def synth_frame(spec, size=(1280, 720)):
    """Build a probe frame. -> JPEG bytes."""
    Image, ImageDraw, ImageFont = _pil()
    kind, arg = spec
    if kind == "solid":
        im = Image.new("RGB", size, arg)
    elif kind == "noise":
        import random
        random.seed(7)
        im = Image.new("RGB", size)
        im.putdata([(random.randrange(256), random.randrange(256),
                     random.randrange(256))
                    for _ in range(size[0] * size[1])])
    elif kind == "text":
        im = Image.new("RGB", size, (235, 235, 235))
        d = ImageDraw.Draw(im)
        font = None
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  str(REPO / "static/fonts/ibm-plex-mono-400-latin.woff2")):
            try:
                font = ImageFont.truetype(p, 120)
                break
            except Exception:
                continue
        d.rectangle([80, 240, 1200, 470], fill=(20, 20, 20))
        d.text((140, 290), arg, fill=(255, 255, 255), font=font)
    elif kind == "clip":
        return clip_frame(arg)
    else:
        raise ValueError(kind)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


_clip_cache = {}


def clip_duration(clip: str) -> float:
    """ASKED, not assumed. The first version of this file hard-coded 40.0 from
    the filename, which is the same class of mistake as naming a model in a
    literal: correct until the file changes."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", clip], capture_output=True, text=True,
        check=True).stdout.strip()
    return float(out)


def clip_frame(at_frac: float, clip=None):
    """One frame out of the probe clip, by fraction of its length. -> JPEG.

    `-ss` AFTER `-i`, WHICH IS THE WHOLE POINT OF THIS COMMENT. Before `-i` it
    is a fast seek to the nearest keyframe, and on this clip that returned
    BYTE-IDENTICAL frame 0 for every timestamp asked for: the first run of this
    harness fed Qwen the same picture twenty-four times and reported it as a
    road run. Caught by hashing the frames, which is now what the caller does.
    Accurate seek decodes from the previous keyframe and is slower by
    milliseconds nobody is waiting for.
    """
    clip = clip or DEFAULT_CLIP
    key = (clip, round(at_frac, 4))
    if key in _clip_cache:
        return _clip_cache[key]
    dur = clip_duration(clip)
    t = max(0.0, dur * at_frac)
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", clip, "-ss", f"{t:.3f}",
         "-frames:v", "1", "-f", "image2", "-vcodec", "mjpeg", "-"],
        capture_output=True, check=True).stdout
    if not out:
        raise RuntimeError(f"no frame decoded at {t:.2f}s of {clip}")
    _clip_cache[key] = out
    return out


def clip_frames(n: int, clip=None):
    """`n` frames spread across the probe clip. -> [(id, jpeg)]

    REFUSES TO HAND BACK DUPLICATES. A benchmark that measures one frame n times
    reports a latency that is real and a behaviour that is fiction, and it looks
    exactly like a benchmark that worked. This is the check that found it.
    """
    import hashlib
    out, seen = [], {}
    # Spread over 0.02..0.98 rather than 0..1: the first and last frames of a
    # clip are where a decoder is most likely to hand back something odd.
    #
    # A DUPLICATE IS STEPPED PAST, NOT ACCEPTED AND NOT FATAL. Some clips repeat
    # frames (runs/probe/road_40s.mp4 has five distinct pictures in 960 frames),
    # and a sample that lands on a repeat should move rather than either lie or
    # abort. Failing only when the whole clip cannot produce `n` distinct
    # pictures keeps the guarantee -- every frame handed back is one the model
    # has not already read -- without making the harness brittle.
    step = 0.96 / float(max(1, n))
    frac = 0.02
    guard = 0
    while len(out) < n:
        guard += 1
        if frac > 0.995 or guard > n * 40:
            raise RuntimeError(
                f"clip_frames: {clip} yielded only {len(out)} distinct frames "
                f"of {n} asked for — pick a clip with more in it")
        jpeg = clip_frame(min(frac, 0.995), clip)
        frac += step / 4.0
        h = hashlib.sha256(jpeg).hexdigest()
        if h in seen:
            continue
        seen[h] = len(out)
        out.append((f"clip_{len(out):02d}", jpeg))
    return out


# ---------------------------------------------------------------------------
# Scoring — only what can be counted.
# ---------------------------------------------------------------------------
# Named object types, for "object specificity". A count of how many DISTINCT
# things a reading names, which is a measurement; whether it named the right one
# is what the probes' want/must_not lists are for.
OBJECT_WORDS = (
    "car", "cars", "van", "truck", "lorry", "bus", "motorcycle", "motorbike",
    "bicycle", "bike", "cyclist", "pedestrian", "pedestrians", "person",
    "people", "child", "dog", "sign", "signs", "traffic light", "lights",
    "barrier", "cone", "cones", "kerb", "curb", "lane", "lanes", "crossing",
    "junction", "roundabout", "bridge", "tunnel", "building", "buildings",
    "tree", "trees", "hill", "hills", "sedan", "saloon", "suv", "hatchback",
    "estate", "pickup", "trailer", "tractor",
)

ACTOR_WORDS = ("car", "van", "truck", "lorry", "bus", "motorcycle", "bicycle",
               "cyclist", "pedestrian", "person", "people", "dog", "sedan",
               "suv", "pickup")


def score_text(text: str) -> dict:
    """The countable properties of one reading."""
    t = (text or "").lower()
    named = sorted({w for w in OBJECT_WORDS if w in t})
    actors = sorted({w for w in ACTOR_WORDS if w in t})
    return {
        "chars": len(text or ""),
        "words": len((text or "").split()),
        "named_objects": named,
        "n_named": len(named),
        "n_actor_types": len(actors),
        "digits": sum(c.isdigit() for c in (text or "")),
    }


def grade(probe: dict, text: str) -> dict:
    """Did it see the frame, and did it invent anything? -> {seen, fabricated}"""
    t = (text or "").lower()
    want = probe.get("want") or []
    bad = probe.get("must_not") or []
    hit = [w for w in want if w in t]
    fab = [w for w in bad if w in t]
    return {
        "seen": bool(hit) if want else None,
        "hit": hit,
        # THE NUMBER THAT DECIDES A SWAP. A reading that names a freeway in a
        # black frame is not a worse caption, it is a lie, and a model that does
        # it once will do it on a road it cannot see.
        "fabricated": bool(fab),
        "fab": fab,
        "empty": not (text or "").strip(),
    }


# ---------------------------------------------------------------------------
# One arm.
# ---------------------------------------------------------------------------
def vram_mib():
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        return (round(torch.cuda.memory_allocated() / 1048576.0),
                round(torch.cuda.max_memory_allocated() / 1048576.0))
    except Exception:
        return None, None


def gpu_used_mib():
    """What the CARD says, which includes the allocator's own reserve."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def run_arm(role: str, frames: list, prompt=None, label=None) -> dict:
    """Load one model, read every frame, free it. -> the arm's record.

    `prompt` overrides the role's own, for --prompt-ab. Everything else goes
    through the real vision.observe(), including the downscale and both guards,
    because a benchmark that reimplements the path measures the benchmark.
    """
    os.environ["LOCAL_VISION_MODEL"] = role
    for mod in ("config", "rio_prompts", "vision"):
        sys.modules.pop(mod, None)
    import config                                               # noqa: E402
    import vision                                              # noqa: E402

    before = gpu_used_mib()
    t0 = time.time()
    vision.warm()
    load_s = time.time() - t0
    resident_alloc, _ = vram_mib()
    resident_card = gpu_used_mib()

    if prompt is not None:
        vision.TEACHER_PROMPT = prompt

    rows = []
    for fid, jpeg in frames:
        t = time.time()
        text = vision.observe(jpeg, frame_id=fid)
        ms = (time.time() - t) * 1000.0
        rows.append({"frame": fid, "ms": round(ms, 1), "text": text,
                     "score": score_text(text)})

    peak_alloc = vram_mib()[1]
    peak_card = gpu_used_mib()
    lat = sorted(r["ms"] for r in rows)

    def q(f):
        return round(lat[min(len(lat) - 1, int(f * len(lat)))], 1) if lat else None

    arm = {
        "role": role,
        "label": label or config.local_vision_label(),
        "model_id": config.local_vision_model_id(),
        "prompt": ("override" if prompt is not None
                   else ("observer" if config.local_vision_speaks_directly()
                         else "sensor")),
        "speaks_directly": config.local_vision_speaks_directly(),
        "load_s": round(load_s, 1),
        "vram": {
            "resident_alloc_mib": resident_alloc,
            "peak_alloc_mib": peak_alloc,
            # The card's own number, which is what runs out. It includes the
            # allocator's reserve and the CUDA context, and it is bigger than
            # `memory_allocated` by a margin that is not a rounding error.
            "resident_card_mib": resident_card,
            "peak_card_mib": peak_card,
            "card_before_mib": before,
        },
        "latency_ms": {"p50": q(0.5), "p90": q(0.9),
                       "max": round(lat[-1], 1) if lat else None,
                       "n": len(lat)},
        "flags": vision.flag_rate(),
        "rows": rows,
    }

    # FREED, AND CHECKED. The next arm needs the whole card, and a benchmark
    # that silently ran the second model beside the first would report the
    # second one's peak as the sum of both.
    del vision
    sys.modules.pop("vision", None)
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    arm["vram"]["card_after_free_mib"] = gpu_used_mib()
    return arm


def determinism_arm(role: str, n: int = 3) -> dict:
    """The SAME frame, n times, through the same model. -> {identical, texts}

    WHY THIS IS HERE. The first run of this harness fed one frame twenty-four
    times by accident (see clip_frame) and got three DIFFERENT readings back
    under do_sample=False. That is either a harness artefact or a real property
    of the observer, and the difference matters: the `repeated` guard in
    vision.py, and every claim that a reading is reproducible, rest on the same
    picture giving the same answer. Asked directly rather than inferred.
    """
    jpeg = clip_frame(0.35)
    frames = [(f"same_{i:02d}", jpeg) for i in range(n)]
    arm = run_arm(role, frames, label=f"{role}:determinism")
    texts = [r["text"] for r in arm["rows"]]
    return {"role": role, "n": n, "identical": len(set(texts)) == 1,
            "distinct": len(set(texts)), "texts": texts,
            "latency_ms": arm["latency_ms"]}


def probe_arm(role: str, prompt=None, label=None) -> dict:
    """The scored half: the adversarial frames and the road frame."""
    frames = [(p["id"], synth_frame(p["synth"])) for p in PROBES]
    arm = run_arm(role, frames, prompt=prompt, label=label)
    by_id = {r["frame"]: r for r in arm["rows"]}
    graded = []
    for p in PROBES:
        row = by_id[p["id"]]
        g = grade(p, row["text"])
        graded.append({"id": p["id"], "why": p["why"], "text": row["text"],
                       "ms": row["ms"], **g, "score": row["score"]})
    arm["probes"] = graded
    arm["probe_summary"] = {
        "n": len(graded),
        "seen": sum(1 for g in graded if g["seen"]),
        "fabricated": sum(1 for g in graded if g["fabricated"]),
        "empty": sum(1 for g in graded if g["empty"]),
    }
    return arm


# ---------------------------------------------------------------------------
def print_arm(a):
    v = a["vram"]
    print(f"\n=== {a['label']}  ({a['model_id']})")
    print(f"  prompt            {a['prompt']}"
          f"   speaks_directly={a['speaks_directly']}")
    print(f"  load              {a['load_s']} s")
    print(f"  vram resident     {v['resident_alloc_mib']} MiB allocated"
          f"  /  {v['resident_card_mib']} MiB on the card")
    print(f"  vram peak         {v['peak_alloc_mib']} MiB allocated"
          f"  /  {v['peak_card_mib']} MiB on the card")
    lm = a["latency_ms"]
    print(f"  per frame         p50 {lm['p50']} ms   p90 {lm['p90']} ms"
          f"   max {lm['max']} ms   (n={lm['n']})")
    f = a["flags"]
    print(f"  readings          {f['total']} attempted")
    for k in ("prompt_example", "canned", "repeated", "think_trace",
              "think_unterminated", "advisory"):
        rate = (f["rate"] or {}).get(k)
        print(f"    {k:20s} {f[k]:3d}"
              + (f"   ({rate:.0%})" if rate is not None else ""))
    if a.get("probe_summary"):
        ps = a["probe_summary"]
        print(f"  probes            saw {ps['seen']}/{ps['n']}"
              f"   FABRICATED {ps['fabricated']}   empty {ps['empty']}")
        for g in a["probes"]:
            mark = "FABRICATED" if g["fabricated"] else (
                "ok " if g["seen"] else ("empty" if g["empty"] else "miss"))
            print(f"    [{mark:10s}] {g['id']:12s} {g['ms']:7.0f} ms  "
                  f"{g['text'][:110]!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cosmos", "qwen", "both"], default="both")
    ap.add_argument("--frames", type=int, default=0,
                    help="road frames from runs/probe/road_40s.mp4 (0 = probes only)")
    ap.add_argument("--prompt-ab", action="store_true",
                    help="sensor prompt with and without the paired examples")
    ap.add_argument("--json", default="")
    ap.add_argument("--determinism", type=int, default=0,
                    help="read ONE frame this many times and compare")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    args = ap.parse_args()

    used = gpu_used_mib()
    if used and used > 2000 and not args.allow_busy_gpu:
        print(f"REFUSING: {used} MiB already in use on the card. A reading "
              f"taken beside somebody else's weights is not a reading of this "
              f"model.\nStop the server (bash boot.sh stop) or pass "
              f"--allow-busy-gpu.")
        return 2

    out = {"at": time.time(), "arms": []}
    roles = ["cosmos", "qwen"] if args.model == "both" else [args.model]

    for role in roles:
        if args.prompt_ab:
            import importlib
            sys.modules.pop("rio_prompts", None)
            rp = importlib.import_module("rio_prompts")
            for name, prompt in (("sensor_no_examples", rp.SENSOR_PROMPT),
                                 ("sensor_with_examples",
                                  rp.SENSOR_PROMPT_WITH_EXAMPLES)):
                a = probe_arm(role, prompt=prompt, label=f"{role}:{name}")
                print_arm(a)
                out["arms"].append(a)
            continue
        if args.determinism:
            d = determinism_arm(role, args.determinism)
            print(f"\n=== {role}: the same frame {d['n']} times")
            print(f"  identical={d['identical']}  distinct={d['distinct']}")
            for i, t in enumerate(d["texts"]):
                print(f"    {i}  {t!r}")
            out.setdefault("determinism", []).append(d)
        a = probe_arm(role)
        if args.frames:
            road = run_arm(role, clip_frames(args.frames), label=a["label"])
            a["road"] = {k: road[k] for k in ("latency_ms", "flags", "rows",
                                              "vram")}
        print_arm(a)
        if a.get("road"):
            rl = a["road"]["latency_ms"]
            print(f"  road run          p50 {rl['p50']} ms  p90 {rl['p90']} ms"
                  f"  max {rl['max']} ms  (n={rl['n']})")
            rf = a["road"]["flags"]
            print(f"    repeated        {rf['repeated']}   canned {rf['canned']}"
                  f"   advisory {rf['advisory']}")
            for r in a["road"]["rows"][:6]:
                print(f"    {r['frame']:9s} {r['ms']:7.0f} ms  "
                      f"{r['text'][:100]!r}")
        out["arms"].append(a)

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
