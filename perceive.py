"""The caption, and the deterministic geometry that goes under it.

/observe answers "what do you see?" in one sentence. /perceive answers the same
question and returns the ego corridor and the detected lane lines with it. It is
additive -- /observe is untouched and still serves RIO's voice path.

NOTHING HERE PRODUCES A BOX THE OVERLAY DRAWS, AND THAT IS THE POINT.
It used to. Qwen was asked to ground its caption, the grounded boxes came back
with a distance measured inside each one, and the dashboard drew them. The
boxes were the right shape and the wrong place often enough to matter: a
pedestrian bracket a hundred and fifty pixels off the pedestrians, a car-sized
rectangle on empty tarmac with a confident number under it. A language model is
a superb describer of a road scene and is not a detector, and mixing its output
into the same overlay as RF-DETR's spent the credibility of the boxes that were
right on the ones that were not.

So the drawn boxes have ONE source now -- RF-DETR and the tracker, through
/headway_frame -- and Qwen keeps the job it is actually good at. What it says it
saw still ships, as `qwen_boxes`, because a caption is easier to review next to
the pixels it was talking about; they are a record of a claim, they carry no
measured distance, and nothing draws them. The depth pass that existed to put a
number inside each one has gone with them.

Everything geometric here (the corridor, the lane lines) is deterministic and
comes out of headway/, per the design's LLM firewall: Qwen says what the road
looks like, never where anything is.

Model sharing: this lends vision.py's already-resident Qwen3-VL to
headway.anchor through set_qwen_provider(), so a /perceive never pulls a second
copy of the weights.
"""
import io
import json
import re
import time

import numpy as np
from PIL import Image

import config
import vision
from headway import anchor as anchor_mod
from headway import lanes as lanes_mod

# Lend the app's Qwen to headway. Import-time so any entry point into this
# module (endpoint, warm, self-test) is covered before the first anchor call.
anchor_mod.set_qwen_provider(vision.get_handles)

# These went with the boxes. VEHICLE_LABELS and VULNERABLE_LABELS told the
# overlay which colour to stroke a Qwen box in; LEAD_LABELS said which of them
# could be called the lead. Nothing here draws and nothing here picks a lead any
# more, and headway/membership.py has owned the real versions of both lists for
# as long as the detector has -- so a second copy under this roof could only
# ever drift away from the one that decides something.

# Prefill scales with vision tokens, so the frame is downscaled for the model
# call. Boxes come back normalised (0-1000) and are rescaled to the ORIGINAL
# pixel size regardless, so this costs nothing in coordinate accuracy.
MAX_SIDE_PX = 768

# Decode, not prefill, is what sets latency here: the reply is token-serial and
# every object costs tokens. A verbose {"label": ..., "bbox_2d": [...]} object
# runs ~28 tokens; the flat ["car", x1, y1, x2, y2] form runs ~13, which is why
# the schema below is terse at the cost of being less self-describing. Measured
# on this GPU: 9 objects verbose = 5.7 s, the same frame compact = see README
# note. It is still strict JSON -- the format these models follow most reliably.
# Measured on this GPU: ~52 decode tokens/s, ~13 tokens per compact object.
# Six objects plus the caption is what fits the latency budget; asking for more
# only means the cap truncates the tail anyway.
MAX_OBJECTS = 6
MAX_NEW_TOKENS = 130

PERCEIVE_PROMPT = (
    "Look at this forward-facing driving frame.\n"
    "Report the caption and the objects you can see.\n"
    "Labels allowed: car, truck, bus, van, motorcycle, bicycle, pedestrian.\n"
    "Each object is [label, x1, y1, x2, y2] using integer coordinates.\n"
    # The model will otherwise emit the label once and send bare coordinate
    # arrays for the rest, which costs the class of every object after the
    # first -- including, potentially, the lead vehicle.
    "EVERY object must begin with its label in quotes, even when the label "
    "repeats.\n"
    f"List at most {MAX_OBJECTS} objects, closest (largest) first.\n"
    "The caption is one factual sentence of 8 words or fewer.\n"
    "Reply with ONLY this JSON and nothing else:\n"
    '{"c": "<caption>", "o": [["car", x1, y1, x2, y2], ["truck", x1, y1, x2, y2]]}\n'
    'If nothing is visible use an empty list: {"c": "<caption>", "o": []}'
)



# ---------------------------------------------------------------------------
# Qwen call
# ---------------------------------------------------------------------------
def _query_qwen(pil: Image.Image) -> str:
    processor, model, lock = anchor_mod.qwen_handles()
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text", "text": PERCEIVE_PROMPT},
    ]}]
    import torch
    with lock:
        inputs = processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        return processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()


def _downscale(pil: Image.Image):
    """Shrink the long side to MAX_SIDE_PX for the model call only."""
    w, h = pil.size
    longest = max(w, h)
    if longest <= MAX_SIDE_PX:
        return pil
    s = MAX_SIDE_PX / float(longest)
    return pil.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      Image.BILINEAR)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# Both of these now live in headway.anchor, which needs them too: its enumerated
# anchor prompt gets truncated by the same token cap and has to survive it the
# same way. One implementation, imported in the direction that already exists
# (perceive -> headway), so the two paths cannot drift apart on how a cut-off
# reply is recovered.
_strip_fence = anchor_mod.strip_fence
_repair = anchor_mod.repair_truncated_json


def _parse(text: str, width: int, height: int):
    """Qwen's reply -> (caption, [(label, (x1,y1,x2,y2)), ...]) in frame pixels.

    Coordinate rescaling is delegated to anchor._rescale, which is the module
    that already knows this checkpoint emits 0-1000 normalised coordinates. Two
    copies of that rule would be one copy too many.
    """
    t = _strip_fence(text)
    caption = ""
    objects = []

    obj = None
    # Greedy span first: the whole reply should be one JSON object, and a lazy
    # match would stop at the first nested brace inside the object list.
    for pattern in (r"\{.*\}", r"\{.*?\}"):
        m = re.search(pattern, t, re.S)
        if not m:
            continue
        try:
            cand = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict):
            obj = cand
            break
    if obj is None:
        obj = _repair(t)

    def _nums(seq):
        return [float(n) for n in seq] if all(
            isinstance(n, (int, float)) and not isinstance(n, bool) for n in seq
        ) else None

    if obj is not None:
        for key in ("c", "caption"):
            cap = obj.get(key)
            if isinstance(cap, str) and cap.strip():
                caption = cap.strip()
                break

        raw_objects = obj.get("o")
        if not isinstance(raw_objects, list):
            raw_objects = obj.get("objects")
        if isinstance(raw_objects, list):
            for entry in raw_objects:
                # Compact form: ["car", x1, y1, x2, y2].
                if isinstance(entry, (list, tuple)):
                    if len(entry) == 5 and isinstance(entry[0], str):
                        box = _nums(entry[1:])
                        if box:
                            objects.append((entry[0].strip().lower(), box))
                    elif len(entry) == 4:
                        box = _nums(entry)
                        if box:
                            objects.append(("object", box))
                    continue
                # Verbose form, still accepted: {"label": ..., "bbox_2d": [...]}.
                if not isinstance(entry, dict):
                    continue
                box = None
                for key in ("bbox_2d", "bbox", "box"):
                    val = entry.get(key)
                    if isinstance(val, (list, tuple)) and len(val) == 4:
                        box = _nums(val)
                        if box:
                            break
                if box is None:
                    continue
                label = entry.get("label")
                objects.append((str(label).strip().lower() if label else "object", box))

    # Fallback: the JSON did not parse (truncated by MAX_NEW_TOKENS, or prose
    # crept in). Recover label/box pairs positionally rather than losing the
    # whole frame.
    num = r"(-?\d+\.?\d*)"
    if not objects:
        for m in re.finditer(
            r'"([A-Za-z_ ]+)"\s*,\s*' + num + r"\s*,\s*" + num + r"\s*,\s*"
            + num + r"\s*,\s*" + num, t
        ):
            objects.append((m.group(1).strip().lower(),
                            [float(m.group(i)) for i in range(2, 6)]))
    if not objects:
        for m in re.finditer(
            r'"label"\s*:\s*"([^"]+)"[^{}]*?\[\s*' + num + r"\s*,\s*"
            + num + r"\s*,\s*" + num + r"\s*,\s*" + num + r"\s*\]", t, re.S
        ):
            objects.append((m.group(1).strip().lower(),
                            [float(m.group(i)) for i in range(2, 6)]))
    # Last resort: bare coordinate arrays with the label dropped entirely. The
    # class is unknown, so it stays "object" rather than being guessed -- an
    # object drawn in the default style is honest, a pedestrian mislabelled as
    # a car is not.
    if not objects:
        for m in re.finditer(
            r"\[\s*" + num + r"\s*,\s*" + num + r"\s*,\s*"
            + num + r"\s*,\s*" + num + r"\s*\]", t
        ):
            objects.append(("object", [float(m.group(i)) for i in range(1, 5)]))
    if not caption:
        m = re.search(r'"(?:c|caption)"\s*:\s*"([^"]*)"', t)
        if m:
            caption = m.group(1).strip()

    out = []
    for label, box in objects:
        x1, y1, x2, y2 = anchor_mod._rescale(box, width, height)
        # Degenerate slivers are parse noise, not objects.
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            out.append((label, (x1, y1, x2, y2)))
    return caption, out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def perceive(image_bytes: bytes, debug: bool = False) -> dict:
    """One frame -> a caption, the ego corridor, and the lane lines.

    Plus `qwen_boxes`: what the model said it saw, as a record beside the
    caption. Nothing here is drawn on the video -- see the module docstring.

    `debug` adds the raw model reply and token count to the response. Off by
    default: it is verbose and would bloat every session-log line.
    """
    t0 = time.time()
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = pil.size

    if not config.VISION_ENABLED:
        corridor = anchor_mod.EgoCorridor(width, height)
        return {
            "qwen_boxes": [], "corridor": [list(p) for p in corridor.polygon()],
            "caption": "", "observation": "",
            "image": {"w": width, "h": height},
            "timing_ms": {"total": 0.0},
        }

    raw = _query_qwen(_downscale(pil))
    t_qwen = time.time()
    caption, parsed = _parse(raw, width, height)

    # BGR uint8 for the lane detector: every headway consumer is OpenCV-native.
    frame_bgr = np.ascontiguousarray(np.asarray(pil)[:, :, ::-1])

    # The SAME corridor the headway loop uses this frame, not a second opinion.
    # These two paths pick a lead independently -- headway to warn on, perceive
    # to flag in the overlay -- and if they gate on different geometry they
    # disagree exactly where it matters most: on a bend, where the static
    # trapezoid projects straight off the road and the paint does not. The
    # driver would see the overlay bracket one car while the warning tracked
    # another. Lane detection costs ~2 ms against this endpoint's ~1 s Qwen
    # call, so there is no reason for the cheaper path to be the stale one.
    base_corridor = anchor_mod.EgoCorridor(width, height)
    lane_result = None
    try:
        lane_result = lanes_mod.detect_lanes(frame_bgr)
    except Exception as e:
        # Same contract as headway.live: no lanes means the trapezoid, not a
        # failed frame.
        print(f"[perceive] lane detection unavailable: {e}", flush=True)
    corridor, lane_info = anchor_mod.build_corridor(base_corridor, lane_result)

    # What Qwen said it saw, in frame pixels, and nothing derived from it.
    #
    # There is no distance here and no lead. Both used to exist, and both
    # existed only to be drawn: a depth median inside each box, a corridor
    # fallback when the depth was untrustworthy, and a plausibility veto over
    # the top of the two. That veto was doing real work -- it caught 6 m
    # claimed on a 10 px box -- but the thing it was protecting was a number
    # measured inside a rectangle that a language model had placed, and no
    # amount of checking makes such a number worth putting on a screen next to
    # a measured one. Range belongs to the path that has a detector and a
    # tracker under it.
    #
    # The boxes themselves stay because they cost nothing and they make the
    # caption reviewable: "four cars on the highway" is much easier to judge
    # against the four rectangles the model was looking at when it said it.
    qwen_boxes = [{"label": label,
                   "box": [round(float(v), 1) for v in box]}
                  for label, box in parsed]

    if caption:
        # Drive mode now calls /perceive instead of /observe, so this is the
        # only thing keeping RIO's cached view of the road current.
        vision.set_observation(caption)

    t_end = time.time()
    if debug:
        return {
            "qwen_boxes": qwen_boxes,
            "corridor": [[round(float(x), 1), round(float(y), 1)]
                         for x, y in corridor.polygon()],
            "corridor_source": corridor.source,
            "lane_conf": lane_info.get("lane_conf"),
            "lanes": [[[round(float(x), 1), round(float(y), 1)] for x, y in pts]
                      for pts in ((lane_result or {}).get("lanes") or [])],
            "lane_scores": list((lane_result or {}).get("lane_conf") or []),
            "lane_plausible": list((lane_result or {}).get("lane_plausible") or []),
            "caption": caption, "observation": caption,
            "image": {"w": width, "h": height},
            "raw": raw,
            "timing_ms": {
                "qwen": round((t_qwen - t0) * 1000, 1),
                "geometry": round((t_end - t_qwen) * 1000, 1),
                "total": round((t_end - t0) * 1000, 1),
            },
        }
    return {
        "qwen_boxes": qwen_boxes,
        "corridor": [[round(float(x), 1), round(float(y), 1)]
                     for x, y in corridor.polygon()],
        "corridor_source": corridor.source,
        "lane_conf": lane_info.get("lane_conf"),
        # Parallel to `lanes`: what each line is worth, and whether its shape
        # is a lane's. The overlay draws this endpoint's lanes on the same
        # canvas as headway's, so it has to be told the same things about them.
        "lanes": [[[round(float(x), 1), round(float(y), 1)] for x, y in pts]
                  for pts in ((lane_result or {}).get("lanes") or [])],
        "lane_scores": list((lane_result or {}).get("lane_conf") or []),
        "lane_plausible": list((lane_result or {}).get("lane_plausible") or []),
        "caption": caption,
        # Same text under the key the existing dashboard code already reads.
        "observation": caption,
        "image": {"w": width, "h": height},
        "timing_ms": {
            "qwen": round((t_qwen - t0) * 1000, 1),
            "geometry": round((t_end - t_qwen) * 1000, 1),
            "total": round((t_end - t0) * 1000, 1),
        },
    }


