"""Structured perception for the dashboard overlay: boxes + depth + corridor.

/observe answers "what do you see?" in one sentence. /perceive answers the same
question *and* returns where each thing is and how far away, so the dashboard can
draw it over the video. It is additive -- /observe is untouched and still serves
RIO's voice path.

Budget is one Qwen call plus one depth pass per frame, so the caption is produced
by the SAME grounding call that produces the boxes rather than a second
generate(). Everything geometric (corridor, lead choice) stays deterministic in
headway/, per the design's LLM firewall: Qwen says which pixels, never which
object matters.

Model sharing: this lends vision.py's already-resident Qwen3-VL to
headway.anchor through set_qwen_provider(), so a /perceive never pulls a second
copy of the weights.
"""
import io
import json
import re
import threading
import time

import numpy as np
from PIL import Image

import config
import vision
from headway import anchor as anchor_mod
from headway import depth as depth_mod
from headway import lanes as lanes_mod

# Lend the app's Qwen to headway. Import-time so any entry point into this
# module (endpoint, warm, self-test) is covered before the first anchor call.
anchor_mod.set_qwen_provider(vision.get_handles)

# Labels the overlay knows how to colour. Anything else Qwen invents is kept but
# drawn in the vehicle style, which is the safer default for a driving scene.
VEHICLE_LABELS = {"car", "truck", "bus", "van", "motorcycle", "train"}
VULNERABLE_LABELS = {"pedestrian", "person", "cyclist", "bicycle"}

# Only these can be the lead. anchor.py's corridor logic exists to pick the
# vehicle being followed; a pedestrian in the corridor is a hazard for the
# warning path, not a headway target, and calling one "lead" would be wrong.
LEAD_LABELS = VEHICLE_LABELS

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

_warm_lock = threading.Lock()
_warmed = False


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
    """One frame -> boxes with distances, ego corridor, and a caption.

    `debug` adds the raw model reply and token count to the response. Off by
    default: it is verbose and would bloat every session-log line.
    """
    t0 = time.time()
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = pil.size

    if not config.VISION_ENABLED:
        corridor = anchor_mod.EgoCorridor(width, height)
        return {
            "boxes": [], "corridor": [list(p) for p in corridor.polygon()],
            "caption": "", "observation": "",
            "image": {"w": width, "h": height},
            "timing_ms": {"total": 0.0},
        }

    raw = _query_qwen(_downscale(pil))
    t_qwen = time.time()
    caption, parsed = _parse(raw, width, height)

    # BGR uint8 for depth_map: every headway consumer is OpenCV-native.
    frame_bgr = np.ascontiguousarray(np.asarray(pil)[:, :, ::-1])
    depth_map = None
    try:
        depth_map = depth_mod.depth_map(frame_bgr)
    except Exception as e:
        # A depth failure costs distances, not the whole frame: boxes and the
        # caption are still worth returning to the overlay.
        print(f"[perceive] depth failed: {e}", flush=True)
    t_depth = time.time()

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

    boxes = []
    for label, box in parsed:
        x1, y1, x2, y2 = box
        distance_m = None
        conf = 0.0
        if depth_map is not None:
            d, conf, _ = depth_mod.roi_depth(depth_map, box)
            if np.isfinite(d) and conf > 0.2:
                distance_m = round(float(d), 1)

        # Bottom-centre is where the object meets the road -- the same anchor
        # point anchor.py validates against the corridor.
        u, v = (x1 + x2) / 2.0, y2
        inside, geo = corridor.contains(u, v)
        if distance_m is None and inside and "forward_m" in geo:
            # No trustworthy depth: fall back to the corridor's flat-road range
            # so the box still carries a distance the driver can read.
            distance_m = round(float(geo["forward_m"]), 1)

        boxes.append({
            "label": label,
            "box": [round(float(x1), 1), round(float(y1), 1),
                    round(float(x2), 1), round(float(y2), 1)],
            "distance_m": distance_m,
            "lead": False,
            "in_corridor": bool(inside),
            "depth_conf": round(float(conf), 3),
        })

    # Lead = nearest in-corridor vehicle, chosen by geometry exactly as
    # anchor.anchor() does. Qwen's ordering never decides this.
    lead_idx, lead_range = None, None
    for i, b in enumerate(boxes):
        if not b["in_corridor"] or b["label"] not in LEAD_LABELS:
            continue
        r = b["distance_m"]
        if r is None:
            continue
        if lead_range is None or r < lead_range:
            lead_idx, lead_range = i, r
    if lead_idx is not None:
        boxes[lead_idx]["lead"] = True

    if caption:
        # Drive mode now calls /perceive instead of /observe, so this is the
        # only thing keeping RIO's cached view of the road current.
        vision.set_observation(caption)

    t_end = time.time()
    if debug:
        return {
            "boxes": boxes,
            "corridor": [[round(float(x), 1), round(float(y), 1)]
                         for x, y in corridor.polygon()],
            "corridor_source": corridor.source,
            "lane_conf": lane_info.get("lane_conf"),
            "lanes": [[[round(float(x), 1), round(float(y), 1)] for x, y in pts]
                      for pts in ((lane_result or {}).get("lanes") or [])],
            "caption": caption, "observation": caption,
            "image": {"w": width, "h": height},
            "lead_range_m": lead_range,
            "raw": raw,
            "timing_ms": {
                "qwen": round((t_qwen - t0) * 1000, 1),
                "depth": round((t_depth - t_qwen) * 1000, 1),
                "total": round((t_end - t0) * 1000, 1),
            },
        }
    return {
        "boxes": boxes,
        "corridor": [[round(float(x), 1), round(float(y), 1)]
                     for x, y in corridor.polygon()],
        "corridor_source": corridor.source,
        "lane_conf": lane_info.get("lane_conf"),
        "lanes": [[[round(float(x), 1), round(float(y), 1)] for x, y in pts]
                  for pts in ((lane_result or {}).get("lanes") or [])],
        "caption": caption,
        # Same text under the key the existing dashboard code already reads.
        "observation": caption,
        "image": {"w": width, "h": height},
        "lead_range_m": lead_range,
        "timing_ms": {
            "qwen": round((t_qwen - t0) * 1000, 1),
            "depth": round((t_depth - t_qwen) * 1000, 1),
            "total": round((t_end - t0) * 1000, 1),
        },
    }


def warm() -> None:
    """Load the depth model and pay its kernel warmup.

    Qwen is already warmed by vision.warm(); this only covers the second model
    so the first real /perceive is not the one that waits for it.
    """
    global _warmed
    if not config.VISION_ENABLED:
        return
    with _warm_lock:
        if _warmed:
            return
        try:
            depth_mod.warm()
            _warmed = True
        except Exception as e:
            print(f"[perceive] depth warm failed: {e}", flush=True)
