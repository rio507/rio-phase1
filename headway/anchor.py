"""Lead-vehicle anchor — Qwen3-VL proposes, deterministic geometry disposes.

Design ref: headway_design.md §5 (lead-vehicle selection), §1 (slow loop), §7.8.

The LLM firewall (§1) lives here: Qwen may *inform which object is measured*, and
nothing more. Its box is validated against a geometric ego corridor before the
tracker will accept it, and if the two disagree the corridor wins and we re-query.
No Qwen output ever reaches the state machine.

Stage 0 runs Qwen on the L40S. By default this loads its OWN Qwen3-VL instance
(~17 GB); pass `use_qwen=False` with `--init-box` when you only need the
deterministic path (e.g. the synthetic clip test). A host that already has the
model resident can call `set_qwen_provider()` to lend it instead -- that is what
the live app's /perceive does, so the two never hold two copies.
"""
import json
import re
import threading

import numpy as np

# ---------------------------------------------------------------------------
# Ego-corridor constants (§5.1). Stage 0 assumes a straight, flat corridor with
# a rigidly mounted camera; Stage 2 replaces these with a real cone calibration.
# ---------------------------------------------------------------------------
LANE_WIDTH_M = 3.5          # standard lane; corridor half-width is half of this
CAMERA_HEIGHT_M = 1.3       # camera above road surface
CAMERA_PITCH_RAD = 0.0      # 0 = optical axis horizontal; +ve = pitched down
HFOV_DEG = 60.0             # horizontal field of view -> focal length in pixels

# A lead's box-bottom-centre rarely sits exactly on the lane centreline (tracker
# slop, non-flat road, box jitter), so the corridor is widened a little before
# rejecting a candidate. Too tight and every real lead gets thrown away.
CORRIDOR_MARGIN_M = 0.6

# Range gate for a plausible lead. Below 3 m nothing is measurable; past 80 m
# DA-V2 has flattened anyway (§7.2).
MIN_RANGE_M = 3.0
MAX_RANGE_M = 80.0

DEFAULT_ANCHOR_INTERVAL = 60  # frames between Qwen re-anchors (§5, ≤5 s per §7.4)

QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Only a vehicle can be the lead. A pedestrian or cyclist inside the corridor is
# a hazard for the warning path, not a following target -- calling one "lead"
# and computing a headway to it would be wrong. Same set as perceive.py's
# LEAD_LABELS, restated here because headway imports nothing from the app.
LEAD_LABELS = {"car", "truck", "bus", "van", "motorcycle"}

# Cap on vehicles requested. Latency here is DECODE-bound: ~52 tokens/s on this
# GPU, ~13 tokens per compact object, so every extra vehicle is a quarter of a
# second of frame time. Four is what the downstream actually consumes -- the
# corridor keeps the nearest in-lane one, and merge detection watches the lanes
# either side. A fifth car three hundred metres up the road costs 250 ms and
# changes no decision.
MAX_ANCHOR_OBJECTS = 4

# Decode budget for the anchor reply, and the single most expensive number in
# the fast loop. Measured: 192 tokens is 3.4 SECONDS of decode, and the model
# will happily fill whatever budget it is given on a busy motorway -- a bench
# run with 192 put p95 frame time at 3.4 s against a 250 ms cadence.
#
# It is set BELOW the length a full reply would need, on purpose. The prompt
# asks for closest-first ordering and repair_truncated_json() keeps whatever
# arrived, so truncation drops the FAR tail -- exactly the vehicles that change
# no decision. Spending a second of frame time to hear about them would be the
# error; losing them is not.
ANCHOR_MAX_NEW_TOKENS = 80

# Grounding prompt. ENUMERATE, DO NOT SELECT.
#
# This asks for every vehicle in view and explicitly tells the model NOT to
# decide which one we are following. That instruction is the whole point. The
# previous prompt asked for "the single vehicle directly ahead in MY lane",
# which handed Qwen the lane judgement and left the corridor able only to VETO
# its pick -- never to substitute the right one, because the right one was
# never in the candidate set. If Qwen named an adjacent-lane car and omitted
# the true lead, geometry rejected the answer and we anchored nothing.
#
# That fails safe (silence, not a warning on the wrong car) but it is a silent
# miss, and it put Qwen's lane judgement on the critical path, which is exactly
# what §1's firewall exists to prevent. Now the model reports pixels and labels;
# which of them is the lead is decided downstream by the corridor alone -- and
# since UFLDv2 that corridor is measured lane paint, not a guessed trapezoid.
ANCHOR_PROMPT = (
    # Scoped to what the geometry downstream can actually use: our lane (lead
    # candidates) and the lanes either side (merge candidates). This is a
    # SCOPING instruction, not a selection one -- the model is still forbidden
    # from saying which vehicle we are following, and the corridor still
    # decides membership. Enumerating the whole horizon instead cost seconds of
    # decode for vehicles no rule reads.
    "Look at this forward-facing dashcam frame. List the vehicles AHEAD of me "
    "travelling my way, in my lane or the lanes immediately left or right.\n"
    "Labels allowed: car, truck, bus, van, motorcycle.\n"
    "Each vehicle is [label, x1, y1, x2, y2] using integer coordinates.\n"
    # Same failure perceive.py hit: without this the model emits the label once
    # and sends bare coordinate arrays after it, losing the class of every
    # vehicle but the first.
    "EVERY vehicle must begin with its label in quotes, even when the label "
    "repeats.\n"
    f"List at most {MAX_ANCHOR_OBJECTS} vehicles, closest (largest) first.\n"
    "Do NOT decide which vehicle I am following. Just list what you see.\n"
    "Reply with ONLY this JSON and nothing else:\n"
    '{"o": [["car", x1, y1, x2, y2], ["truck", x1, y1, x2, y2]]}\n'
    'If there are no vehicles, reply exactly: {"o": []}'
)


class EgoCorridor:
    """Trapezoidal ground corridor projected through a pinhole camera (§5.1).

    Works in both directions: `ground_from_pixel` inverts the projection so a
    box-bottom-centre pixel becomes (forward distance, lateral offset), which is
    what the containment test actually needs.

    Since UFLDv2 landed this is the FALLBACK corridor, not the only one — see
    LaneCorridor and build_corridor() below. It is still what runs whenever the
    paint cannot be read, so it has not been weakened or retuned.
    """

    source = "static"

    def __init__(self, width_px, height_px, lane_width_m=LANE_WIDTH_M,
                 camera_height_m=CAMERA_HEIGHT_M, pitch_rad=CAMERA_PITCH_RAD,
                 hfov_deg=HFOV_DEG, margin_m=CORRIDOR_MARGIN_M):
        self.w = float(width_px)
        self.h = float(height_px)
        self.lane_width_m = float(lane_width_m)
        self.camera_height_m = float(camera_height_m)
        self.pitch_rad = float(pitch_rad)
        self.margin_m = float(margin_m)

        # Pinhole focal length from horizontal FOV. This is the one number that
        # most affects corridor accuracy -- calibrate it in Stage 2.
        self.f_px = (self.w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0

    def ground_from_pixel(self, u: float, v: float):
        """Pixel -> (forward_m, lateral_m) on the road plane. None if above horizon.

        Camera frame: x right, y down, z forward, pitched down by `pitch_rad`.
        A ground point sits at world (X, h, L). Rotating into camera coords and
        projecting gives v = cy + f*y_c/z_c, which inverts to:

            L = h(cosθ - k sinθ) / (k cosθ + sinθ),   k = (v - cy)/f

        At θ=0 this collapses to the familiar L = f·h/(v - cy).
        """
        k = (float(v) - self.cy) / self.f_px
        ct, st = np.cos(self.pitch_rad), np.sin(self.pitch_rad)

        denom = k * ct + st
        if denom <= 1e-9:
            return None  # at or above the horizon -- no ground intersection

        forward = self.camera_height_m * (ct - k * st) / denom
        if not np.isfinite(forward) or forward <= 0:
            return None

        z_c = self.camera_height_m * st + forward * ct
        lateral = (float(u) - self.cx) * z_c / self.f_px
        return float(forward), float(lateral)

    def half_width_at(self, forward_m: float) -> float:
        return self.lane_width_m / 2.0 + self.margin_m

    def contains(self, u: float, v: float, yaw_rate=None, v_host=None):
        """Is this pixel's ground point inside the ego corridor? -> (bool, info)."""
        g = self.ground_from_pixel(u, v)
        if g is None:
            return False, {"reason": "above_horizon"}

        forward, lateral = g
        if not (MIN_RANGE_M <= forward <= MAX_RANGE_M):
            return False, {"reason": "out_of_range", "forward_m": forward, "lateral_m": lateral}

        # §5.2: bend the corridor into the turn when yaw rate is available. Stage 0
        # has no IMU, so this is normally skipped -- the known curve weakness (§7.3).
        bend = 0.0
        if yaw_rate is not None and v_host is not None and abs(v_host) > 1.0:
            bend = (yaw_rate / v_host) * forward ** 2 / 2.0

        offset = lateral - bend
        half = self.half_width_at(forward)
        inside = abs(offset) <= half
        return inside, {
            "forward_m": round(forward, 2),
            "lateral_m": round(lateral, 2),
            "bend_m": round(bend, 3),
            "offset_m": round(offset, 2),
            "half_width_m": round(half, 2),
            "corridor_source": "static",
            "reason": "inside" if inside else "outside_corridor",
        }

    def bounds_at_row(self, v):
        """(x_left, x_right) of the corridor at image row v, or None.

        Same signature LaneCorridor exposes, so membership.py has ONE code path
        for "how much of this vehicle is in my lane" whether the corridor came
        from paint or from the trapezoid. The trapezoid's answer is worse -- it
        is a projection of assumed geometry -- but it is the same shape of
        answer, and an overlap test against it still beats the centre-point
        test it replaces.

        Returned WITHOUT the corridor margin: `contains()` owns the slack, and
        membership expresses its own tolerance as the 40%/25% thresholds
        instead. Adding both would double-count it.
        """
        g = self.ground_from_pixel(u=self.cx, v=v)
        if g is None:
            return None
        forward, _ = g
        ct, st = np.cos(self.pitch_rad), np.sin(self.pitch_rad)
        z_c = self.camera_height_m * st + forward * ct
        if z_c <= 1e-6:
            return None
        half_px = self.f_px * (self.lane_width_m / 2.0) / z_c
        return (float(self.cx - half_px), float(self.cx + half_px))

    def polygon(self, near_m: float = 5.0, far_m: float = 60.0, steps: int = 12):
        """Corridor outline in pixel coords, for the debug overlay."""
        left, right = [], []
        ct, st = np.cos(self.pitch_rad), np.sin(self.pitch_rad)
        for i in range(steps + 1):
            L = near_m + (far_m - near_m) * i / steps
            half = self.half_width_at(L)
            z_c = self.camera_height_m * st + L * ct
            y_c = self.camera_height_m * ct - L * st
            if z_c <= 1e-6:
                continue
            v = self.cy + self.f_px * y_c / z_c
            for sign, acc in ((-1.0, left), (1.0, right)):
                u = self.cx + self.f_px * (sign * half) / z_c
                acc.append((float(u), float(v)))
        return left + right[::-1]


# ---------------------------------------------------------------------------
# Lane-derived corridor (UFLDv2)
# ---------------------------------------------------------------------------
# The corridor margin as a fraction of lane width, rather than a metre count
# projected through a guessed focal length. CORRIDOR_MARGIN_M of LANE_WIDTH_M is
# the same slack the trapezoid allows -- expressed against the lane the camera
# can actually see, it needs no calibration at all, and it stays correct on a
# narrow urban lane where 0.6 m is proportionally much more forgiving.
LANE_MARGIN_FRAC = CORRIDOR_MARGIN_M / LANE_WIDTH_M

# How far below its lowest detected point a boundary may be extended before the
# containment test stops trusting it and hands the row back to the trapezoid.
LANE_EXTRAP_FRAC = 0.20


class LaneCorridor:
    """Ego corridor bounded by two detected lane boundaries (UFLDv2).

    Same contract as EgoCorridor -- `contains`, `polygon`, `ground_from_pixel`
    -- so nothing downstream needs to know which one it holds.

    The division of labour matters. Lane paint answers "is this vehicle between
    my lines?", which is the question the corridor exists to ask, and answers it
    without assuming a straight road, a level camera or a 3.5 m lane. It cannot
    answer "how far away is it" -- so range still comes from the pinhole
    projection in `base` (and, further downstream, from Depth Anything). The
    curve weakness the trapezoid has on bends (§7.3) is exactly the part that
    lane paint fixes; the range estimate is unchanged and no better than it was.
    """

    source = "ufld"

    def __init__(self, base: EgoCorridor, ego_left, ego_right, confidence,
                 margin_frac: float = LANE_MARGIN_FRAC):
        self.base = base
        self.w, self.h = base.w, base.h
        self.cx, self.cy, self.f_px = base.cx, base.cy, base.f_px
        self.camera_height_m = base.camera_height_m
        self.pitch_rad = base.pitch_rad
        self.left = list(ego_left)
        self.right = list(ego_right)
        self.confidence = float(confidence)
        self.margin_frac = float(margin_frac)
        self._max_extrap = base.h * LANE_EXTRAP_FRAC

    # -- delegated projection ------------------------------------------------
    def ground_from_pixel(self, u, v):
        return self.base.ground_from_pixel(u, v)

    def half_width_at(self, forward_m):
        return self.base.half_width_at(forward_m)

    def bounds_at_row(self, v):
        """(x_left, x_right) of the ego lane at image row v, or None.

        The real thing: measured paint, no margin. membership.py's overlap test
        reads this, and EgoCorridor exposes the same call so the fallback
        corridor answers the same question in the same units.
        """
        from .lanes import x_at_y
        a = x_at_y(self.left, v, self._max_extrap)
        b = x_at_y(self.right, v, self._max_extrap)
        if a is None or b is None:
            return None
        xl, xr = a[0], b[0]
        return (xl, xr) if xr > xl else None

    # -- containment ---------------------------------------------------------
    def contains(self, u, v, yaw_rate=None, v_host=None):
        """Is this pixel between the detected lane lines? -> (bool, info).

        The range gate is unchanged -- a vehicle 200 m away is still not a lead
        even if it is dead centre between the lines. Only the lateral test moves
        from projected metres to measured paint.
        """
        g = self.ground_from_pixel(u, v)
        if g is None:
            return False, {"reason": "above_horizon", "corridor_source": "ufld"}
        forward, lateral = g
        if not (MIN_RANGE_M <= forward <= MAX_RANGE_M):
            return False, {"reason": "out_of_range", "forward_m": round(forward, 2),
                           "lateral_m": round(lateral, 2), "corridor_source": "ufld"}

        bounds = self.bounds_at_row(v)
        if bounds is None:
            # The paint does not reach this row (a lead near the horizon, above
            # where the net declared the lane). Rather than extrapolate a
            # boundary a hundred rows past its last evidence, this one pixel
            # falls back to the trapezoid. Logged, so it is visible in review.
            inside, info = self.base.contains(u, v, yaw_rate, v_host)
            info["corridor_source"] = "ufld_row_fallback"
            return inside, info

        xl, xr = bounds
        lane_w = xr - xl
        margin = self.margin_frac * lane_w
        centre = (xl + xr) / 2.0
        # Signed position across the lane: 0 at the centreline, +-1 at a
        # boundary. The margin is what allows a shade past 1.0.
        offset_norm = (float(u) - centre) / (lane_w / 2.0)
        inside = (xl - margin) <= float(u) <= (xr + margin)

        return inside, {
            "forward_m": round(forward, 2),
            "lateral_m": round(offset_norm * LANE_WIDTH_M / 2.0, 2),
            "offset_norm": round(offset_norm, 3),
            "lane_px": [round(xl, 1), round(xr, 1)],
            "margin_px": round(margin, 1),
            "corridor_source": "ufld",
            "reason": "inside" if inside else "outside_lane",
        }

    def polygon(self, near_m: float = 5.0, far_m: float = 60.0, steps: int = 12):
        """The lane polygon itself — left boundary out, right boundary back.

        Point order matches EgoCorridor.polygon() (near->far down one side, far
        ->near back the other) so the overlay's existing fill/stroke needs no
        change. near_m/far_m/steps are accepted and ignored: the extent of this
        polygon is however far the paint was actually visible, which is the
        honest thing to draw.
        """
        return ([(float(x), float(y)) for x, y in reversed(self.left)]
                + [(float(x), float(y)) for x, y in self.right])


def build_corridor(base: EgoCorridor, lane_result, conf_min=None):
    """Pick the corridor for this frame. -> (corridor, info).

    THE FALLBACK IS THE POINT. UFLDv2 is confident on daylight highway paint and
    much less so at night, in rain, on worn or snow-covered markings, and on
    unmarked roads -- the conditions where a wrong corridor would do the most
    damage. Below the confidence floor the static trapezoid takes over, which is
    exactly the behaviour this system had before lanes existed. There is no
    third mode where a low-confidence lane is used anyway.
    """
    from . import lanes as lanes_mod
    floor = lanes_mod.LANE_CONF_MIN if conf_min is None else float(conf_min)

    if not lane_result:
        return base, {"corridor_source": "static", "lane_conf": 0.0,
                      "fallback_reason": "no_lane_result"}

    conf = float(lane_result.get("confidence") or 0.0)
    left, right = lane_result.get("ego_left"), lane_result.get("ego_right")

    if left is None or right is None:
        return base, {"corridor_source": "static", "lane_conf": conf,
                      "fallback_reason": (lane_result.get("ego") or {}).get(
                          "reason", "no_ego_pair")}
    if conf < floor:
        return base, {"corridor_source": "static", "lane_conf": round(conf, 3),
                      "fallback_reason": "low_confidence",
                      "conf_floor": floor}

    return LaneCorridor(base, left, right, conf), {
        "corridor_source": "ufld",
        "lane_conf": round(conf, 3),
        "conf_floor": floor,
        "lane_width_frac": round(
            float((lane_result.get("ego") or {}).get("lane_width_frac") or 0.0), 3),
        "extrapolated": bool((lane_result.get("ego") or {}).get("extrapolated")),
    }


# ---------------------------------------------------------------------------
# Qwen3-VL grounding
# ---------------------------------------------------------------------------
_qwen_model = None
_qwen_processor = None
_qwen_lock = threading.Lock()

# Optional injection point for an already-loaded Qwen3-VL. Set by the live app
# (see perceive.py) so /perceive grounds boxes on the instance vision.py already
# holds rather than loading a second ~17 GB copy -- the VRAM fits, but a second
# 35 s load on a running service does not. Left None, this module keeps its
# standalone behaviour and loads its own weights, so the Stage 0 clip tools are
# unaffected and headway still imports nothing from the app.
_qwen_provider = None


def set_qwen_provider(provider) -> None:
    """Supply a zero-arg callable returning (processor, model, lock)."""
    global _qwen_provider
    _qwen_provider = provider


def has_qwen_provider() -> bool:
    """True once someone has lent this module a resident Qwen3-VL.

    Lets a caller check whether grounding would load a second copy of the
    weights *before* it triggers one — headway.live uses it to install the
    app's provider on any entry path that reaches it before perceive.py has.
    """
    return _qwen_provider is not None


def _ensure_qwen():
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    _qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    _qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )


def qwen_handles():
    """(processor, model, lock) — the injected instance if there is one.

    The lock travels with the handles on purpose: when the app injects its
    model, /observe and /perceive are two threads holding one set of weights,
    and they must serialise on the *same* lock rather than each on their own.
    """
    if _qwen_provider is not None:
        return _qwen_provider()
    _ensure_qwen()
    return _qwen_processor, _qwen_model, _qwen_lock


def strip_fence(text: str) -> str:
    """Drop a ```-fenced wrapper the model sometimes adds around its JSON."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def repair_truncated_json(text: str):
    """Recover a reply the token cap cut off mid-array. -> dict or None.

    Truncation is NORMAL here, not exceptional: the cap is set to a latency
    budget, not to the longest possible reply, so a busy frame routinely ends
    part-way through an object. Dropping the partial tail and closing the
    brackets keeps every object that did arrive.

    Getting this wrong is not a lost frame, it is a WRONG frame. Without it a
    truncated reply falls through to the bare-coordinate regex, which cannot
    see labels at all -- so every recovered box arrives unlabelled and a road
    sign is as eligible to be the lead as a lorry. Observed on real footage
    before this existed.
    """
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]
    cut = body.rfind("]")
    if cut < 0:
        return None
    stem = body[: cut + 1]
    for suffix in ("]}", "}", "]]}"):
        try:
            obj = json.loads(stem + suffix)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _parse_vehicles(text: str, width: int, height: int):
    """Qwen's reply -> [(label, (x1,y1,x2,y2)), ...] in frame pixels.

    Accepts the enumerated form the prompt asks for, and still accepts the
    single-`bbox_2d` shape the old prompt produced, so a checkpoint that
    answers the old way degrades to one candidate instead of to none.

    Coordinate rescaling goes through `_rescale`, which is the one place that
    knows this checkpoint emits 0-1000 normalised coordinates. perceive.py has
    its own JSON reader for a different payload (captioned, longer, needs
    truncation repair) but delegates rescaling here for the same reason: the
    shape of the JSON is cosmetic, the coordinate convention is not.
    """
    if not text:
        return []

    t = strip_fence(text)
    obj = None
    # Greedy span first: the whole reply should be one JSON object, and a lazy
    # match stops at the first nested brace inside the vehicle list.
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
        obj = repair_truncated_json(t)
    if obj is None:
        return [(None, b) for b in _parse_boxes(text, width, height)]

    def _nums(seq):
        if not all(isinstance(n, (int, float)) and not isinstance(n, bool)
                   for n in seq):
            return None
        return [float(n) for n in seq]

    raw = obj.get("o")
    if not isinstance(raw, list):
        raw = obj.get("objects")
    if not isinstance(raw, list):
        # Old single-box shape, or something else entirely -- let the legacy
        # reader have a go rather than returning nothing.
        return [(None, b) for b in _parse_boxes(text, width, height)]

    out = []
    for entry in raw:
        label, nums = None, None
        if isinstance(entry, (list, tuple)):
            if len(entry) == 5 and isinstance(entry[0], str):
                label, nums = entry[0].strip().lower(), _nums(entry[1:])
            elif len(entry) == 4:
                nums = _nums(entry)
        elif isinstance(entry, dict):
            for key in ("bbox_2d", "bbox", "box"):
                if isinstance(entry.get(key), (list, tuple)) and len(entry[key]) == 4:
                    nums = _nums(entry[key])
                    break
            lab = entry.get("label")
            if isinstance(lab, str):
                label = lab.strip().lower()
        if not nums:
            continue
        x1, y1, x2, y2 = _rescale(nums, width, height)
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            out.append((label, (x1, y1, x2, y2)))

    # These models emit the label once and then send bare coordinate arrays for
    # the rest -- the exact failure the prompt's "EVERY vehicle must begin with
    # its label" line fights, and does not always win. An unlabelled entry
    # alongside labelled ones is therefore of UNKNOWN class, and an unknown
    # object must not be eligible to become the lead: that is how a gantry sign
    # or a bridge parapet ends up with a headway computed to it. They are
    # dropped, not guessed at, and not inherited from the previous entry.
    #
    # When NOTHING carried a label the reply is the old single-box shape (or the
    # bare-coordinate fallback), where unlabelled is the only thing on offer --
    # so those are kept, and the caller still range-gates them.
    if any(lab for lab, _ in out):
        out = [(lab, b) for lab, b in out if lab]
    return out


def _parse_boxes(text: str, width: int, height: int):
    """Pull [x1,y1,x2,y2] boxes out of Qwen's reply, whatever dialect it used.

    Qwen variants emit absolute pixels, 0-1 normalised, or 0-1000 normalised
    depending on version and phrasing, so the scale is inferred from magnitude
    rather than assumed.
    """
    if not text:
        return []

    candidates = []

    # Preferred path: the JSON we asked for (possibly fenced or with prose around it).
    for blob in re.findall(r"\{.*?\}", text, re.S):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        for key in ("bbox_2d", "bbox", "box", "boxes"):
            if key in obj and obj[key]:
                val = obj[key]
                if isinstance(val, (list, tuple)) and len(val) == 4 and all(
                    isinstance(n, (int, float)) for n in val
                ):
                    candidates.append([float(n) for n in val])
                elif isinstance(val, (list, tuple)):
                    for sub in val:
                        if isinstance(sub, (list, tuple)) and len(sub) == 4:
                            candidates.append([float(n) for n in sub])

    # Fallback: a bare 4-number list anywhere in the text.
    if not candidates:
        for m in re.findall(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*"
                            r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", text):
            candidates.append([float(x) for x in m])

    out = []
    for box in candidates:
        x1, y1, x2, y2 = _rescale(box, width, height)
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            out.append((x1, y1, x2, y2))
    return out


def _rescale(box, width: int, height: int, coord_space: str = "auto"):
    """Map Qwen's coordinates into frame pixels.

    Qwen3-VL-8B-Instruct emits 0-1000 NORMALISED coordinates, not absolute
    pixels. Verified against this checkpoint: an 810x1080 frame came back as
    [0, 213, 999, 677] -- x2=999 exceeds the 810 px width, so it cannot be
    pixels. The earlier guard here only applied the /1000 rescale when the frame
    was itself larger than 1000 px, which would silently mis-scale every box on
    a 640x480 or 960x540 feed. Normalised is now the default reading, and only
    a coordinate that overflows the 0-1000 range is treated as pixels.
    """
    x1, y1, x2, y2 = box
    mx = max(abs(v) for v in box)

    space = coord_space
    if space == "auto":
        if mx <= 1.5:
            space = "unit"          # 0-1
        elif mx <= 1000.0:
            space = "norm1000"      # Qwen3-VL's convention
        else:
            space = "pixels"        # overflows 0-1000, so it must be absolute

    if space == "unit":
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif space == "norm1000":
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))


class LeadAnchor:
    """Re-anchors the tracker on the lead vehicle every `interval` frames (§5)."""

    def __init__(self, width, height, interval=DEFAULT_ANCHOR_INTERVAL, use_qwen=True,
                 corridor=None, max_new_tokens=ANCHOR_MAX_NEW_TOKENS):
        self.corridor = corridor or EgoCorridor(width, height)
        self.interval = int(interval)
        self.use_qwen = bool(use_qwen)
        self.max_new_tokens = int(max_new_tokens)
        self.width = int(width)
        self.height = int(height)
        self.last_anchor_frame = None
        self.last_result = None
        # Every vehicle the last enumeration reported, label included, BEFORE
        # the corridor filtered it down to one. membership.py needs the ones
        # that were rejected as much as the one that was kept -- an adjacent
        # car is exactly what a merge starts as.
        self.last_detections = []

    def due(self, frame_idx: int) -> bool:
        if self.last_anchor_frame is None:
            return True
        return (frame_idx - self.last_anchor_frame) >= self.interval

    def anchor(self, frame, frame_idx: int, depth=None):
        """Propose + validate a lead box. Returns (box|None, info)."""
        self.last_anchor_frame = frame_idx
        self.last_detections = []
        if not self.use_qwen:
            self.last_result = (None, {"reason": "qwen_disabled"})
            return self.last_result

        raw = self._query_qwen(frame)
        detected = _parse_vehicles(raw, self.width, self.height)
        self.last_detections = list(detected)
        info = {"raw": (raw or "")[:300], "n_detected": len(detected),
                "corridor_source": getattr(self.corridor, "source", "static")}

        if not detected:
            info["reason"] = "no_box_parsed"
            self.last_result = (None, info)
            return self.last_result

        # A label the prompt did not offer means the model volunteered a
        # non-vehicle (it does occasionally name a sign or a traffic light).
        # `None` is kept: it is the legacy single-box shape or an unlabelled
        # entry, and dropping those would silently disable the fallback path.
        candidates = [(lab, b) for lab, b in detected
                      if lab is None or lab in LEAD_LABELS]
        info["n_vehicles"] = len(candidates)
        if not candidates:
            info["reason"] = "no_vehicle_labels"
            info["labels"] = sorted({lab for lab, _ in detected if lab})
            self.last_result = (None, info)
            return self.last_result

        # §5.3-5.4: keep only candidates whose box-bottom-centre lands in the
        # corridor, then take the nearest. THE CORRIDOR DOES THE SELECTING --
        # the prompt above deliberately declines to, so this is the only place
        # "which vehicle am I following" is decided, and it is decided by
        # geometry (measured lane paint when UFLDv2 is confident).
        validated, rejected = [], []
        for label, box in candidates:
            x1, y1, x2, y2 = box
            u, v = (x1 + x2) / 2.0, y2      # bottom-centre = where it meets the road
            inside, geo = self.corridor.contains(u, v)
            if not inside:
                rejected.append({"label": label, **geo})
                continue
            d = geo["forward_m"]
            # Prefer measured depth over geometric range when we have it:
            # geometry assumes a flat road and exact pitch, DA-V2 does not.
            if depth is not None:
                from . import depth as depth_mod
                d_meas, conf, _ = depth_mod.roi_depth(depth, box)
                if np.isfinite(d_meas) and conf > 0.2:
                    d = d_meas
            validated.append((d, box, geo, label))

        if not validated:
            info["reason"] = "all_boxes_outside_corridor"
            info["n_rejected"] = len(rejected)
            info["rejected"] = rejected
            self.last_result = (None, info)
            return self.last_result

        validated.sort(key=lambda t: t[0])
        d, box, geo, label = validated[0]
        info.update({"reason": "ok", "range_m": round(float(d), 2), "geo": geo,
                     "label": label, "n_in_corridor": len(validated),
                     "n_rejected": len(rejected)})
        self.last_result = (box, info)
        return self.last_result

    def _query_qwen(self, frame):
        import torch
        from PIL import Image

        processor, model, lock = qwen_handles()
        pil = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))  # BGR -> RGB
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": ANCHOR_PROMPT},
        ]}]
        with lock:
            inputs = processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            return processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
