"""Lead-vehicle anchor — Qwen3-VL proposes, deterministic geometry disposes.

Design ref: headway_design.md §5 (lead-vehicle selection), §1 (slow loop), §7.8.

The LLM firewall (§1) lives here: Qwen may *inform which object is measured*, and
nothing more. Its box is validated against a geometric ego corridor before the
tracker will accept it, and if the two disagree the corridor wins and we re-query.
No Qwen output ever reaches the state machine.

Stage 0 runs Qwen on the L40S. Note this loads a SECOND Qwen3-VL instance (~17 GB)
if the live RIO app is already serving one -- both fit on a 46 GB L40S, but pass
`use_qwen=False` and supply `--init-box` when you only need the deterministic path
(e.g. the synthetic clip test).
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

# Grounding prompt. Deliberately narrow: one box, ego lane only, strict JSON.
# Asking for prose here would mean parsing prose, and §1 forbids Qwen having any
# influence beyond "which pixels".
ANCHOR_PROMPT = (
    "Look at this forward-facing dashcam frame. Find the single vehicle directly "
    "ahead in MY lane (the lane I am driving in). Ignore vehicles in other lanes, "
    "oncoming traffic, and parked cars. Reply with ONLY this JSON and nothing else:\n"
    '{"bbox_2d": [x1, y1, x2, y2], "label": "<vehicle type>"}\n'
    'If there is no vehicle ahead in my lane, reply exactly: {"bbox_2d": null}'
)


class EgoCorridor:
    """Trapezoidal ground corridor projected through a pinhole camera (§5.1).

    Works in both directions: `ground_from_pixel` inverts the projection so a
    box-bottom-centre pixel becomes (forward distance, lateral offset), which is
    what the containment test actually needs.
    """

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
            "reason": "inside" if inside else "outside_corridor",
        }

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
# Qwen3-VL grounding
# ---------------------------------------------------------------------------
_qwen_model = None
_qwen_processor = None
_qwen_lock = threading.Lock()


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
                 corridor=None, max_new_tokens=128):
        self.corridor = corridor or EgoCorridor(width, height)
        self.interval = int(interval)
        self.use_qwen = bool(use_qwen)
        self.max_new_tokens = int(max_new_tokens)
        self.width = int(width)
        self.height = int(height)
        self.last_anchor_frame = None
        self.last_result = None

    def due(self, frame_idx: int) -> bool:
        if self.last_anchor_frame is None:
            return True
        return (frame_idx - self.last_anchor_frame) >= self.interval

    def anchor(self, frame, frame_idx: int, depth=None):
        """Propose + validate a lead box. Returns (box|None, info)."""
        self.last_anchor_frame = frame_idx
        if not self.use_qwen:
            self.last_result = (None, {"reason": "qwen_disabled"})
            return self.last_result

        raw = self._query_qwen(frame)
        boxes = _parse_boxes(raw, self.width, self.height)
        info = {"raw": (raw or "")[:300], "n_boxes": len(boxes)}

        if not boxes:
            info["reason"] = "no_box_parsed"
            self.last_result = (None, info)
            return self.last_result

        # §5.3-5.4: keep only candidates whose box-bottom-centre lands in the
        # corridor, then take the nearest. Geometry, never Qwen's ranking.
        validated = []
        for box in boxes:
            x1, y1, x2, y2 = box
            u, v = (x1 + x2) / 2.0, y2      # bottom-centre = where it meets the road
            inside, geo = self.corridor.contains(u, v)
            if inside:
                d = geo["forward_m"]
                # Prefer measured depth over geometric range when we have it:
                # geometry assumes a flat road and exact pitch, DA-V2 does not.
                if depth is not None:
                    from . import depth as depth_mod
                    d_meas, conf, _ = depth_mod.roi_depth(depth, box)
                    if np.isfinite(d_meas) and conf > 0.2:
                        d = d_meas
                validated.append((d, box, geo))

        if not validated:
            info["reason"] = "all_boxes_outside_corridor"
            info["rejected"] = [
                self.corridor.contains((b[0] + b[2]) / 2.0, b[3])[1] for b in boxes
            ]
            self.last_result = (None, info)
            return self.last_result

        validated.sort(key=lambda t: t[0])
        d, box, geo = validated[0]
        info.update({"reason": "ok", "range_m": round(float(d), 2), "geo": geo})
        self.last_result = (box, info)
        return self.last_result

    def _query_qwen(self, frame):
        import torch
        from PIL import Image

        _ensure_qwen()
        pil = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))  # BGR -> RGB
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": ANCHOR_PROMPT},
        ]}]
        with _qwen_lock:
            inputs = _qwen_processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(_qwen_model.device)
            with torch.inference_mode():
                out = _qwen_model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            return _qwen_processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
