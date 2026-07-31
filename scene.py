"""Scene graph — the picture RIO and the driver are both looking at.

Design ref: docs/visual_qa.md. This module is the seam between the perception
stack and the conversation: everything upstream of it measures the road,
everything downstream of it talks about the road, and this is where a tracked
box becomes something you can refer to in a sentence.

WHAT IT DOES NOT DO
-------------------
It does not detect, and it does not track. Both already happen, on every frame,
in headway/: RF-DETR proposes boxes (headway/detect.py), membership.CandidateSet
associates them into candidates with stable ids, Depth Anything attaches a range,
and the UFLDv2 corridor says where the lane is. A second detector living in the
conversation path would be a second opinion about what is on the road, and two
opinions is exactly the thing the headway design spent its whole history
removing.

So this module is arithmetic on state that already exists. What it adds is the
two things a sentence needs and a warning does not:

  POSITION.  membership.py asks "how much of this box is in MY lane", which is
             the right question for a following distance and useless for
             "the one on the left". Sidedness is recovered here from the same
             corridor, by measuring the object's offset from the lane centre in
             lane widths -- so "left_adjacent_lane" means one lane over from
             the paint we actually detected, not one third of the way across
             the image.

  MOTION.    A candidate holds three depth samples for a median. Whether it is
             closing, holding station or being driven past needs a couple of
             seconds, so the history lives here, keyed by track id, updated once
             per frame from the ring.

THE FIREWALL
------------
Nothing here consults a model, and nothing here decides anything a warning
depends on. The scene graph is read by the conversation path only. If this file
returned nonsense, RIO would describe the road badly and the safety system would
be entirely unaffected -- which is the property worth keeping.
"""
import io
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

import config

# --- position vocabulary ----------------------------------------------------
# Offsets are measured in LANE WIDTHS from the ego lane centre, at the row where
# the object meets the road. Half a lane width either side is still our lane;
# beyond one and a half is a lane we are not adjacent to.
EGO_HALF = 0.5
ADJACENT_HALF = 1.5

POS_EGO = "ego_lane"
POS_LEFT = "left_adjacent_lane"
POS_RIGHT = "right_adjacent_lane"
POS_FAR_LEFT = "two_or_more_lanes_left"
POS_FAR_RIGHT = "two_or_more_lanes_right"
POS_UNKNOWN = "unknown"

# Frame-relative fallback, used when the corridor has no bounds at that row --
# above the detected paint, or beyond the horizon. Named differently on purpose:
# "left_side_of_frame" is a weaker claim than "left_adjacent_lane" and the
# model is told which one it is getting.
POS_FRAME_LEFT = "left_side_of_frame"
POS_FRAME_AHEAD = "ahead_in_frame"
POS_FRAME_RIGHT = "right_side_of_frame"

# --- motion vocabulary ------------------------------------------------------
MOTION_CLOSING = "closing"
MOTION_RECEDING = "pulling_away"
MOTION_PARALLEL = "traveling_parallel"
MOTION_STATIONARY = "stationary"
MOTION_UNKNOWN = "unknown"

# A range changing slower than this is not moving relative to us. Generous,
# because ROI depth on a distant box is noisy and a jittery "closing/receding"
# flicker in the graph would read to the model as a car surging back and forth.
MOTION_DEADBAND_MS = 0.9
# ...and an object whose range shrinks at roughly our own road speed is not
# approaching, it is parked and we are driving past it.
STATIONARY_TOL_FRAC = 0.30
MOTION_WINDOW_S = 2.5
MOTION_MIN_SAMPLES = 3
MOTION_MIN_SPAN_S = 0.6

# --- track id namespace -----------------------------------------------------
# The spec's example is "vehicle_27". The numeric part is membership.py's own
# candidate id, unchanged, so a track id can always be traced back to the
# candidate that produced it.
_FAMILY = {
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
    "van": "vehicle", "motorcycle": "vehicle",
    "pedestrian": "person", "cyclist": "cyclist",
}


def track_id_for(label: Optional[str], cid: int) -> str:
    return f"{_FAMILY.get(label or '', 'object')}_{cid}"


def candidate_id_of(track_id: str):
    """"vehicle_27" -> 27. None if it is not one of ours."""
    try:
        return int(str(track_id).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def lane_offset(box, lane_bounds):
    """Offset of a box's road contact point from the lane centre, in lane widths.

    None when the corridor could not say where the lane was at that row, which
    is a different answer from zero and must not be rounded into one.
    """
    if not lane_bounds:
        return None
    xl, xr = float(lane_bounds[0]), float(lane_bounds[1])
    lane_w = xr - xl
    if lane_w <= 1.0:
        return None
    u = (float(box[0]) + float(box[2])) / 2.0
    return (u - (xl + xr) / 2.0) / lane_w


def position_of(box, lane_bounds, frame_w):
    """-> (position, source). source is "lane" or "frame"; see POS_FRAME_*."""
    off = lane_offset(box, lane_bounds)
    if off is None:
        u = (float(box[0]) + float(box[2])) / 2.0
        third = max(frame_w, 1) / 3.0
        if u < third:
            return POS_FRAME_LEFT, "frame"
        if u > 2 * third:
            return POS_FRAME_RIGHT, "frame"
        return POS_FRAME_AHEAD, "frame"
    a = abs(off)
    if a <= EGO_HALF:
        return POS_EGO, "lane"
    if a <= ADJACENT_HALF:
        return (POS_LEFT if off < 0 else POS_RIGHT), "lane"
    return (POS_FAR_LEFT if off < 0 else POS_FAR_RIGHT), "lane"


def side_of(position):
    """Coarse left / ahead / right, for matching what the driver said."""
    if position in (POS_LEFT, POS_FAR_LEFT, POS_FRAME_LEFT):
        return "left"
    if position in (POS_RIGHT, POS_FAR_RIGHT, POS_FRAME_RIGHT):
        return "right"
    if position in (POS_EGO, POS_FRAME_AHEAD):
        return "ahead"
    return "unknown"


def _slope(samples):
    """Least-squares d(value)/dt. Same reasoning as membership._slope: one noisy
    end sample must not be what decides a description."""
    n = len(samples)
    if n < 2:
        return None
    mt = sum(t for t, _ in samples) / n
    mv = sum(v for _, v in samples) / n
    den = sum((t - mt) ** 2 for t, _ in samples)
    if den < 1e-9:
        return None
    return sum((t - mt) * (v - mv) for t, v in samples) / den


def motion_of(samples, v_host):
    """Range history + our own speed -> a word for what the object is doing.

    `samples` is [(t, range_m)] over the last couple of seconds.
    """
    usable = [(t, r) for t, r in samples if r is not None and math.isfinite(r)]
    if len(usable) < MOTION_MIN_SAMPLES:
        return MOTION_UNKNOWN, None
    span = usable[-1][0] - usable[0][0]
    if span < MOTION_MIN_SPAN_S:
        return MOTION_UNKNOWN, None
    rate = _slope(usable)          # m/s; negative = the gap is shrinking
    if rate is None:
        return MOTION_UNKNOWN, None

    # Driving past something parked: the gap shrinks at our own road speed.
    if v_host is not None and math.isfinite(v_host) and v_host > 3.0:
        if abs(-rate - v_host) <= STATIONARY_TOL_FRAC * v_host:
            return MOTION_STATIONARY, rate
    if abs(rate) < MOTION_DEADBAND_MS:
        # Holding station relative to us. With no speed to compare against we
        # cannot tell "both moving together" from "both stopped", so the
        # weaker word is the honest one.
        if v_host is not None and math.isfinite(v_host) and v_host > 2.0:
            return MOTION_PARALLEL, rate
        return MOTION_UNKNOWN, rate
    return (MOTION_CLOSING if rate < 0 else MOTION_RECEDING), rate


# ---------------------------------------------------------------------------
# Frame quality
# ---------------------------------------------------------------------------
def sharpness(frame_bgr) -> float:
    """Variance of the Laplacian on a downscaled grey copy.

    The standard cheap blur metric. Downscaled first so a 1280x720 frame costs
    well under a millisecond, and because at full resolution the number is
    dominated by sensor noise rather than by whether the picture is sharp.
    """
    import cv2

    h, w = frame_bgr.shape[:2]
    if w > 480:
        s = 480.0 / w
        frame_bgr = cv2.resize(frame_bgr, (480, max(1, int(round(h * s)))),
                               interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


# Above this the frame is sharp enough that more sharpness is not better; the
# score saturates so a slightly crisper but much worse-framed shot cannot win.
SHARP_SATURATE = 300.0


def exposure_score(frame_bgr) -> float:
    """1.0 for a well-exposed frame, falling off as it clips black or white."""
    import cv2

    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clipped = float(((grey < 8) | (grey > 247)).mean())
    return float(max(0.0, 1.0 - clipped / 0.35))


@dataclass
class FrameQuality:
    """What is known about a frame's usability, cheaply and then fully.

    `sharp` and `exposure` are None until someone actually needs to choose
    between frames -- they cost a JPEG decode, and the 4 fps path must not pay
    for a question nobody asked.
    """
    glare_p01: Optional[float] = None
    depth_trusted: bool = True
    n_objects: int = 0
    mean_score: float = 0.0
    sharp: Optional[float] = None
    exposure: Optional[float] = None

    def measured(self) -> bool:
        return self.sharp is not None

    def base(self) -> float:
        """The part that is known without decoding: glare and detector health."""
        s = 1.0
        if not self.depth_trusted:
            s *= 0.5
        return s

    def score(self) -> float:
        sharp = 1.0 if self.sharp is None else min(1.0, self.sharp / SHARP_SATURATE)
        expo = 1.0 if self.exposure is None else self.exposure
        return self.base() * (0.6 * sharp + 0.4 * expo)

    def to_dict(self) -> dict:
        return {
            "glare_p01": self.glare_p01,
            "depth_trusted": self.depth_trusted,
            "n_objects": self.n_objects,
            "mean_score": round(self.mean_score, 3),
            "sharpness": None if self.sharp is None else round(self.sharp, 1),
            "exposure": None if self.exposure is None else round(self.exposure, 3),
            "score": round(self.score(), 4),
        }


def quality_from_result(result: dict) -> FrameQuality:
    """The cheap half, straight out of a /headway_frame result."""
    objs = result.get("scene_objects") or []
    scores = [o.get("score") or 0.0 for o in objs]
    return FrameQuality(
        glare_p01=result.get("glare_p01"),
        depth_trusted=bool(result.get("depth_trusted", True)),
        n_objects=len(objs),
        mean_score=(sum(scores) / len(scores)) if scores else 0.0,
    )


# ---------------------------------------------------------------------------
# Crops
# ---------------------------------------------------------------------------
def crop_jpeg(jpeg_bytes: bytes, box, pad_frac: float = None,
              min_px: int = None, max_px: int = None):
    """Cut one object out of a frame, with context. -> (jpeg_bytes, info).

    Context matters more than tightness here. A car cropped exactly at its own
    edges has lost the road under it, the lane beside it and the vehicles around
    it, and those are most of what makes a silhouette readable as a particular
    car rather than a generic one.

    `info` reports the TRUE source size in frame pixels and whether the crop was
    upscaled to reach min_px. That travels to the model, because a 38x24 patch
    stretched to 336 px looks like a photograph and is not one -- and a model
    that is not told will read a badge off the interpolation.
    """
    pad_frac = config.CROP_PAD_FRAC if pad_frac is None else pad_frac
    min_px = config.CROP_MIN_PX if min_px is None else min_px
    max_px = config.CROP_MAX_PX if max_px is None else max_px

    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    px, py = bw * pad_frac, bh * pad_frac

    cx1 = int(max(0, math.floor(x1 - px)))
    cy1 = int(max(0, math.floor(y1 - py)))
    cx2 = int(min(W, math.ceil(x2 + px)))
    cy2 = int(min(H, math.ceil(y2 + py)))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        cx1, cy1, cx2, cy2 = 0, 0, W, H

    patch = img.crop((cx1, cy1, cx2, cy2))
    src_w, src_h = patch.size
    longest = max(src_w, src_h)
    upscaled = False
    if longest < min_px:
        s = min_px / float(longest)
        patch = patch.resize((max(1, int(round(src_w * s))),
                              max(1, int(round(src_h * s)))), Image.LANCZOS)
        upscaled = True
    elif longest > max_px:
        s = max_px / float(longest)
        patch = patch.resize((max(1, int(round(src_w * s))),
                              max(1, int(round(src_h * s)))), Image.LANCZOS)

    buf = io.BytesIO()
    patch.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), {
        "crop_box": [cx1, cy1, cx2, cy2],
        "object_box": [round(v, 1) for v in (x1, y1, x2, y2)],
        "source_px": [src_w, src_h],
        "object_px": [round(bw), round(bh)],
        "output_px": list(patch.size),
        "upscaled": upscaled,
    }


def occlusion_of(box, others) -> float:
    """Largest fraction of this box covered by any other box. 0 = clear.

    Not a true occlusion test -- nothing here knows which vehicle is in front --
    but two boxes that overlap heavily produce a crop containing both, and that
    is the thing worth avoiding when choosing which frame to show the model.
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    worst = 0.0
    for o in others:
        ox1, oy1, ox2, oy2 = [float(v) for v in o]
        ix = max(0.0, min(x2, ox2) - max(x1, ox1))
        iy = max(0.0, min(y2, oy2) - max(y1, oy1))
        if ix > 0 and iy > 0:
            worst = max(worst, (ix * iy) / area)
    return worst


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------
@dataclass
class SceneObject:
    """One thing on the road, as something you can talk about."""
    track_id: str
    candidate_id: int
    label: str                       # RF-DETR's class. Authoritative.
    box: list
    position: str
    position_source: str
    depth_m: Optional[float]
    motion: str
    range_rate_ms: Optional[float]
    confidence: float
    is_lead: bool
    in_ego_lane: bool
    age_s: float
    lane_offset: Optional[float]
    attributes: dict = field(default_factory=dict)   # Qwen's, when asked for
    fine_label: Optional[str] = None                 # Qwen's, when asked for

    def to_dict(self) -> dict:
        """The shape the spec asks for, plus what the model needs to hedge.

        `label` stays the detector's coarse class and `fine_label` carries
        Qwen's guess, rather than one field that silently changes meaning. The
        spec's example writes "sports_coupe" into `label`; splitting them is the
        one place this deviates, and it is deliberate -- a detector class that a
        deterministic warning path also relies on must not be overwritten by a
        language model's opinion, and the model downstream needs to know which
        of the two it is reading.
        """
        d = {
            "track_id": self.track_id,
            "label": self.label,
            "bounding_box": [round(float(v)) for v in self.box],
            "position": self.position,
            "motion": self.motion,
            "depth_meters": self.depth_m,
            "attributes": dict(self.attributes),
            "confidence": round(float(self.confidence), 3),
        }
        if self.fine_label:
            d["fine_label"] = self.fine_label
        if self.is_lead:
            d["is_lead_vehicle"] = True
        if self.position_source != "lane":
            # Say so out loud. "left_side_of_frame" is a claim about the
            # picture; "left_adjacent_lane" is a claim about the road.
            d["position_basis"] = self.position_source
        return d


@dataclass
class SceneGraph:
    t: float
    wall_t: float
    frame_id: Optional[str]
    image: dict
    objects: list
    ego: dict

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "t": round(self.t, 3),
            "image": self.image,
            "ego": self.ego,
            "objects": [o.to_dict() for o in self.objects],
        }

    def by_track(self, track_id: str) -> Optional[SceneObject]:
        for o in self.objects:
            if o.track_id == track_id:
                return o
        return None


class SceneTracker:
    """Per-session history that a single frame cannot supply.

    Only motion, for now. It lives here rather than in membership.Candidate
    because it is needed to describe a scene and never to decide a warning, and
    the candidate's own three-sample depth deque exists for a median and would
    be the wrong window for a trend.
    """

    def __init__(self):
        self._range = {}         # candidate_id -> deque[(t, range_m)]
        self._first_seen = {}
        self.v_host = None

    def update(self, objects, t: float, v_host=None) -> None:
        if v_host is not None:
            self.v_host = v_host
        live = set()
        for o in objects:
            cid = o.get("id")
            if cid is None:
                continue
            live.add(cid)
            self._first_seen.setdefault(cid, t)
            hist = self._range.setdefault(cid, deque())
            hist.append((float(t), o.get("range_m")))
            while hist and (t - hist[0][0]) > MOTION_WINDOW_S:
                hist.popleft()
        # Candidates gone from the detector are dropped, so a re-used id from a
        # later drive cannot inherit an old trajectory. membership.py evicts at
        # 1 s undetected, which is well inside the motion window.
        for cid in [c for c in self._range if c not in live]:
            if t - self._range[cid][-1][0] > MOTION_WINDOW_S:
                self._range.pop(cid, None)
                self._first_seen.pop(cid, None)

    def motion(self, cid: int):
        return motion_of(list(self._range.get(cid, ())), self.v_host)

    def reset(self) -> None:
        self._range.clear()
        self._first_seen.clear()


def build(objects, ego: dict, image: dict, t: float, wall_t: float,
          frame_id: str = None, tracker: SceneTracker = None,
          attributes: dict = None) -> SceneGraph:
    """Raw per-frame perception -> the scene graph.

    `objects` is what headway.live puts in `scene_objects`; `attributes` maps
    track_id -> Qwen's enrichment, when any has been asked for.
    """
    attributes = attributes or {}
    out = []
    for o in objects:
        cid = o["id"]
        tid = track_id_for(o.get("label"), cid)
        pos, src = position_of(o["box"], o.get("lane_bounds"),
                               image.get("w") or 1)
        if tracker is not None:
            motion, rate = tracker.motion(cid)
        else:
            motion, rate = MOTION_UNKNOWN, None
        enr = attributes.get(tid) or {}
        obj = SceneObject(
            track_id=tid,
            candidate_id=cid,
            label=o.get("label") or "object",
            box=list(o["box"]),
            position=pos,
            position_source=src,
            depth_m=o.get("range_m"),
            motion=motion,
            range_rate_ms=(None if rate is None else round(float(rate), 2)),
            confidence=float(o.get("score") or 0.0),
            is_lead=bool(o.get("is_lead")),
            in_ego_lane=(pos == POS_EGO),
            age_s=float(o.get("age_s") or 0.0),
            lane_offset=(lambda v: None if v is None else round(v, 2))(
                lane_offset(o["box"], o.get("lane_bounds"))),
            attributes={k: v for k, v in (enr.get("attributes") or {}).items() if v},
            fine_label=enr.get("fine_label"),
        )
        out.append(obj)

    # Nearest first. The order is what a reader skims, and on a road the nearest
    # vehicle is almost always the one the sentence is about.
    out.sort(key=lambda o: (o.depth_m is None, o.depth_m if o.depth_m is not None else 0.0))
    return SceneGraph(t=t, wall_t=wall_t, frame_id=frame_id, image=dict(image),
                      objects=out, ego=dict(ego))


def ego_from_result(result: dict) -> dict:
    """The ego block: what the car itself is doing, as context for the answer.

    Deliberately excludes the warning band's voice decision. The safety system
    speaks for itself through the arbiter; a visual answer must never turn into
    a second, softer warning about the same gap.
    """
    return {
        "speed_ms": result.get("v_host"),
        "following_distance_m": result.get("distance_m"),
        "lane_source": result.get("corridor_source"),
        "lane_confidence": result.get("lane_conf"),
        "depth_trusted": bool(result.get("depth_trusted", True)),
    }
