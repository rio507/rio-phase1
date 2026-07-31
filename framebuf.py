"""Frame ring buffer — the last few seconds of road, kept in RAM.

Design ref: docs/visual_qa.md §3.

WHY KEEP FRAMES AT ALL
----------------------
Because the newest frame is usually the wrong one. A driver asks about a car a
beat after noticing it, and by the time the question has been spoken and
transcribed the vehicle has moved: smaller, further off-centre, half out of
shot, or behind the pillar of a sign. The frame where it was biggest, sharpest
and unoccluded went past two seconds ago. Six seconds of history is what makes
"the clearest recent frame containing it" a thing you can actually go and find,
and it is the whole reason frame selection is a scoring problem rather than an
array index.

WHAT IS KEPT, AND WHAT IS NOT
-----------------------------
Each entry holds the JPEG exactly as the client sent it -- not re-encoded, so
the picture the model sees is the picture the detector measured -- plus the
per-object perception for that frame and the cheap half of a quality score.
Sharpness and exposure are NOT computed on the way in: they cost a decode, the
4 fps path must not pay for a question nobody asked, and they are only ever
needed for the handful of frames on a shortlist. They are filled in lazily and
cached on the entry.

RETENTION
---------
RAM only, and it dies with the session. Nothing here writes a picture of the
road to disk unless config.RING_PERSIST is turned on, which is an operator's
decision and not a default. That is the whole of the privacy posture at this
layer: the buffer is six seconds long, it is not a recording, and when the
drive ends it is gone.
"""
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config
import scene as scene_mod

PERSIST_DIR = Path("/workspace/rio-phase1/training_data/visual")


@dataclass
class RingFrame:
    """One retained frame and everything known about it."""
    frame_id: str
    t: float                      # session clock, seconds (headway's own clock)
    wall_t: float
    jpeg: bytes
    w: int
    h: int
    objects: list                 # headway.live's `scene_objects` for this frame
    ego: dict
    quality: "scene_mod.FrameQuality"
    _decoded: object = field(default=None, repr=False)

    @property
    def age_s(self) -> float:
        return time.time() - self.wall_t

    def frame_bgr(self):
        """Decode once, keep it. Only frames on a shortlist are ever decoded."""
        if self._decoded is None:
            import cv2
            import numpy as np
            self._decoded = cv2.imdecode(
                np.frombuffer(self.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        return self._decoded

    def measure(self) -> "scene_mod.FrameQuality":
        """Fill in the half of the quality score that needs the pixels."""
        if self.quality.measured():
            return self.quality
        try:
            frame = self.frame_bgr()
            if frame is not None:
                self.quality.sharp = scene_mod.sharpness(frame)
                self.quality.exposure = scene_mod.exposure_score(frame)
        except Exception:
            # A frame we cannot decode simply keeps its unmeasured score, which
            # is optimistic -- but a frame that fails to decode will also fail
            # to crop, and the selector's caller handles that.
            pass
        return self.quality

    def object_by_id(self, cid: int):
        for o in self.objects:
            if o.get("id") == cid:
                return o
        return None

    def to_meta(self) -> dict:
        """Everything except the pixels. Safe to log."""
        return {
            "frame_id": self.frame_id,
            "t": round(self.t, 3),
            "age_s": round(self.age_s, 2),
            "image": {"w": self.w, "h": self.h},
            "n_objects": len(self.objects),
            "quality": self.quality.to_dict(),
        }


class FrameRing:
    """One session's rolling window of frames."""

    def __init__(self, seconds: float = None, max_frames: int = None):
        self.seconds = config.RING_SECONDS if seconds is None else seconds
        self.max_frames = config.RING_MAX_FRAMES if max_frames is None else max_frames
        self._frames = deque()
        self._lock = threading.Lock()
        self._seq = 0
        self.tracker = scene_mod.SceneTracker()
        self.n_pushed = 0

    # -- writing -------------------------------------------------------------
    def push(self, jpeg: bytes, result: dict) -> Optional[RingFrame]:
        """Retain one frame, given the /headway_frame result that describes it.

        Called from the request handler AFTER the frame has been processed, so
        nothing here is inside the 250 ms budget's critical path; the cost is a
        reference copy and a dict.
        """
        if not jpeg or not result or result.get("ok") is False:
            return None
        objects = result.get("scene_objects") or []
        img = result.get("image") or {}
        t = float(result.get("t") or 0.0)
        self._seq += 1
        rf = RingFrame(
            frame_id=f"f{self._seq:06d}",
            t=t,
            wall_t=time.time(),
            jpeg=jpeg,
            w=int(img.get("w") or 0),
            h=int(img.get("h") or 0),
            objects=objects,
            ego=scene_mod.ego_from_result(result),
            quality=scene_mod.quality_from_result(result),
        )
        with self._lock:
            self._frames.append(rf)
            self._trim_locked(rf.wall_t)
            self.n_pushed += 1
        # Motion history is per session and must see EVERY frame, not just the
        # ones still in the window -- so it is updated here rather than being
        # recomputed from whatever survives.
        self.tracker.update(objects, t, v_host=result.get("v_host"))
        return rf

    def _trim_locked(self, now_wall: float) -> None:
        while len(self._frames) > self.max_frames:
            self._frames.popleft()
        while self._frames and (now_wall - self._frames[0].wall_t) > self.seconds:
            self._frames.popleft()

    # -- reading -------------------------------------------------------------
    def frames(self, max_age_s: float = None) -> list:
        """Newest last. Nothing older than max_age_s."""
        now = time.time()
        with self._lock:
            self._trim_locked(now)
            out = list(self._frames)
        if max_age_s is not None:
            out = [f for f in out if (now - f.wall_t) <= max_age_s]
        return out

    def latest(self) -> Optional[RingFrame]:
        fs = self.frames()
        return fs[-1] if fs else None

    def get(self, frame_id: str) -> Optional[RingFrame]:
        for f in self.frames():
            if f.frame_id == frame_id:
                return f
        return None

    def reset(self) -> None:
        with self._lock:
            self._frames.clear()
        self.tracker.reset()

    def stats(self) -> dict:
        fs = self.frames()
        return {
            "frames": len(fs),
            "pushed": self.n_pushed,
            "span_s": round(fs[-1].wall_t - fs[0].wall_t, 2) if len(fs) > 1 else 0.0,
            "bytes": sum(len(f.jpeg) for f in fs),
            "oldest_age_s": round(fs[0].age_s, 2) if fs else None,
        }


# ---------------------------------------------------------------------------
# Registry — one ring per session, dropped with the session
# ---------------------------------------------------------------------------
_rings: "OrderedDict[str, FrameRing]" = OrderedDict()
_rings_lock = threading.Lock()
_MAX_RINGS = 4          # one driver is the expected case; this is a leak guard


def get_ring(session_key: str) -> FrameRing:
    key = session_key or "default"
    with _rings_lock:
        ring = _rings.get(key)
        if ring is None:
            ring = FrameRing()
            _rings[key] = ring
            while len(_rings) > _MAX_RINGS:
                _rings.popitem(last=False)
        return ring


def peek_ring(session_key: str) -> Optional[FrameRing]:
    """The ring for this session if one exists — never creates one."""
    with _rings_lock:
        return _rings.get(session_key or "default")


def drop_ring(session_key: str) -> bool:
    with _rings_lock:
        ring = _rings.pop(session_key or "default", None)
    if ring is not None:
        ring.reset()
        return True
    return False


def active_rings() -> dict:
    with _rings_lock:
        keys = list(_rings.keys())
    return {k: _rings[k].stats() for k in keys if k in _rings}


# ---------------------------------------------------------------------------
# Optional persistence
# ---------------------------------------------------------------------------
def persist(session_id: str, name: str, jpeg: bytes) -> Optional[str]:
    """Write one image out, if and only if the operator turned that on.

    Returns the path written, or None. Callers store the return value verbatim,
    so an answer's record honestly says "no image was kept" when none was.
    """
    if not config.RING_PERSIST or not jpeg or not session_id:
        return None
    try:
        d = PERSIST_DIR / str(session_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name}.jpg"
        path.write_bytes(jpeg)
        return str(path)
    except Exception as e:
        print(f"[framebuf] could not persist {name}: {e}", flush=True)
        return None
