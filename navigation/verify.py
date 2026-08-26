"""The visual verifier — the camera's one question, and its narrow answer.

    "Is this specific expected landmark clearly visible right now?"

That is all vision does in navigation. It does not locate the turn, it does not
compute intersection coordinates, it does not estimate where the turn is
relative to the landmark (the map did that, deterministically, in
landmarks.py), and it cannot alter a single field of the route.

WHAT IS REUSED, AND WHY NOTHING NEW WAS BUILT
---------------------------------------------
RIO already has a perception stack, and this file deliberately adds no second
one:

    framebuf.FrameRing   the last few seconds of road, already retained for
                         visual conversation. Verification reads the same
                         frames the detector already measured — no new capture
                         path, no second camera consumer.
    enrich.get_adapter() the local VLM, already resident and already the thing
                         that answers small closed questions about a frame.
                         `landmark()` is one more such question.
    headway.depth        Depth Anything V2 Metric-Small, already loaded for the
                         headway loop. Used here as a plausibility check only.

TEMPORAL PERSISTENCE IS THE POINT
---------------------------------
A landmark seen in one frame is a detection. A landmark held across several
frames, seconds apart, is an observation. Single-frame anchors are exactly the
ones that turn out to be a billboard, a reflection in a window, or a livery on
a passing truck — so verification always reads several frames and reports how
long the thing has been held and how fresh the last look was. The gates in
anchors.py then decide.

DEPTH IS A CONSISTENCY SIGNAL, NOT A MEASUREMENT (§15)
------------------------------------------------------
Depth is never asked "how far is the intersection". Route geometry already
knows. It is asked whether a thing claimed to be the Shell 40 m from the
maneuver is plausibly in front of the car at a plausible range — a check that
catches the gross contradictions and abstains on everything else.
"""
import threading
import time
from typing import List, Optional

import config

from . import anchors as anchors_mod


class LandmarkObserver:
    """Where observations come from. One method, so the whole perception stack
    can be replaced by a script in a test without touching a gate."""

    name = "none"

    def observe(self, session_key: str, candidates: List[dict],
                now: float) -> dict:
        """-> {anchor_id: observation}. Missing entries mean "not seen"."""
        return {}


class VisionObserver(LandmarkObserver):
    """The real one: recent frames, one VLM question each, depth for sanity."""

    name = "vision"

    def observe(self, session_key: str, candidates: List[dict], now: float) -> dict:
        import framebuf

        ring = framebuf.peek_ring(session_key) if session_key else None
        if ring is None:
            return {"_reason": "camera_unavailable"}
        frames = ring.frames(config.NAV_VERIFY_FRAME_MAX_AGE_S)
        if not frames:
            return {"_reason": "no_recent_frames"}

        # Newest first, thinned so two reads of effectively the same instant do
        # not count as two observations — persistence has to mean elapsed time,
        # not a burst.
        picked = []
        last_t = None
        for f in reversed(frames):
            if last_t is None or (last_t - f.wall_t) >= config.NAV_VERIFY_MIN_SPACING_S:
                picked.append(f)
                last_t = f.wall_t
            if len(picked) >= config.NAV_VERIFY_MAX_FRAMES:
                break
        if len(picked) < config.NAV_ANCHOR_MIN_OBSERVATIONS:
            return {"_reason": "not_enough_frames"}

        labels = [c.get("label", "") for c in candidates]
        adapter = _get_adapter()
        acc = {c["anchor_id"]: {
            "visible": False, "hits": 0, "identity": [], "clarity": [],
            "instances": 1, "first_seen_t": None, "last_seen_t": None,
            "side": None, "box": None, "box_frame": None,
        } for c in candidates}

        for frame in picked:
            try:
                reports = adapter.landmark(frame.jpeg, labels)
            except Exception as e:
                print(f"[nav] landmark verify failed: {type(e).__name__}: {e}", flush=True)
                continue
            for c, rep in zip(candidates, reports or []):
                a = acc[c["anchor_id"]]
                if not isinstance(rep, dict) or not rep.get("visible"):
                    continue
                # A reply that does not parse is read as "not visible", never
                # as "probably". An adapter is a third party — a stray string
                # where a number belongs must cost this landmark and nothing
                # else.
                ident = _number(rep.get("identity"))
                clarity = _number(rep.get("clarity"))
                if ident is None or clarity is None:
                    continue
                a["visible"] = True
                a["hits"] += 1
                a["identity"].append(ident)
                a["clarity"].append(clarity)
                a["instances"] = max(a["instances"], int(_number(rep.get("count")) or 1))
                a["last_seen_t"] = max(a["last_seen_t"] or 0.0, frame.wall_t)
                a["first_seen_t"] = min(a["first_seen_t"] or frame.wall_t, frame.wall_t)
                side = rep.get("side")
                if side in ("left", "right"):
                    a["side"] = side.upper()
                if rep.get("box") and a["box"] is None:
                    a["box"] = rep["box"]
                    a["box_frame"] = frame

        out = {}
        for c in candidates:
            a = acc[c["anchor_id"]]
            if not a["visible"]:
                out[c["anchor_id"]] = {"visible": False, "observations": 0,
                                       "tracking_duration_s": 0.0,
                                       "identity_confidence": 0.0,
                                       "visibility_confidence": 0.0,
                                       "instances": 0, "last_seen_t": None}
                continue
            ident = sum(a["identity"]) / len(a["identity"])
            clarity = sum(a["clarity"]) / len(a["clarity"])
            # Being seen in only some of the frames looked at is itself a
            # visibility statement: a sign that flickers in and out of view is
            # not one a driver can be told to turn at.
            hit_rate = a["hits"] / float(len(picked))
            out[c["anchor_id"]] = {
                "visible": True,
                "observations": a["hits"],
                "tracking_duration_s": max(0.0, (a["last_seen_t"] or 0.0) -
                                           (a["first_seen_t"] or 0.0)),
                "identity_confidence": round(ident, 3),
                "visibility_confidence": round(clarity * hit_rate, 3),
                "instances": a["instances"],
                "last_seen_t": a["last_seen_t"],
                "side": a["side"],
                "depth_m": _depth_for(a),
                "frames_examined": len(picked),
            }
        return out


def _number(value) -> Optional[float]:
    """A float, or None. Never an exception: see the caller."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return None


def _depth_for(acc_entry: dict) -> Optional[float]:
    """Median depth inside the reported box, when there is one.

    Abstains freely. No box, no depth model, a failed sample: all return None,
    and None is not a rejection — it only means this particular consistency
    check had nothing to say.
    """
    if not config.NAV_VERIFY_DEPTH_ENABLED:
        return None
    box, frame = acc_entry.get("box"), acc_entry.get("box_frame")
    if not box or frame is None:
        return None
    try:
        from headway import depth as depth_mod

        img = frame.frame_bgr()
        if img is None:
            return None
        dmap = depth_mod.depth_map(img)
        stats = depth_mod.roi_depth(dmap, box)
        if isinstance(stats, dict):
            for k in ("median_m", "median", "depth_m", "range_m"):
                if stats.get(k) is not None:
                    return float(stats[k])
            return None
        return float(stats) if stats is not None else None
    except Exception as e:
        print(f"[nav] depth check unavailable: {type(e).__name__}: {e}", flush=True)
        return None


_adapter_override = None
_observer: Optional[LandmarkObserver] = None
_lock = threading.Lock()


def _get_adapter():
    if _adapter_override is not None:
        return _adapter_override
    import enrich

    return enrich.get_adapter()


def set_adapter(adapter) -> None:
    """Tests point this at a scripted adapter instead of loading an 8B model."""
    global _adapter_override
    _adapter_override = adapter


def get_observer() -> LandmarkObserver:
    global _observer
    with _lock:
        if _observer is None:
            _observer = VisionObserver()
        return _observer


def set_observer(observer: Optional[LandmarkObserver]) -> None:
    """Swap the whole perception path — the harness's simulated landmark
    observations (§32) enter here, and everything downstream of this point is
    the code that ships."""
    global _observer
    _observer = observer


def verify(session_key: str, candidates: List[dict],
           now: Optional[float] = None) -> dict:
    """Candidates in, at most one VerifiedAnchor out.

    Returns {"anchor": {...}|None, "reason": str, "rejections": {...}}.

    `anchor: None` is the ordinary outcome and carries no error. Every reason
    it can be None — vision off, no camera, no recent frames, nothing visible,
    uncertain identity, unstable tracking, a duplicate in view, a relation that
    cannot be supported, a stale observation, an implausible depth — ends the
    same way: RIO says the canonical instruction, and the drive is unaffected.
    """
    now = time.time() if now is None else now
    if not config.NAV_VISION_ENABLED:
        return {"anchor": None, "reason": "vision_disabled", "rejections": {}}
    if not candidates:
        return {"anchor": None, "reason": "no_candidates", "rejections": {}}

    candidates = candidates[:config.NAV_LANDMARK_MAX_CANDIDATES_PER_MANEUVER]
    try:
        observations = get_observer().observe(session_key, candidates, now) or {}
    except Exception as e:
        print(f"[nav] observer failed: {type(e).__name__}: {e}", flush=True)
        return {"anchor": None, "reason": "observer_error", "rejections": {}}

    if "_reason" in observations:
        return {"anchor": None, "reason": observations["_reason"], "rejections": {}}

    passed, rejections = [], {}
    for c in candidates:
        obs = observations.get(c.get("anchor_id")) or {}
        ok, fails = anchors_mod.validate(c, obs, now)
        if ok:
            passed.append({"candidate": c, "observation": obs})
        else:
            rejections[c.get("anchor_id")] = fails

    best = anchors_mod.select(passed)
    if not best:
        return {"anchor": None,
                "reason": "no_candidate_passed" if rejections else "not_visible",
                "rejections": rejections}

    anchor = anchors_mod.build(best["candidate"], best["observation"], now)
    if anchor is None:
        return {"anchor": None, "reason": "relation_unsupported",
                "rejections": rejections}
    return {"anchor": anchor.to_dict(), "reason": "verified",
            "rejections": rejections,
            "observation": {k: best["observation"].get(k) for k in
                            ("observations", "tracking_duration_s", "instances",
                             "depth_m", "frames_examined")}}
