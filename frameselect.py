"""Frame selection — which of the last six seconds to actually look at.

Design ref: docs/visual_qa.md §4. The spec's rule, and the reason this file
exists at all: *never automatically the newest frame*.

The newest frame is simply the one that happened to arrive last. The frame worth
sending is the one where the thing being asked about is big, sharp, unclipped,
unoccluded and well exposed -- and for a vehicle being overtaken, or one the
driver noticed and then asked about, that frame is usually already a second or
two in the past.

TWO PASSES, BECAUSE MEASURING COSTS A DECODE
--------------------------------------------
Sharpness and exposure need the pixels, at roughly 5 ms a frame. Scoring all 24
retained frames that way would put 120 ms into every question for information
that only separates the top few. So: score everything on the cheap terms
(geometry, detector confidence, recency, glare), take a shortlist, decode only
those, and re-score with the measured terms folded in. The shortlist size is
config.FRAME_SHORTLIST.

WEIGHTS ARE PROVISIONAL
-----------------------
Same status as the headway ladder's thresholds: starting points chosen to be
argued with against real drives, not validated numbers. Every component is
reported in the result so a bad choice can be read back out of the log rather
than guessed at.
"""
import config
import scene as scene_mod

# --- object-frame scoring weights -------------------------------------------
W_SIZE = 0.30          # a bigger box carries more of the detail an answer needs
W_QUALITY = 0.25       # sharpness + exposure, once measured
W_CLEAR = 0.20         # not occluded by another box
W_SCORE = 0.10         # the detector's own confidence in this frame
W_WHOLE = 0.10         # not clipped by the frame edge
W_RECENT = 0.05        # ties go to the more recent frame

# A box whose longest side is this fraction of the frame height is as big as it
# needs to be; past here, more size is not more information.
SIZE_SATURATE_FRAC = 0.45
# Within this many pixels of an edge, a box is treated as cut off.
EDGE_SLOP_PX = 3.0


def _size_score(box, frame_h) -> float:
    if not frame_h:
        return 0.0
    longest = max(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))
    return min(1.0, (longest / float(frame_h)) / SIZE_SATURATE_FRAC)


def _whole_score(box, w, h) -> float:
    """1.0 if the object is fully inside the frame, falling off as it is cut.

    A vehicle sliced by the edge of the picture is the classic bad crop: half a
    car reads as a different car, and the missing half is exactly the part a
    model would use to identify it.
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    touching = sum([x1 <= EDGE_SLOP_PX, y1 <= EDGE_SLOP_PX,
                    x2 >= w - EDGE_SLOP_PX, y2 >= h - EDGE_SLOP_PX])
    return max(0.0, 1.0 - 0.5 * touching)


def _recency_score(frame, ref_wall, max_age_s) -> float:
    if ref_wall is None:
        return 1.0
    age = max(0.0, ref_wall - frame.wall_t)
    return max(0.0, 1.0 - age / max(max_age_s, 1e-6))


def score_object_frame(frame, cid, ref_wall=None, max_age_s=None,
                       measured=False) -> dict:
    """How good is this frame for talking about candidate `cid`? -> components.

    Returns None if the object is not in this frame at all -- which is most of
    what "the clearest recent frame CONTAINING it" means.
    """
    obj = frame.object_by_id(cid)
    if obj is None:
        return None
    max_age_s = config.FRAME_MAX_AGE_S if max_age_s is None else max_age_s
    box = obj["box"]
    w = frame.w or 1
    h = frame.h or 1

    others = [o["box"] for o in frame.objects if o.get("id") != cid]
    clear = 1.0 - scene_mod.occlusion_of(box, others)

    comp = {
        "size": _size_score(box, h),
        "clear": max(0.0, clear),
        "whole": _whole_score(box, w, h),
        "score": float(obj.get("score") or 0.0),
        "recent": _recency_score(frame, ref_wall, max_age_s),
        "quality": frame.quality.score() if measured else frame.quality.base(),
    }
    total = (W_SIZE * comp["size"] + W_QUALITY * comp["quality"]
             + W_CLEAR * comp["clear"] + W_SCORE * comp["score"]
             + W_WHOLE * comp["whole"] + W_RECENT * comp["recent"])
    comp["total"] = total
    return comp


def best_frame_for(ring, cid, ref_wall=None, max_age_s=None):
    """The clearest recent frame containing candidate `cid`. -> (frame, info).

    (None, info) when the ring holds no frame with that object in it, which is
    the honest answer when a vehicle has left the view -- the caller decides
    what to do about it, and in Phase A that is to fall back to the referent's
    stored frame.
    """
    max_age_s = config.FRAME_MAX_AGE_S if max_age_s is None else max_age_s
    frames = ring.frames(max_age_s=max_age_s)
    scored = []
    for f in frames:
        c = score_object_frame(f, cid, ref_wall, max_age_s, measured=False)
        if c is not None:
            scored.append((c["total"], f, c))
    if not scored:
        return None, {"reason": "not_in_ring", "n_frames": len(frames)}

    # Pass two: decode only the shortlist and re-score with sharpness/exposure.
    scored.sort(key=lambda s: -s[0])
    shortlist = scored[:max(1, config.FRAME_SHORTLIST)]
    rescored = []
    for _, f, _ in shortlist:
        f.measure()
        c = score_object_frame(f, cid, ref_wall, max_age_s, measured=True)
        if c is not None:
            rescored.append((c["total"], f, c))
    rescored.sort(key=lambda s: -s[0])
    total, frame, comp = rescored[0]

    return frame, {
        "reason": "ok",
        "n_frames": len(frames),
        "n_containing": len(scored),
        "n_measured": len(rescored),
        "chosen_frame_id": frame.frame_id,
        "chosen_age_s": round(frame.age_s, 2),
        "components": {k: round(v, 4) for k, v in comp.items()},
        # The point of the whole file, stated in the log: how much better the
        # chosen frame was than simply taking the newest one.
        "newest_frame_id": frames[-1].frame_id if frames else None,
        "was_newest": bool(frames and frame.frame_id == frames[-1].frame_id),
    }


def best_scene_frame(ring, ref_wall=None, max_age_s=None):
    """The best frame for a question about the whole scene. -> (frame, info).

    Different question from the per-object one: nothing is being singled out, so
    the terms that matter are how sharp and well exposed the picture is and how
    much of the scene the detector could see in it. Recency counts for more here
    than it does for an object -- "what do you see" is a question about now.
    """
    max_age_s = config.FRAME_MAX_AGE_S if max_age_s is None else max_age_s
    frames = ring.frames(max_age_s=max_age_s)
    if not frames:
        return None, {"reason": "empty_ring", "n_frames": 0}

    def cheap(f):
        return f.quality.base() + 0.3 * _recency_score(f, ref_wall, max_age_s)

    shortlist = sorted(frames, key=cheap, reverse=True)[:max(1, config.FRAME_SHORTLIST)]
    best, best_total, best_comp = None, -1.0, None
    for f in shortlist:
        f.measure()
        comp = {
            "quality": f.quality.score(),
            "recent": _recency_score(f, ref_wall, max_age_s),
            "coverage": min(1.0, len(f.objects) / 4.0),
        }
        total = 0.6 * comp["quality"] + 0.3 * comp["recent"] + 0.1 * comp["coverage"]
        comp["total"] = total
        if total > best_total:
            best, best_total, best_comp = f, total, comp

    return best, {
        "reason": "ok",
        "n_frames": len(frames),
        "n_measured": len(shortlist),
        "chosen_frame_id": best.frame_id,
        "chosen_age_s": round(best.age_s, 2),
        "components": {k: round(v, 4) for k, v in best_comp.items()},
        "newest_frame_id": frames[-1].frame_id,
        "was_newest": best.frame_id == frames[-1].frame_id,
    }
