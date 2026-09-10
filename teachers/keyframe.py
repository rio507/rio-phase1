"""One instant of the drive, packaged identically for both teachers.

THE WHOLE POINT IS THAT THERE IS ONE OF THESE
---------------------------------------------
Two models are only comparable if they were shown the same thing. Not "the same
sort of thing" -- the same four JPEGs, the same t0, the same ego history, the
same words. So the window is built once, here, and handed to both clients; a
per-model preprocessing step is exactly the bug this file exists to prevent.

THE WINDOW
----------
Four frames at t0-0.3, t0-0.2, t0-0.1 and t0. That is not an arbitrary choice:
it is what Alpamayo's own dataset loader builds (num_frames=4, time_step=0.1)
and therefore the temporal spacing the model was trained to read. Cosmos gets
the same four as a short image sequence, so the two are looking at the same
0.3 s of road.

The ring does not contain frames at exactly those instants -- it contains
whatever the transport delivered, at 8-15 fps on the socket and about 4 fps on
the POST fallback. So each slot takes the NEAREST frame within a tolerance, and
the record carries how far off it was. At 10 fps the window is near-exact. At
4 fps it is not, and `exact: false` in the row is what stops a later reader
assuming otherwise.

WHY DUPLICATES ARE ALLOWED AND UNDER-FILLING IS NOT
---------------------------------------------------
At 4 fps two slots can resolve to the same frame. That is fine -- it is a real
description of what the camera had, both models see the same repeat, and the
comparison is unaffected. What is NOT fine is handing over three frames where
the model expects four, because the camera-name/frame-index prompt scaffolding
Alpamayo builds is indexed by position. So a window that cannot fill four slots
from distinct-or-repeated frames is refused, with a reason.
"""
import time

import config


def _pick(frames, target_wall, tolerance):
    """Nearest frame to an instant, or (None, error). Frames are newest last."""
    best, best_dt = None, None
    for f in frames:
        dt = abs(f.wall_t - target_wall)
        if best_dt is None or dt < best_dt:
            best, best_dt = f, dt
    if best is None:
        return None, None
    return best, best_dt


def build(session_key: str, ring, result: dict, trigger: str,
          ego: dict = None, spoken: dict = None, seq: int = 0) -> dict:
    """-> a keyframe dict, or {"ok": False, "reason": ...}.

    `ring` is the session's framebuf.FrameRing and `result` is the headway
    result for its newest frame. Nothing here mutates either: the ring is read
    through its own public accessor and the frames are held by reference, so
    building a keyframe costs a list of four pointers and some arithmetic.
    """
    n = int(getattr(config, "TEACHER_WINDOW_FRAMES", 4))
    step = float(getattr(config, "TEACHER_WINDOW_STEP_S", 0.1))
    tol = float(getattr(config, "TEACHER_WINDOW_SLOT_TOLERANCE_S", 0.14))
    max_age = float(getattr(config, "TEACHER_FRAME_MAX_AGE_S", 1.0))

    if ring is None:
        return {"ok": False, "reason": "no_ring"}
    frames = ring.frames()
    if not frames:
        return {"ok": False, "reason": "ring_empty"}

    newest = frames[-1]
    t0 = float(newest.wall_t)
    age = time.time() - t0
    # THE FIRST FRESHNESS GATE. A keyframe of a road the car has left is work
    # spent to produce a reading that will be refused on arrival, so it is
    # refused here instead -- for the same reason observer._tick refuses to
    # describe a stalled feed's last picture.
    if age > max_age:
        return {"ok": False, "reason": "stale_frame", "age_s": round(age, 2)}

    slots, worst, exact = [], 0.0, True
    for i in range(n):
        offset = -(n - 1 - i) * step          # -0.3, -0.2, -0.1, 0.0
        f, dt = _pick(frames, t0 + offset, tol)
        if f is None:
            return {"ok": False, "reason": "window_underfilled", "slot": i}
        if dt is not None:
            worst = max(worst, dt)
            if dt > tol:
                exact = False
        slots.append((offset, f))

    return {
        "ok": True,
        "kf_id": f"{session_key}-{seq:05d}",
        "seq": seq,
        "session_key": session_key,
        "t0": t0,
        "t0_session": float(result.get("t")) if result.get("t") is not None else None,
        "trigger": trigger,
        "frames": [f for _, f in slots],
        "frame_offsets_s": [round(o, 3) for o, _ in slots],
        "window": {
            "n": n,
            "step_s": step,
            "exact": exact,
            "spacing_error_ms": round(worst * 1000.0, 1),
            "frames": [{
                "frame_id": f.frame_id,
                "wall_t": round(f.wall_t, 4),
                "t": round(f.t, 4),
                "age_s": round(t0 - f.wall_t, 3),
                "slot_s": round(o, 3),
                "w": f.w, "h": f.h,
                "path": None,          # filled in by the corpus writer
            } for o, f in slots],
            "origin": getattr(ring, "origin", None),
        },
        "image": {"w": newest.w, "h": newest.h},
        "state": state_from(result),
        "tracks": [dict(o) for o in (result.get("scene_objects") or [])],
        "ego": ego,
        "spoken": spoken,
        "prompts": dict(getattr(config, "TEACHER_PROMPTS", {})),
        # Cosmos's fourth question. Carried on the keyframe rather than known
        # by the service so that ONE place in this repo decides what is asked,
        # and a corpus row records the words that were actually used.
        "physical_prompt": getattr(config, "TEACHER_COSMOS_PHYSICAL_PROMPT", ""),
    }


def state_from(result: dict) -> dict:
    """RIO's deterministic state at t0 — the corpus's ground-truth column.

    Copied out of the headway result verbatim rather than recomputed. If the
    band in a corpus row and the band in the drive's own JSONL ever disagree,
    that is a bug worth finding, and it can only be found if this is a copy.
    """
    speed = result.get("speed") or {}
    return {
        "band": result.get("band"),
        "gap_m": result.get("distance_m"),
        "ttc_s": result.get("ttc_s"),
        "tau_s": result.get("tau_s"),
        "trend": result.get("trend"),
        "speed_ms": speed.get("v_ms", result.get("v_host")),
        "speed_source": speed.get("source"),
        "speed_degraded": speed.get("degraded"),
        "lead_id": result.get("lead_id"),
        "corridor_source": result.get("corridor_source"),
        "lane_conf": result.get("lane_conf"),
    }


def to_job(kf: dict) -> dict:
    """A keyframe -> what a client puts on the wire. Pixels resolved here.

    This is where the JPEG bytes are actually reached for -- late, on the
    submitting thread, so a keyframe that is dropped before it is sent never
    touches them.
    """
    ego = kf.get("ego") or {}
    return {
        "kf_id": kf["kf_id"],
        "t0": kf["t0"],
        "submitted_at": time.time(),
        "jpegs": [f.jpeg for f in kf["frames"]],
        "frame_offsets_s": kf["frame_offsets_s"],
        "ego_xyz": ego.get("xyz"),
        "ego_yaw": ego.get("yaw"),
        "prompts": kf["prompts"],
        "physical_prompt": kf.get("physical_prompt") or "",
        "image": kf.get("image") or {},
    }
