"""What the detector is tracking right now, per session, for anyone who asks.

WHY THIS EXISTS
---------------
The resident eye and the detector look at the same road and never at each
other. Cosmos reads a frame and writes "TRAFFIC: none"; RF-DETR, on the same
frame, is tracking a pedestrian and a motorcycle. Both answers went to the
driver -- one as boxes on the picture, one as a sentence on the card and as the
evidence RIO composed from -- and nothing anywhere compared them. On
2026-09-21 that produced exactly that pair, and the reading was the half she
was told.

So the frame loop leaves a census behind: how many road users are being
tracked, of what classes, how long each has been held. It is a by-product of
work already done -- the same candidate set membership and the geometry
already ran on -- and it costs a dict per frame.

WHAT THIS IS NOT
----------------
It is NOT a second opinion inside the warning path. Nothing here is read by
headway/live.py, by the policy, or by anything that can raise a band or speak a
warning; it is written at the end of a frame and read by the reading path. The
direction is one-way on purpose: the detector already decides what it decides,
and a census that could feed back into it would be the same number arriving
twice.

KEYED BY THE VISUAL KEY, WHICH IS NOT THE HEADWAY KEY
-----------------------------------------------------
`headway_live` keys its sessions `session_id or "default"` while frames, the
observer and every visual answer are keyed by `_visual_key()` -- the drive's id
when there is one, the TAB's id when there is not. Those differ for exactly the
case this is for: a clip playing with no drive started. So the census is filed
under the VISUAL key, by the endpoint that knows both, which is also the key
the reading it will be compared against is filed under.
"""
import threading
import time

# A candidate has to have been held for this long before it may contradict a
# reading. One frame of RF-DETR on a shadow is not evidence that a model missed
# something; a box held for half a second through the tracker is.
MIN_HELD_S = 0.5

# ...and the census itself goes stale. A reading is compared against what the
# detector saw at about the same moment, not against the last thing it ever
# saw: on a page whose clip has ended, the final frame's tracks must not go on
# contradicting readings forever.
MAX_AGE_S = 3.0

_lock = threading.Lock()
_by_key = {}


def note(key, scene_objects, t=None) -> None:
    """File what this frame was tracking. Never raises into the frame loop."""
    try:
        t = float(t if t is not None else time.time())
        held = []
        for o in (scene_objects or []):
            if not isinstance(o, dict):
                continue
            age = o.get("age_s")
            held.append({
                "label": str(o.get("label") or "object"),
                "confirmed": bool(o.get("confirmed")),
                "vulnerable": bool(o.get("vulnerable")),
                "is_lead": bool(o.get("is_lead")),
                "age_s": float(age) if isinstance(age, (int, float)) else 0.0,
                "range_m": (float(o["range_m"])
                            if isinstance(o.get("range_m"), (int, float)) else None),
            })
        with _lock:
            _by_key[str(key or "default")] = {"at": t, "objects": held}
    except Exception:
        pass


def current(key, max_age_s: float = None) -> dict:
    """-> {"age_s", "objects", "tracked", "classes", ...} or {} when unusable.

    `tracked` is the list that may CONTRADICT a reading: confirmed, held for at
    least MIN_HELD_S. `objects` is everything, for a card that wants to say
    what was there without claiming the model missed it.
    """
    max_age_s = MAX_AGE_S if max_age_s is None else float(max_age_s)
    with _lock:
        rec = _by_key.get(str(key or "default"))
        rec = dict(rec) if rec else None
    if not rec:
        return {}
    age = time.time() - rec["at"]
    if age > max_age_s:
        return {}
    objects = rec["objects"]
    tracked = [o for o in objects
               if o["confirmed"] and o["age_s"] >= MIN_HELD_S]
    classes = {}
    for o in tracked:
        classes[o["label"]] = classes.get(o["label"], 0) + 1
    return {
        "age_s": round(age, 2),
        "objects": objects,
        "tracked": tracked,
        "n_tracked": len(tracked),
        "n_seen": len(objects),
        "classes": classes,
        "vulnerable": sorted({o["label"] for o in tracked if o["vulnerable"]}),
    }


def forget(key) -> bool:
    with _lock:
        return _by_key.pop(str(key or "default"), None) is not None


def clear() -> None:
    with _lock:
        _by_key.clear()
