"""grounding.py — what the eye is TOLD, as against what it is shown.

THE FAULT THIS ANSWERS
----------------------
Cosmos was guessing at space. Asked from a picture, it produced "car 20 km/h",
"the pickup truck 18 m ahead", "a gentle right curve" -- numbers and objects
with the grammar of measurements and none of the provenance. Every one of
those is something this car already MEASURES, on the same frame, with geometry
behind it: RF-DETR gives the box, Depth Anything gives the range, the corridor
gives the lane, the ECU gives the speed, the headway filter gives the gap and
the time to contact.

So stop asking the model for them. Give them to it, and let it do the thing a
2B reasoning model is actually good at -- deciding what matters and saying why.

THE ONE RULE ABOUT WHAT GOES IN HERE
------------------------------------
Only values that came from geometry or the ECU. Never anything a model
produced. That excludes the observer's own previous reading, the caption, any
label a VLM attached to anything, and -- the easy mistake -- RIO's own spoken
warning text, which is generated downstream of the band and would smuggle a
language model's wording back in as evidence. `band` itself is a state machine
output and is allowed; the sentence it produces is not.

THE HARDER RULE, ABOUT TIME
---------------------------
The measured state must cover the same stretch the video covers. A reading
grounded in stale numbers is worse than an ungrounded one: it looks checkable
and is not. The window is six seconds long and the model is given timestamps
inside it, so a claim about "two seconds ago" has to be answerable -- which
means the tracks travel as SERIES, sampled across the window on the window's
own clock, not as one snapshot of the final frame with a confident label on it.

WHAT THE MODEL MAY DO WITH ALL THIS
-----------------------------------
Reason about it. Not extend it. A number in the answer either matches a number
supplied here -- in which case it is `sourced`, and the card says so -- or it
came from nowhere and is stripped and counted exactly as an invented number is
today. `supplied_numbers` below is what that check runs against, and it is
built from the same dict that was rendered into the prompt so the two cannot
drift.
"""
import time

# HOW CLOSE A RESTATEMENT HAS TO BE. A model handed "18.7 m" may write "18.7",
# "19" or "about 19 metres" and mean the one we gave it; it may not write "25"
# and mean anything at all. Ten per cent or half a metre, whichever is larger,
# which is well inside the depth model's own error and well outside the gap
# between two different cars.
NUM_TOL_FRAC = 0.10
NUM_TOL_ABS = 0.5

# Unit families we accept a restatement in, and what one of them is in the unit
# the value was supplied as. A speed given in m/s may come back in km/h or mph
# without being an invention -- the model converted a number it was given, and
# calling that a fabrication would teach whoever reads the count to ignore it.
_SPEED_CONV = {"m/s": 1.0, "ms": 1.0, "mps": 1.0,
               "km/h": 1 / 3.6, "kmh": 1 / 3.6, "kph": 1 / 3.6,
               "mph": 0.44704}
_DIST_CONV = {"m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0,
              "meters": 1.0, "ft": 0.3048, "feet": 0.3048}
_TIME_CONV = {"s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0}


def measured_from_result(result: dict) -> dict:
    """The measured half of one headway frame, kept for the window. -> dict.

    Called at ring push, from the same result the detector just produced, so
    what is stored is what was measured on THOSE pixels rather than a value
    re-derived later from something else.

    Deliberately NOT scene.ego_from_result: that builds the conversation
    path's ego block and its docstring makes a promise -- no warning state in
    a visual answer -- which this would break. Two consumers, two shapes, and
    the promise stays where it was made.
    """
    if not isinstance(result, dict):
        return {}
    speed = result.get("speed") or {}
    return {
        "t": result.get("t"),
        "frame_t": result.get("frame_t"),
        # --- headway, from depth + the corridor + the filter ---------------
        "gap_m": result.get("distance_m"),
        "gap_invalid_reason": result.get("gap_invalid_reason"),
        "ttc_s": result.get("ttc_s"),
        "tau_s": result.get("tau_s"),
        "closing_ms": result.get("d_dot_ms"),
        "band": result.get("band"),
        "lead_id": result.get("lead_id"),
        # The plausibility veto, which is the field that says whether a range
        # was believed. A gap with a reject reason beside it is not a gap.
        "lead_depth_reject": result.get("lead_depth_reject"),
        "n_range_refused": result.get("n_range_refused"),
        "depth_trusted": result.get("depth_trusted"),
        "confidence": result.get("confidence"),
        # --- ego, from the ECU or the GPS ----------------------------------
        "v_host_ms": result.get("v_host"),
        "v_host_stale": result.get("v_host_stale"),
        "speed_source": speed.get("source") or result.get("speed_source"),
        "speed_age_s": speed.get("age_s"),
        "speed_degraded": result.get("speed_degraded"),
        # --- lane, from the lane net + the corridor ------------------------
        "corridor_source": result.get("corridor_source"),
        "lane_conf": result.get("lane_conf"),
        "n_lanes_detected": len(result.get("lanes") or []),
        "lane_plausible": list(result.get("lane_plausible") or []),
        "lane_offset": result.get("lane_offset"),
    }


# The filter's own words for "there is no gap", turned into the reader's.
# Keys are headway.live's `gap_invalid_reason`; an unknown key falls through
# to itself rather than to a guess, so a new reason shows up as jargon on the
# card and gets noticed, instead of being silently rendered as something else.
_GAP_REASONS = {
    "no_estimate": "no vehicle in the ego lane has been ranged yet",
    "no_lead": "no vehicle has been selected as the lead",
    "coasting": "the lead has been out of sight too long to report a gap",
    "track_lost": "the lead's track was lost",
    "depth_reject": "the range failed its size-plausibility check",
    "low_conf": "the range was not confident enough to report",
}


def _fmt(v, nd=1, unit=""):
    """A number the model can read back. -> str.

    THE TRAILING-ZERO STRIP ONLY APPLIES AFTER A DECIMAL POINT, and the first
    version of this did not check. `_fmt(90.0, 0)` produced "9": the ego speed
    of 25 m/s was rendered to the model as "(9 km/h)", and the model -- which
    had no way to know the number was ours to get wrong -- wrote a paragraph
    explaining that 9 km/h was implausibly slow for a highway. It was right.
    Grounding hands a model numbers it cannot check, so a formatting bug here
    does not produce a formatting complaint; it produces a confident, coherent
    reading built on a false premise, which is the worst failure this file has.
    """
    if v is None:
        return "unknown"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    s = f"{f:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return f"{s}{unit}"


def _track_series(window):
    """Every track held across the window, with its trajectory. -> list.

    Keyed by the detector's own track id, which is the identifier the model is
    asked to cite when it names a hazard. A track that appears in only one
    frame of twenty-four still gets an entry -- a pedestrian seen once is a
    fact about the window -- but the number of frames it was held in ships
    with it, because "seen once" and "tracked for six seconds" are different
    kinds of evidence and the model is being asked to weigh them.
    """
    by_id = {}
    for rf, t_rel in zip(window.ring_frames, window.t_rel):
        for o in (getattr(rf, "objects", None) or []):
            tid = o.get("id")
            if tid is None:
                continue
            e = by_id.setdefault(tid, {
                "id": tid, "label": o.get("label"), "samples": [],
                "in_lane": False, "is_lead": False, "vulnerable": False,
                "confirmed": False, "held_s": 0.0, "range_reject": None,
            })
            e["label"] = o.get("label") or e["label"]
            e["in_lane"] = bool(e["in_lane"] or o.get("member"))
            e["is_lead"] = bool(e["is_lead"] or o.get("is_lead"))
            e["vulnerable"] = bool(e["vulnerable"] or o.get("vulnerable"))
            e["confirmed"] = bool(e["confirmed"] or o.get("confirmed"))
            e["held_s"] = max(e["held_s"], float(o.get("age_s") or 0.0))
            if o.get("range_reject"):
                e["range_reject"] = o.get("range_reject")
            e["samples"].append({
                "t": round(float(t_rel), 2),
                "range_m": o.get("range_m"),
                "box": o.get("box"),
                "member": bool(o.get("member")),
                "lane_bounds": o.get("lane_bounds"),
            })
    for e in by_id.values():
        e["n_frames"] = len(e["samples"])
        e["first_t"] = e["samples"][0]["t"] if e["samples"] else None
        e["last_t"] = e["samples"][-1]["t"] if e["samples"] else None
        ranged = [s for s in e["samples"] if s["range_m"] is not None]
        e["first_range_m"] = ranged[0]["range_m"] if ranged else None
        e["last_range_m"] = ranged[-1]["range_m"] if ranged else None
        # CLOSING OR OPENING, FROM THE MEASUREMENTS THEMSELVES. This is the
        # single most useful thing a window carries that a still cannot, and
        # working it out here rather than asking the model to infer it from
        # two numbers is the difference between a fact and a guess.
        if len(ranged) >= 2:
            dt = ranged[-1]["t"] - ranged[0]["t"]
            dr = ranged[-1]["range_m"] - ranged[0]["range_m"]
            e["range_rate_ms"] = round(dr / dt, 2) if dt > 0.2 else None
        else:
            e["range_rate_ms"] = None
        # WHERE IT IS, said in words the model can use without doing geometry.
        # Derived from the box's centre against the lane bounds at the row the
        # object meets the road -- the same bounds membership already used, so
        # the card and the prompt cannot disagree about which lane a thing is
        # in. Unknown when the corridor had nothing to say there.
        e["side"] = _side_of(e["samples"][-1] if e["samples"] else None)
    out = sorted(by_id.values(),
                 key=lambda e: (e["last_range_m"] is None,
                                e["last_range_m"] if e["last_range_m"] is not None else 1e9))
    return out


def _side_of(sample):
    if not sample:
        return None
    box, bounds = sample.get("box"), sample.get("lane_bounds")
    if not box or not bounds:
        return None
    cx = (float(box[0]) + float(box[2])) / 2.0
    lo, hi = float(bounds[0]), float(bounds[1])
    if cx < lo:
        return "left of the ego lane"
    if cx > hi:
        return "right of the ego lane"
    return "in the ego lane"


def window_state(window) -> dict:
    """Everything measured over the same stretch the video covers. -> dict."""
    measured = [getattr(rf, "measured", None) or {} for rf in window.ring_frames]
    last = next((m for m in reversed(measured) if m), {})
    first = next((m for m in measured if m), {})
    series = []
    for m, t_rel in zip(measured, window.t_rel):
        if not m:
            continue
        series.append({
            "t": round(float(t_rel), 2),
            "gap_m": m.get("gap_m"),
            "ttc_s": m.get("ttc_s"),
            "band": m.get("band"),
            "v_host_ms": m.get("v_host_ms"),
        })
    return {
        "window": window.to_meta(),
        "tracks": _track_series(window),
        "headway": {
            "gap_m": last.get("gap_m"),
            "gap_invalid_reason": last.get("gap_invalid_reason"),
            "ttc_s": last.get("ttc_s"),
            "closing_ms": last.get("closing_ms"),
            "band": last.get("band"),
            "lead_id": last.get("lead_id"),
            "plausibility": (last.get("lead_depth_reject") or
                             ("accepted" if last.get("gap_m") is not None else "no lead")),
            "depth_trusted": last.get("depth_trusted"),
            "n_range_refused": last.get("n_range_refused"),
            "gap_first_m": first.get("gap_m"),
            "series": series,
        },
        "ego": {
            "speed_ms": last.get("v_host_ms"),
            "speed_source": last.get("speed_source"),
            "speed_age_s": last.get("speed_age_s"),
            "speed_stale": last.get("v_host_stale"),
            "speed_degraded": last.get("speed_degraded"),
            "speed_first_ms": first.get("v_host_ms"),
        },
        "lane": {
            "n_detected": last.get("n_lanes_detected"),
            "plausible": last.get("lane_plausible"),
            "conf": last.get("lane_conf"),
            "source": last.get("corridor_source"),
            "offset": last.get("lane_offset"),
        },
        "built_at": time.time(),
    }


def render(state: dict) -> str:
    """The measured block, as the model sees it. -> str, or "" if nothing.

    EVERY LINE NAMES ITS SOURCE. Not decoration: the model is about to be told
    it may reason about these numbers and may not invent others, and a block
    that reads like prose would invite it to treat the numbers as a style to
    continue. Naming the instrument on every line is what makes the block read
    as instrument output rather than as a paragraph.

    Written as plain lines rather than JSON. A 2B model handed JSON tends to
    answer in JSON, and what is wanted back is reasoning.

    WHAT IS SUMMARISED, AND WHY THE SERIES IS NOT PRINTED.
    The state carries a per-frame series for the gap, the time to contact and
    the ego speed, and a per-frame sample list for every track. None of it is
    rendered row by row. Twenty-four rows of gap and TTC in front of a 2B
    model is not more grounding, it is a table to complete -- this is the same
    model that answered a field template by completing the example in it. What
    IS rendered is the pair that makes a trend checkable: the value now and
    the value at the start of the window, with the direction named, plus each
    track's own first and last second so a claim about "two seconds back" has
    a span to land in.

    The series is not wasted. `supplied_numbers` reads it, so a reading that
    restates ANY gap the window contained is scored as sourced rather than
    invented -- the checking is at full resolution even though the prompt is
    not.
    """
    if not state:
        return ""
    L = []
    w = state.get("window") or {}
    span = w.get("span_s")
    L.append(f"MEASURED STATE for the {_fmt(span, 1)} s of video above, "
             f"timestamps on the same clock as the video's own markers.")

    ego = state.get("ego") or {}
    if ego.get("speed_ms") is not None:
        src = ego.get("speed_source") or "unknown source"
        stale = " (stale)" if ego.get("speed_stale") else ""
        # SAY WHOSE SPEED IT IS. Measured, not guessed at: given the bare line
        # "EGO SPEED 25 m/s", one reading called it "the highway's 25-meter/sec
        # speed limit" and another attached 90 km/h to a white sedan in front.
        # Both are a model filling in what the label did not say, and both
        # passed the numbers guard because the NUMBER was ours -- the guard
        # checks the value, never what the value was attached to.
        line = (f"EGO SPEED (this vehicle's own speed, not a limit and not "
                f"another vehicle's) {_fmt(ego['speed_ms'], 1, ' m/s')} "
                f"({_fmt(float(ego['speed_ms']) * 3.6, 0, ' km/h')}) "
                f"[source: {src}{stale}]")
        if ego.get("speed_first_ms") is not None:
            d = float(ego["speed_ms"]) - float(ego["speed_first_ms"])
            if abs(d) >= 0.5:
                line += (f", {'gained' if d > 0 else 'lost'} "
                         f"{_fmt(abs(d), 1, ' m/s')} across the window")
        L.append(line)
    else:
        L.append("EGO SPEED not available [source: none]")

    hw = state.get("headway") or {}
    if hw.get("gap_m") is not None:
        line = f"LEAD GAP {_fmt(hw['gap_m'], 1, ' m')} [source: depth + corridor geometry]"
        if hw.get("gap_first_m") is not None:
            d = float(hw["gap_m"]) - float(hw["gap_first_m"])
            if abs(d) >= 1.0:
                line += (f", {'opening' if d > 0 else 'closing'} — was "
                         f"{_fmt(hw['gap_first_m'], 1, ' m')} at the start of the window")
        L.append(line)
        if hw.get("ttc_s") is not None:
            L.append(f"TIME TO CONTACT {_fmt(hw['ttc_s'], 1, ' s')} "
                     f"[source: headway filter]")
        L.append(f"HEADWAY BAND {hw.get('band') or 'unknown'} "
                 f"[source: warning state machine], "
                 f"range plausibility: {hw.get('plausibility')}")
    else:
        # THE REASON IN WORDS, not the enum. The model is going to read this
        # line and reason about it; "no_estimate" is a token from the filter's
        # internals and reads as a thing rather than as an absence.
        why = _GAP_REASONS.get(hw.get("gap_invalid_reason"),
                               hw.get("gap_invalid_reason")
                               or "no vehicle has been selected as the lead")
        L.append(f"LEAD GAP none measured [source: depth + corridor geometry] — {why}")

    lane = state.get("lane") or {}
    if lane.get("n_detected"):
        plaus = lane.get("plausible") or []
        n_ok = sum(1 for p in plaus if p)
        L.append(f"LANES {lane['n_detected']} line(s) detected, {n_ok} judged "
                 f"plausible [source: lane net], corridor from "
                 f"{lane.get('source') or 'unknown'}, "
                 f"confidence {_fmt(lane.get('conf'), 2)}")
    else:
        L.append("LANES none detected [source: lane net]")

    # WHETHER THE PICTURE IS MOVING AT ALL, which is a fact about the frames
    # and needs no model to notice. Here because the model demonstrably does
    # NOT notice: given six seconds of one frame repeated, it wrote that a
    # sedan "maintains a constant speed and is in the same lane", inventing
    # motion in a still picture. A rolling window makes that fault DETECTABLE;
    # it does not make the model detect it.
    if w.get("blank"):
        L.append(f"CAMERA the frames carry almost no structure "
                 f"(luminance sd {_fmt(w.get('structure_std'), 1)}) "
                 f"[source: pixel statistics] — a covered lens, a dead camera "
                 f"or a black feed, not a dark road")
    elif w.get("static"):
        L.append(f"CAMERA the picture is not changing across this window "
                 f"(mean inter-frame difference {_fmt(w.get('motion'), 2)}) "
                 f"[source: pixel statistics] — the feed is frozen or the "
                 f"vehicle is stationary. Nothing in the video is in motion.")

    tracks = state.get("tracks") or []
    if not tracks:
        L.append("TRACKS none — the detector is holding no road users in this window.")
    else:
        L.append(f"TRACKS {len(tracks)} held by the detector across this window:")
        for t in tracks:
            bits = [f"  track {t['id']}: {t.get('label') or 'object'}"]
            if t.get("last_range_m") is not None:
                bits.append(f"range {_fmt(t['last_range_m'], 1, ' m')} "
                            f"[source: depth + size plausibility]")
            elif t.get("range_reject"):
                bits.append(f"no range ({t['range_reject']})")
            else:
                bits.append("no range measured")
            if t.get("side"):
                bits.append(t["side"])
            if t.get("range_rate_ms") is not None:
                rr = t["range_rate_ms"]
                bits.append(f"{'opening' if rr > 0 else 'closing'} at "
                            f"{_fmt(abs(rr), 1, ' m/s')}")
            bits.append(f"held {_fmt(t.get('held_s'), 1, ' s')}, "
                        f"seen in {t['n_frames']} frames "
                        f"({_fmt(t['first_t'], 1)}–{_fmt(t['last_t'], 1)} s)")
            if t.get("is_lead"):
                bits.append("this is the lead")
            if t.get("vulnerable"):
                bits.append("vulnerable road user")
            if not t.get("confirmed"):
                bits.append("too small to claim a range for")
            L.append(", ".join(bits))
    return "\n".join(L)


def supplied_numbers(state: dict) -> list:
    """Every number the model was given, with its unit family. -> list.

    What the numbers guard checks a reading against. Built from the state that
    was rendered, not from the rendered text, so a change to the wording of
    the block cannot silently widen or narrow what counts as sourced.
    """
    out = []

    def add(value, family, what):
        if value is None:
            return
        try:
            f = float(value)
        except (TypeError, ValueError):
            return
        out.append({"value": f, "family": family, "what": what})

    ego = state.get("ego") or {}
    add(ego.get("speed_ms"), "speed", "ego speed")
    add(ego.get("speed_first_ms"), "speed", "ego speed at window start")
    hw = state.get("headway") or {}
    add(hw.get("gap_m"), "distance", "lead gap")
    add(hw.get("gap_first_m"), "distance", "lead gap at window start")
    add(hw.get("ttc_s"), "time", "time to contact")
    add(hw.get("closing_ms"), "speed", "closing rate")
    for s in (hw.get("series") or []):
        add(s.get("gap_m"), "distance", "lead gap in window")
        add(s.get("ttc_s"), "time", "time to contact in window")
        add(s.get("v_host_ms"), "speed", "ego speed in window")
    for t in (state.get("tracks") or []):
        add(t.get("last_range_m"), "distance", f"track {t['id']} range")
        add(t.get("first_range_m"), "distance", f"track {t['id']} range")
        add(t.get("range_rate_ms"), "speed", f"track {t['id']} range rate")
        for s in (t.get("samples") or []):
            add(s.get("range_m"), "distance", f"track {t['id']} range in window")
    lane = state.get("lane") or {}
    add(lane.get("n_detected"), "count", "lane lines detected")
    # The window's own span and the timestamps inside it. A reading that says
    # "at 2 seconds" is citing the video's clock, which we gave it.
    w = state.get("window") or {}
    add(w.get("span_s"), "time", "window span")
    add(w.get("n_frames"), "count", "frames in window")
    for t in (state.get("tracks") or []):
        add(t.get("held_s"), "time", f"track {t['id']} held")
        for s in (t.get("samples") or []):
            add(s.get("t"), "time", "video timestamp")
    return out


def matches_supplied(value: float, unit: str, supplied: list):
    """Did this number come from us? -> the supplied entry, or None.

    Tries the unit families a value could have been converted into, so a gap
    we gave in metres and a model restated in feet is still a restatement.
    """
    unit = (unit or "").strip().lower().rstrip(".")
    families = []
    if unit in _DIST_CONV:
        families.append(("distance", _DIST_CONV[unit]))
    if unit in _SPEED_CONV:
        families.append(("speed", _SPEED_CONV[unit]))
    if unit in _TIME_CONV:
        families.append(("time", _TIME_CONV[unit]))
    if not families:
        families = [("count", 1.0)]
    for family, k in families:
        as_supplied = float(value) * k
        for s in supplied:
            if s["family"] != family:
                continue
            tol = max(abs(s["value"]) * NUM_TOL_FRAC, NUM_TOL_ABS)
            if abs(as_supplied - s["value"]) <= tol:
                return s
    return None


def track_ids(state: dict) -> set:
    return {t["id"] for t in (state.get("tracks") or [])}


def track_labels(state: dict) -> set:
    out = set()
    for t in (state.get("tracks") or []):
        lab = (t.get("label") or "").strip().lower()
        if lab:
            out.add(lab)
    return out
